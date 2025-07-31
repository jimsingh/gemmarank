import argparse
import functools
from datetime import timedelta

import itertools
import math
import os
import random
import re
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from datasets import load_dataset, get_dataset_infos
from torch.utils.data import DataLoader, IterableDataset

from transformers import (
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel,
    T5EncoderModel,
    T5Tokenizer,
)

from gemmarank.dataset import make_loader
from gemmarank.rankt5_model import RankT5Enc, RankT5EncConfig, register_rankt5_model

torch.set_float32_matmul_precision('high')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="t5-large")
    parser.add_argument("--bs", type=int, default=104)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--wandb_project", type=str, default="rankt5-training")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()

def setup_wandb(args):
    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return None

    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    return run

def sanitize_filename(filename):
    return re.sub(r'-+', '-', re.sub(r'[<>:"/\\|?*]', '-', filename)).strip('-.')

def get_cosine_lr(step, warmup, max_steps, base_lr):
    if step < warmup:
        return base_lr * step / warmup

    progress = (step - warmup) / (max_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

def get_linear_lr(step, warmup, max_steps, base_lr):
    if step < warmup:
        return base_lr * step / warmup
    
    progress = (step - warmup) / (max_steps - warmup)
    return base_lr * (1 - progress)


def save_checkpoint(model, opt, step, save_dir="./checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f"step-{step}")
    os.makedirs(path, exist_ok=True)

    model.save_pretrained(path)
    torch.save({
        'optimizer_state_dict': opt.state_dict(),
        'step': step,
    }, os.path.join(path, 'training_state.pt'))

    # keep only last 2 checkpoints
    checkpoints = sorted([d for d in os.listdir(save_dir) if d.startswith("step-")],
                        key=lambda x: int(x.split("-")[1]))

    for old in checkpoints[:-2]:
        shutil.rmtree(os.path.join(save_dir, old))

def load_checkpoint(model, opt, path):
    model.load_state_dict(torch.load(os.path.join(path, 'pytorch_model.bin')))

    state = torch.load(os.path.join(path, 'training_state.pt'))
    opt.load_state_dict(state['optimizer_state_dict'])
    return state['step']


def validate_model(model, val_loader, device, args, tokenizer):
    model.eval()
    total_loss = 0
    total_sep = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs["loss"]
                sep = outputs["score_diff"].mean()

            total_loss += loss.item()
            total_sep += sep
            num_batches += 1
 
    model.train()
    return total_loss / num_batches, total_sep / num_batches

def log_parameter_drift(model, log_dict, initial_params):
   groups = {"embedding": 0, "encoder": 0}
   magnitudes = {"embedding": 0, "encoder": 0}

   for name, param in model.encoder.named_parameters():
       initial = initial_params[f'encoder.{name}']
       drift = torch.norm(param.data - initial).item()
       magnitude = torch.norm(initial).item()

       group = "embedding" if 'shared.weight' in name else "encoder"
       groups[group] += drift
       magnitudes[group] += magnitude

   dense_drift = torch.norm(model.dense.weight.data - initial_params['dense.weight'])
   dense_magnitude = torch.norm(initial_params['dense.weight'])

   log_dict.update({
       "dense_drift_frac": (dense_drift / dense_magnitude).item(),
       "encoder_drift_frac": groups["encoder"] / magnitudes["encoder"],
       "embedding_drift_frac": groups["embedding"] / magnitudes["embedding"],
   })

def log_step(step, args, avg_loss, elapsed, remaining, init_params, model, lrs):
    progress = step / args.steps

    elapsed_str = str(timedelta(seconds=int(elapsed))) 
    eta_str = str(timedelta(seconds=int(remaining))) 

    log_dict = {"loss": avg_loss, "step": step}
    log_dict.update(lrs)
    log_parameter_drift(model, log_dict, init_params)

    if wandb.run:
        wandb.log(log_dict)

    dense_lr = next((v for k, v in lrs.items() if 'dense' in k), 0)

    print(f"Step {step}/{args.steps} ({progress*100:.1f}%): loss={avg_loss:.4f}, LR_d={dense_lr:.1e}, elapsed={elapsed_str}, eta={eta_str}")

def train(model, loader, val_loader, opt, init_params, args, device, tokenizer, start_step=0):
    model.train()
    base_lrs = {group['name']: group['lr'] for group in opt.param_groups}
    os.makedirs("./checkpoints", exist_ok=True)

    step = start_step
    loss_sum = sim_sum = viol_sum = 0
    start_time = time.time()

    for batch in loader:
        
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs["loss"]

        loss.backward()

        lrs = {}
        for group in opt.param_groups:
            if 'dense' in group['name']:
                group['lr'] = get_linear_lr(step, args.warmup, args.steps, base_lrs[group['name']])
            elif step < args.warmup:
                group['lr'] = base_lrs[group['name']] * step / args.warmup
            
            lrs[f"lr_{group['name']}"] = group['lr']


        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        opt.zero_grad()

        loss_sum += loss.item()
        step += 1

        if step % args.log_steps == 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / (step - start_step)) * (args.steps - step) if step > start_step else 0

            avg_loss = loss_sum / args.log_steps
            log_step(step, args, avg_loss, elapsed, remaining, init_params, model, lrs)
            loss_sum = 0

        if step % args.save_steps == 0:
            val_loss, sep = validate_model(model, val_loader, device, args, tokenizer)
            if wandb.run:
                wandb.log({"val_loss": val_loss, "step": step, "separation": sep})
            #save_checkpoint(model, opt, step)

        if step >= args.steps:
            break

    return step, loss_sum / args.log_steps

def create_optimizer(model, config):
   no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
   embedding_params = [n for n, p in model.named_parameters() if 'shared.weight' in n]

   def get_params(selector):
       if callable(selector):
           match_fn = selector  # use provided lambda function
       elif isinstance(selector, list):
           match_fn = lambda n: n in selector  # match exact parameter names from list
       else:
           match_fn = lambda n: selector in n  # match substring in parameter name

       # separate parameters into decay/no_decay groups
       decay = [p for n, p in model.named_parameters() if match_fn(n) and not any(nd in n for nd in no_decay)]
       no_decay_params = [p for n, p in model.named_parameters() if match_fn(n) and any(nd in n for nd in no_decay)]
       return decay, no_decay_params

   groups = []
   specs = [
       ("dense", "dense", config.dense_lr),
       ("embedding", embedding_params, config.embedding_lr),
       ("encoder", lambda n: "encoder" in n and n not in embedding_params, config.encoder_lr),
   ]

   for name, selector, lr in specs:
       decay, no_decay_params = get_params(selector)
       if decay:
           groups.append({"params": decay, "weight_decay": 0.02, "lr": lr, "name": f"{name}_decay"})
       if no_decay_params:
           groups.append({"params": no_decay_params, "weight_decay": 0.0, "lr": lr, "name": f"{name}_no_decay"})

   return torch.optim.AdamW(groups, betas=(0.9, 0.97), fused=True)

def create_data_loaders(tokenizer, args):
    # --- Define constants and dataset info ---
    DS_BM25 = "sentence-transformers/msmarco-bm25"
    DS_COCONDENSER = "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1"
    SUBSET_TRIPLET_HARD  = "triplet-hard"
    VAL_SIZE = 50_000

    SUBSET_TRIPLET_HARD = "triplet-hard"
    SUBSET_TRIPLETS = "triplet-all"

    hard_ds_infos = get_dataset_infos(DS_COCONDENSER)[SUBSET_TRIPLET_HARD]
    hard_train_size = hard_ds_infos.splits['train'].num_examples - VAL_SIZE

    val_loader = make_loader(
        hf_name=DS_COCONDENSER, subset=SUBSET_TRIPLET_HARD, split=f"train[-{VAL_SIZE}:]",
        tokenizer=tokenizer, max_len=128, batch_size=args.bs,
        streaming=False, shuffle=False
    )

    loader_factory = functools.partial(
        make_loader,
        tokenizer=tokenizer,
        max_len=128, 
        batch_size=args.bs
    )

    loader_stage1 = loader_factory(hf_name=DS_BM25, subset=SUBSET_TRIPLETS, split="train")
    loader_stage2 = loader_factory(hf_name=DS_BM25, subset=SUBSET_TRIPLET_HARD, split="train") 
    loader_stage3 = loader_factory(hf_name=DS_COCONDENSER, subset=SUBSET_TRIPLET_HARD, split=f"train[:{hard_train_size}]") 

    steps_stage1 = int(0.20 * args.steps)
    steps_stage2 = int(0.40 * args.steps)

    curriculum_loader = itertools.chain(
        itertools.islice(itertools.cycle(loader_stage1), steps_stage1),
        itertools.islice(itertools.cycle(loader_stage2), steps_stage2),
        itertools.cycle(loader_stage3)
    )

    return curriculum_loader, val_loader

def main():
    args = get_args()
    device = torch.device("cuda")

    register_rankt5_model()

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    config = RankT5EncConfig(model_name=args.model, dense_lr=2e-4, encoder_lr=5e-5, embedding_lr=1e-6)
    model = RankT5Enc(config).to(device)

    if not args.wandb_run_name:
        args.wandb_run_name = f"{args.model}-bs{args.bs}-steps{args.steps}"

    setup_wandb(args)

    init_params = {n: p.data.clone() for n, p in model.named_parameters()}

    model = torch.compile(model)
    
    train_loader, val_loader = create_data_loaders(tokenizer, args)

    opt = create_optimizer(model, config)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, opt, args.resume)
    
    print("\nTraining Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {args.bs}")
    print(f"  Total steps: {args.steps}")
    print(f"  Warmup steps: {args.warmup}")
    print("\nOptimizer Groups:")
    for i, group in enumerate(opt.param_groups):
        print(f"  Group {i} ({group['name']}):")
        print(f"    Learning rate: {group['lr']:.2e}")
        print(f"    Weight decay: {group['weight_decay']}")
        print(f"    Num parameters: {sum(p.numel() for p in group['params'])}")
    print()

    final_step, final_loss = train(model, train_loader, val_loader, opt, init_params, args, device, tokenizer, start_step)

    final_path = f"./rankt5-{sanitize_filename(args.model)}-triplet-final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    if wandb.run:
        artifact = wandb.Artifact(f"rankt5-{sanitize_filename(args.model)}-triplet-final", type="model")
        artifact.add_dir(final_path)
        wandb.log_artifact(artifact)
        wandb.finish()

if __name__ == "__main__":
    main()
