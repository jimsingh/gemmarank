# Standard library imports
import argparse
from functools import lru_cache
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
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel,
    T5EncoderModel,
    T5Tokenizer,
)
from sentence_transformers import SentenceTransformer

torch.set_float32_matmul_precision('high')

from gemmarank.rankt5_model import RankT5Enc, RankT5EncConfig, register_rankt5_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="t5-large")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--wandb_project", type=str, default="rankt5-training")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()

def setup_wandb(args):
    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return None

    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    wandb.config.update({
        "dense_lr": 2e-4, "embedding_lr": 5e-5, "encoder_lr": 5e-5,
        "positive_upsample_factor": 10, "max_length": 128,
    })
    return run

def sanitize_filename(filename):
    return re.sub(r'-+', '-', re.sub(r'[<>:"/\\|?*]', '-', filename)).strip('-.')

def pairwise_collator(batch):
    keys = ["pos_ids", "pos_mask", "neg_ids", "neg_mask"]
    return {key: torch.stack([item[key] for item in batch]) for key in keys}

def get_cosine_lr(step, warmup, max_steps, base_lr):
    if step < warmup:
        return base_lr * step / warmup

    progress = (step - warmup) / (max_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

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

class PairwiseDataset(IterableDataset):
    def __init__(self, tokenizer, max_len, is_validation=False, upsample=10, max_samples=None, device=None, seed=42, epochs=100):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.upsample = upsample
        self.max_samples = max_samples
        self.is_validation = is_validation
        split = "validation" if is_validation else "train"
        self.dataset = load_dataset("ms_marco", "v1.1", split=split, streaming=False)
        self.seed = seed
        self.epochs = epochs
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') 
        self.embedder.eval()
        self.device = device
        self.embedder = self.embedder.to(self.device)

    @lru_cache(maxsize=50_000_000)
    def _compute_similarity(self, query_doc_tuple):
        query, doc = query_doc_tuple
        with torch.no_grad():
            query_emb = self.embedder.encode(query, convert_to_tensor=True, device=self.device)
            doc_emb = self.embedder.encode(doc, convert_to_tensor=True, device=self.device)
            return F.cosine_similarity(query_emb.unsqueeze(0), doc_emb.unsqueeze(0)).item()
   
    def get_similarity(self, query, doc):
        return self._compute_similarity((query, doc))

    def _tokenize_pair(self, query, pos_doc, neg_doc):
        texts = [f"Query: {query}\nDocument: {pos_doc}", f"Query: {query}\nDocument: {neg_doc}"]
        tokens = self.tokenizer(texts, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        similarity = self.get_similarity(query, neg_doc)

        return {"pos_ids": tokens.input_ids[0], "pos_mask": tokens.attention_mask[0],
                "neg_ids": tokens.input_ids[1], "neg_mask": tokens.attention_mask[1],
                "neg_similarity": similarity}

    def _process_item(self, item):
        query, passages = item["query"], item["passages"]
        pos_docs = [d for d, l in zip(passages["passage_text"], passages["is_selected"]) if l]
        neg_docs = [d for d, l in zip(passages["passage_text"], passages["is_selected"]) if not l]

        if not pos_docs or not neg_docs:
            return []

        if self.is_validation:
            return [self._tokenize_pair(query, pos_doc, neg_doc)
                    for pos_doc in pos_docs for neg_doc in neg_docs]
        else:
            query_emb = self.embedder.encode([query])
            neg_embs = self.embedder.encode(neg_docs[:100])
            similarities = self.embedder.similarity(query_emb, neg_embs)[0]
            
            # Filter out very similar negatives (likely mislabeled)
            filtered_negs = [neg_docs[i] for i, sim in enumerate(similarities) if sim < 0.7]
            if len(filtered_negs) < 5: 
                filtered_negs = neg_docs

            count = min(self.upsample, len(filtered_negs))
            pairs = []
            for pos_doc in pos_docs:
                for neg_doc in random.sample(filtered_negs, count):
                    pairs.append(self._tokenize_pair(query, pos_doc, neg_doc))
            return pairs

    def __iter__(self):
        count = 0
        epochs = 1 if self.is_validation else self.epochs

        for epoch in range(epochs):
            seed = self.seed if self.is_validation else self.seed + epoch
            shuffled = self.dataset.shuffle(seed=seed)

            for item in shuffled:
                for pair in self._process_item(item):
                    if self.max_samples and count >= self.max_samples:
                        return
                    yield pair
                    count += 1

def validate_model(model, val_loader, device, args):
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

def log_step(step, args, avg_loss, elapsed, remaining, init_params, model):
    progress = step / args.steps
    elapsed_str = f"{int(elapsed//3600)}h{int((elapsed%3600)//60)}m"
    eta_str = f"{int(remaining//3600)}h{int((remaining%3600)//60)}m" if remaining > 0 else "N/A"

    log_dict = {"loss": avg_loss, "step": step}
    log_parameter_drift(model, log_dict, init_params)

    if wandb.run:
        wandb.log(log_dict)

    print(f"Step {step}/{args.steps} ({progress*100:.1f}%): loss={avg_loss:.4f}, elapsed={elapsed_str}, eta={eta_str}")

def train(model, loader, val_loader, opt, init_params, args, device, start_step=0):
    model.train()
    base_lrs = {group['name']: group['lr'] for group in opt.param_groups}
    os.makedirs("./checkpoints", exist_ok=True)

    step = start_step
    loss_sum = 0
    start_time = time.time()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(**batch)["loss"]

        loss.backward()

        for group in opt.param_groups:
            group['lr'] = get_cosine_lr(step, args.warmup, args.steps, base_lrs[group['name']])

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        opt.zero_grad()

        loss_sum += loss.item()
        step += 1

        if step % args.log_steps == 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / (step - start_step)) * (args.steps - step) if step > start_step else 0

            log_step(step, args, loss_sum / args.log_steps, elapsed, remaining, init_params, model)
            loss_sum = 0

        if step % args.save_steps == 0:
            val_loss, sep = validate_model(model, val_loader, device, args)
            if wandb.run:
                wandb.log({"val_loss": val_loss, "step": step, "separation": sep})
            save_checkpoint(model, opt, step)

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

def main():
    args = get_args()
    device = torch.device("cuda")

    register_rankt5_model()
    if not args.wandb_run_name:
        args.wandb_run_name = f"{args.model}-bs{args.bs}-steps{args.steps}"

    setup_wandb(args)

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    config = RankT5EncConfig(model_name=args.model)
    model = RankT5Enc(config).to(device)

    init_params = {n: p.data.clone() for n, p in model.named_parameters()}

    model = torch.compile(model)

    dataset = PairwiseDataset(tokenizer, max_len=128, device=device)
    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        collate_fn=pairwise_collator,
        num_workers=0,
        pin_memory=True
    )

    val_dataset = PairwiseDataset(tokenizer, max_len=128, is_validation=True, max_samples=args.val_samples)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        collate_fn=pairwise_collator,
        num_workers=0,
        pin_memory=True
    )

    opt = create_optimizer(model, config)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, opt, args.resume)

    final_step, final_loss = train(model, loader, val_loader, opt, init_params, args, device, start_step)

    final_path = f"./rankt5-{sanitize_filename(args.model)}-withsim_final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    if wandb.run:
        artifact = wandb.Artifact(f"rankt5-{sanitize_filename(args.model)}-final", type="model")
        artifact.add_dir(final_path)
        wandb.log_artifact(artifact)
        wandb.finish()

if __name__ == "__main__":
    main()
