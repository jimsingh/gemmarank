from itertools import cycle, islice
import re
import os
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from itertools import islice
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset

def _clean_columns(ds: Dataset) -> Dataset:
    needed = {"query", "positive", "negative"}
    return ds.select_columns(list(needed))

class TokenizingCollator:
    def __init__(self, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, examples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        if not examples: return {}

        q  = [e["query"]    for e in examples]
        dp = [e["positive"] for e in examples]
        dn = [e["negative"] for e in examples]

        pos_inputs = [f"Query: {qq}\nDocument: {pp}" for qq, pp in zip(q, dp)]
        neg_inputs = [f"Query: {qq}\nDocument: {nn}" for qq, nn in zip(q, dn)]

        tok = self.tokenizer(
            pos_inputs + neg_inputs,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        n = len(examples)
        return {
            "pos_ids":  tok["input_ids"][:n], "pos_mask": tok["attention_mask"][:n],
            "neg_ids":  tok["input_ids"][n:], "neg_mask": tok["attention_mask"][n:],
        }


def make_loader(
    hf_name: str, subset: str, split: str,
    tokenizer, max_len: int, batch_size: int,
    *,
    streaming: bool = True, shuffle: bool = False, seed: int = 42,
    buffer_size: int = 10_000, num_workers: int = 0, pin_memory: bool = True
) -> DataLoader:
    if num_workers > 0:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if streaming:
        match = re.match(r"(\w+)\[:(\d+)\]", split)
        if match:
            base_split, num_samples = match.groups()
            ds = load_dataset(hf_name, subset, split=base_split, streaming=True)
            ds = ds.take(int(num_samples))
        else:
            ds = load_dataset(hf_name, subset, split=split, streaming=True)
    else:
        ds = load_dataset(hf_name, subset, split=split, streaming=False)

    if shuffle and streaming:
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    ds = _clean_columns(ds)
    collator = TokenizingCollator(tokenizer, max_len)

    # For non-streaming, shuffling is handled by the DataLoader
    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collator,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=shuffle if not streaming else False,
    )
