"""Download + cache OLMoE router traces from HuggingFace."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

OLMOE_FILES = {
    "5000": "c4_results_5000.jsonl",
    "120000": "c4_results_120000.jsonl",
    "245000": "c4_results_245000.jsonl",
    "490000": "c4_results_490000.jsonl",
    "final": "c4_results.jsonl",
}
OLMOE_CHECKPOINTS = ["5000", "120000", "245000", "490000", "final"]

SEQ_LEN = 2048


def _cache_path(ckpt: str, n) -> Path:
    base = config.DATA_CACHE / "olmoe"
    base.mkdir(parents=True, exist_ok=True)
    if n is None:
        return base / f"{ckpt}_indices_full.npy"
    return base / f"{ckpt}_indices_n{n}.npy"


def load_checkpoint(ckpt: str, n_tokens: int | None, verbose: bool = True) -> np.ndarray:
    """Return expert indices for the first ``n_tokens`` tokens, or the whole file."""
    cache = _cache_path(ckpt, n_tokens)
    if cache.exists():
        return np.load(cache)
    if n_tokens is not None:
        full_cache = _cache_path(ckpt, None)
        if full_cache.exists():
            packed = np.load(full_cache)
            if packed.shape[0] >= n_tokens:
                return packed[:n_tokens]

    from huggingface_hub import HfFileSystem

    fname = OLMOE_FILES[ckpt]
    remote = f"datasets/{config.OLMOE_REPO}/{fname}"
    if verbose:
        if n_tokens is None:
            print(f"  streaming {remote} (all sequences)...", flush=True)
        else:
            print(f"  streaming {remote} (first {n_tokens:,} tokens)...", flush=True)

    fs = HfFileSystem()
    seqs_needed = None if n_tokens is None else (n_tokens + SEQ_LEN - 1) // SEQ_LEN
    chunks: list[np.ndarray] = []
    collected_seqs = 0

    with fs.open(remote, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            arr = np.asarray(obj["exp_ids"], dtype=np.int16)  # (2048, top_k, num_layers)
            arr = np.transpose(arr, (0, 2, 1))                # (2048, num_layers, top_k)
            chunks.append(arr)
            collected_seqs += 1
            if seqs_needed is not None and collected_seqs >= seqs_needed:
                break

    indices = np.concatenate(chunks, axis=0)
    if n_tokens is not None:
        indices = indices[:n_tokens]
    np.save(cache, indices)
    if verbose:
        print(f"  cached {indices.shape[0]:,} tokens -> {cache.name}", flush=True)
    return indices
