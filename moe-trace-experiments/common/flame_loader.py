"""Stream + cache FLAME-MoE router traces from HuggingFace."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _remote_path(model: str, checkpoint: int, layer: str) -> str:
    """Build the in-repo parquet path (iteration dirs are zero-padded to 4)."""
    return f"{model}/actives/iter_{checkpoint:04d}/{layer}.parquet"


def _cache_paths(model: str, checkpoint: int, layer: str, n: int):
    base = config.DATA_CACHE / "flame" / model / f"iter_{checkpoint:04d}"
    base.mkdir(parents=True, exist_ok=True)
    scores = base / f"{layer}_scores_n{n}.npy"
    indices = base / f"{layer}_indices_n{n}.npy"
    return scores, indices


def load_layer(
    checkpoint: int,
    layer: str,
    n_rows: int,
    model: str | None = None,
    verbose: bool = True,
):
    """Return (scores, indices) for the first ``n_rows`` tokens of a file."""
    model = model or config.FLAME_MODEL
    scores_path, indices_path = _cache_paths(model, checkpoint, layer, n_rows)

    if scores_path.exists() and indices_path.exists():
        return np.load(scores_path), np.load(indices_path)

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    remote = f"datasets/{config.FLAME_REPO}/{_remote_path(model, checkpoint, layer)}"
    if verbose:
        print(f"  streaming {remote} (first {n_rows:,} rows)...", flush=True)

    scores_chunks: list[np.ndarray] = []
    indices_chunks: list[np.ndarray] = []
    collected = 0

    with fs.open(remote, "rb") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=65_536, columns=["scores", "indices"]):
            s = np.asarray(batch.column("scores").to_pylist(), dtype=np.float32)
            i = np.asarray(batch.column("indices").to_pylist(), dtype=np.int16)
            take = min(n_rows - collected, len(s))
            scores_chunks.append(s[:take])
            indices_chunks.append(i[:take])
            collected += take
            if collected >= n_rows:
                break

    scores = np.concatenate(scores_chunks, axis=0) if scores_chunks else np.empty((0, config.FLAME_TOP_K), np.float32)
    indices = np.concatenate(indices_chunks, axis=0) if indices_chunks else np.empty((0, config.FLAME_TOP_K), np.int16)

    np.save(scores_path, scores)
    np.save(indices_path, indices)
    if verbose:
        print(f"  cached {collected:,} rows -> {scores_path.name}", flush=True)
    return scores, indices


def iter_layer_indices(
    checkpoint: int,
    layer: str,
    model: str | None = None,
    batch_size: int = 65_536,
    verbose: bool = True,
):
    """Yield index batches for a full actives file. Does not cache the dump."""
    model = model or config.FLAME_MODEL
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    remote = f"datasets/{config.FLAME_REPO}/{_remote_path(model, checkpoint, layer)}"
    if verbose:
        print(f"  streaming {remote} (indices only, full file)...", flush=True)
    with fs.open(remote, "rb") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=batch_size, columns=["indices"]):
            yield np.asarray(batch.column("indices").to_pylist(), dtype=np.int16)
