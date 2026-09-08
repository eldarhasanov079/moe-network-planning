"""Run ASTRA-sim's analytical engine on a Chakra ET prefix."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_BIN = (
    WORKSPACE
    / "astra-sim"
    / "build"
    / "astra_analytical"
    / "build"
    / "bin"
    / "AstraSim_Analytical_Congestion_Aware"
)
DEFAULT_SYSTEM = (
    WORKSPACE / "astra-sim" / "examples" / "system" / "native_collectives" / "Ring_4chunks.json"
)
DEFAULT_REMOTE = (
    WORKSPACE
    / "astra-sim"
    / "examples"
    / "remote_memory"
    / "analytical"
    / "no_memory_expansion.json"
)


def write_switch_yaml(path: Path, npus: int = 8, bandwidth: float = 100.0, latency: float = 500.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "topology: [ Switch ]\nnpus_count: [ {} ]\nbandwidth: [ {} ]\nlatency: [ {} ]\n".format(
            npus, bandwidth, latency
        )
    )
    return path


def run_analytical(
    workload_prefix: str,
    network_yaml: str,
    *,
    binary: Optional[str] = None,
    system: Optional[str] = None,
    remote: Optional[str] = None,
) -> int:
    """Return max per-rank ``Comm time`` in ASTRA-sim cycles."""
    cmd = [
        str(binary or DEFAULT_BIN),
        "--workload-configuration={}".format(workload_prefix),
        "--system-configuration={}".format(system or DEFAULT_SYSTEM),
        "--network-configuration={}".format(network_yaml),
        "--remote-memory-configuration={}".format(remote or DEFAULT_REMOTE),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    times = [int(m) for m in re.findall(r"Comm time: (\d+)", text)]
    if not times:
        raise RuntimeError("no Comm time in ASTRA-sim output:\n{}".format(text[-1500:]))
    return max(times)


def matrix_key(phases: Sequence[np.ndarray], bandwidth: float, latency: float) -> str:
    digest = hashlib.sha256()
    digest.update("{:.6f}|{:.6f}|{}".format(bandwidth, latency, len(phases)).encode())
    for matrix in phases:
        arr = np.asarray(matrix, dtype=np.float64)
        digest.update(np.array(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.tobytes())
    return digest.hexdigest()[:20]


def run_phased(
    phases: Sequence[np.ndarray],
    work: Path,
    name: str,
    network: Path,
    *,
    bytes_per_slot: int,
    cache_dir: Optional[Path] = None,
    bandwidth: float = 100.0,
    latency: float = 500.0,
) -> int:
    """Write a sequential multi-phase ET and return max Comm time (cached)."""
    from .et import write_phased_et

    key = matrix_key(phases, bandwidth, latency)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        hit = cache_dir / "{}.json".format(key)
        if hit.is_file():
            try:
                return int(json.loads(hit.read_text())["cycles"])
            except (ValueError, KeyError):
                pass
    write_phased_et(phases, str(work), name=name, bytes_per_slot=bytes_per_slot)
    cycles = run_analytical(str(work / "et" / name), str(network))
    if cache_dir is not None:
        final = cache_dir / "{}.json".format(key)
        tmp = cache_dir / "{}.{}.tmp".format(key, name)
        tmp.write_text(json.dumps({"cycles": cycles, "name": name}) + "\n")
        os.replace(tmp, final)
    return cycles


def run_phased_jobs(jobs: Iterable[dict], workers: int = 6) -> List[dict]:
    """Run many ``run_phased`` jobs. Each job dict must include the kwargs."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs = list(jobs)
    if not jobs:
        return []
    if workers <= 1 or len(jobs) == 1:
        return [_execute_phased_job(job) for job in jobs]
    out = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_execute_phased_job, job): i for i, job in enumerate(jobs)}
        for future in as_completed(future_map):
            out[future_map[future]] = future.result()
    return out


def _execute_phased_job(job: dict) -> dict:
    cycles = run_phased(
        job["phases"],
        Path(job["work"]),
        job["name"],
        Path(job["network"]),
        bytes_per_slot=job["bytes_per_slot"],
        cache_dir=Path(job["cache_dir"]) if job.get("cache_dir") else None,
        bandwidth=job.get("bandwidth", 100.0),
        latency=job.get("latency", 500.0),
    )
    result = dict(job)
    result["cycles"] = cycles
    result.pop("phases", None)
    return result
