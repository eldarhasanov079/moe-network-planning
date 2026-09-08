"""Causal cold-start experiment: uniform paths, profile, then reconfigure once."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .admit import admit_slots
from .astrasim import run_phased_jobs, write_switch_yaml
from .claim_v1 import BYTES, MatrixBank, _ckpt_key, source_bundle
from .config import DEFAULT_TOPOLOGY
from .planner import freeze_paths, plan_matrix
from .policy import collapse_history, score_heldout

ECMP_SEEDS = 32
SWITCH_BW_GBPS = 100.0
SWITCH_LATENCY_NS = 500.0
TOP_K = {"flame": 6, "olmoe": 8}
UNIFORM_CAPACITY_FACTORS = (1.0, 1.25)


def topology_only_uniform(bundle: Dict, capacity_factor: float = 1.0) -> np.ndarray:
    """Return an 8x8 row-uniform matrix without consulting a traffic trace."""
    ranks = 8
    tokens_per_source = float(bundle["n_tokens"]) / ranks
    slots_per_cell = tokens_per_source * TOP_K[bundle["name"]] / ranks
    return np.full((ranks, ranks), slots_per_cell * float(capacity_factor), dtype=np.float64)


def _write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _pair_time(
    matrix: np.ndarray,
    *,
    router_kind: str = "least-loaded",
    seed: int = 0,
    paths=None,
    reverse_paths=None,
) -> float:
    dispatch = plan_matrix(
        matrix,
        str(DEFAULT_TOPOLOGY),
        bytes_per_slot=BYTES,
        router_kind=router_kind,
        router_seed=seed,
        frozen_paths=paths,
    )
    combine = plan_matrix(
        matrix.T.copy(),
        str(DEFAULT_TOPOLOGY),
        bytes_per_slot=BYTES,
        router_kind=router_kind,
        router_seed=seed,
        frozen_paths=reverse_paths,
    )
    return float(dispatch["iteration_time_s"] + combine["iteration_time_s"])


def _build_profile_paths(
    bank: MatrixBank, bundle: Dict, repeats: int = 3
) -> Tuple[Dict[str, Tuple[dict, dict]], Dict]:
    """Build all per-layer direction maps and return a median wall-time measurement."""
    window = int(bundle["default_window"])
    means = {
        layer: np.mean(
            [bank.M(bundle, layer, c) for c in bundle["ckpts"][:window]], axis=0
        )
        for layer in bundle["layers"]
    }
    elapsed = []
    result: Dict[str, Tuple[dict, dict]] = {}
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        trial = {}
        for layer in bundle["layers"]:
            matrix = means[layer]
            trial[layer] = (
                freeze_paths(matrix, str(DEFAULT_TOPOLOGY), bytes_per_slot=BYTES),
                freeze_paths(matrix.T.copy(), str(DEFAULT_TOPOLOGY), bytes_per_slot=BYTES),
            )
        elapsed.append(time.perf_counter() - t0)
        result = trial
    return result, {
        "samples_s": elapsed,
        "median_s": float(np.median(elapsed)),
        "min_s": float(np.min(elapsed)),
        "max_s": float(np.max(elapsed)),
        "directions_planned": 2 * len(bundle["layers"]),
    }


def _path_signature(paths: dict) -> Tuple[Tuple[int, int, Tuple[str, ...]], ...]:
    return tuple(
        (int(src), int(dst), tuple(route.links))
        for (src, dst), route in sorted(paths.items())
    )


def _signature_hash(signature) -> str:
    return hashlib.sha256(repr(signature).encode()).hexdigest()[:12]


def _path_rows(
    bank: MatrixBank, bundles: Sequence[Dict]
) -> Tuple[List[Dict], Dict, Dict]:
    rows: List[Dict] = []
    plan_cost: Dict = {}
    route_audit: Dict = {}
    for bundle in bundles:
        name = bundle["name"]
        uniform = topology_only_uniform(bundle)
        uniform_paths = (
            freeze_paths(uniform, str(DEFAULT_TOPOLOGY), bytes_per_slot=BYTES),
            freeze_paths(uniform.T.copy(), str(DEFAULT_TOPOLOGY), bytes_per_slot=BYTES),
        )
        profiled_paths, cost = _build_profile_paths(bank, bundle)
        plan_cost[name] = cost
        uniform_signatures = tuple(_path_signature(paths) for paths in uniform_paths)
        distinct = {_signature_hash(signature) for signature in uniform_signatures}
        profiled_changed = 0
        current_changed = 0
        profiled_entries = 0
        current_entries = 0
        for layer in bundle["layers"]:
            for direction, paths in enumerate(profiled_paths[layer]):
                signature = _path_signature(paths)
                distinct.add(_signature_hash(signature))
                profiled_changed += sum(
                    left != right
                    for left, right in zip(uniform_signatures[direction], signature)
                )
                profiled_entries += len(signature)
            for checkpoint in bundle["ckpts"]:
                matrix = bank.M(bundle, layer, checkpoint)
                current_paths = (
                    freeze_paths(matrix, str(DEFAULT_TOPOLOGY), bytes_per_slot=BYTES),
                    freeze_paths(matrix.T.copy(), str(DEFAULT_TOPOLOGY), bytes_per_slot=BYTES),
                )
                for direction, paths in enumerate(current_paths):
                    signature = _path_signature(paths)
                    distinct.add(_signature_hash(signature))
                    current_changed += sum(
                        left != right
                        for left, right in zip(uniform_signatures[direction], signature)
                    )
                    current_entries += len(signature)
        route_audit[name] = {
            "off_diagonal_pairs_per_direction": len(uniform_signatures[0]),
            "profiled_route_entries_checked": profiled_entries,
            "profiled_route_entries_changed_from_uniform": profiled_changed,
            "current_matrix_route_entries_checked": current_entries,
            "current_matrix_route_entries_changed_from_uniform": current_changed,
            "distinct_path_table_hashes": sorted(distinct),
            "n_distinct_path_tables": len(distinct),
        }

        for checkpoint in bundle["ckpts"]:
            ckpt = _ckpt_key(checkpoint)
            for layer in bundle["layers"]:
                matrix = bank.M(bundle, layer, checkpoint)
                up = uniform_paths
                pp = profiled_paths[layer]
                deterministic = {
                    "uniform_frozen": _pair_time(matrix, paths=up[0], reverse_paths=up[1]),
                    "profiled_frozen": _pair_time(matrix, paths=pp[0], reverse_paths=pp[1]),
                    "oracle_least_loaded": _pair_time(matrix),
                    "spray": _pair_time(matrix, router_kind="spray"),
                }
                for strategy, duration in deterministic.items():
                    rows.append(
                        {
                            "source": name,
                            "layer": layer,
                            "checkpoint": ckpt,
                            "strategy": strategy,
                            "seed": 0,
                            "iteration_time_s": duration,
                        }
                    )
                for seed in range(ECMP_SEEDS):
                    rows.append(
                        {
                            "source": name,
                            "layer": layer,
                            "checkpoint": ckpt,
                            "strategy": "ecmp",
                            "seed": seed,
                            "iteration_time_s": _pair_time(
                                matrix, router_kind="ecmp", seed=seed
                            ),
                        }
                    )
            print("cold-start paths", name, ckpt, flush=True)
    return rows, plan_cost, route_audit


def _whole_model_rows(path_rows: Sequence[Dict], bundles: Sequence[Dict]) -> List[Dict]:
    totals: Dict[Tuple[str, str, str, int], float] = {}
    for row in path_rows:
        key = (
            row["source"],
            row["checkpoint"],
            row["strategy"],
            int(row["seed"]),
        )
        totals[key] = totals.get(key, 0.0) + float(row["iteration_time_s"])

    rows = [
        {
            "source": key[0],
            "checkpoint": key[1],
            "strategy": key[2],
            "seed": key[3],
            "whole_model_time_s": value,
        }
        for key, value in totals.items()
    ]

    for bundle in bundles:
        name = bundle["name"]
        window = int(bundle["default_window"])
        for t, checkpoint in enumerate(bundle["ckpts"]):
            ckpt = _ckpt_key(checkpoint)
            base = "uniform_frozen" if t < window else "profiled_frozen"
            match = next(
                r
                for r in rows
                if r["source"] == name
                and r["checkpoint"] == ckpt
                and r["strategy"] == base
                and r["seed"] == 0
            )
            rows.append(
                {
                    "source": name,
                    "checkpoint": ckpt,
                    "strategy": "cold_start_reconfig",
                    "seed": 0,
                    "whole_model_time_s": match["whole_model_time_s"],
                }
            )
    return rows


def _values(
    rows: Sequence[Dict], source: str, checkpoints: Sequence[str], strategy: str
) -> List[float]:
    checkpoint_set = set(checkpoints)
    return [
        float(row["whole_model_time_s"])
        for row in rows
        if row["source"] == source
        and row["strategy"] == strategy
        and row["checkpoint"] in checkpoint_set
    ]


def _path_summary(
    rows: Sequence[Dict], bundles: Sequence[Dict], plan_cost: Dict, route_audit: Dict
) -> Dict:
    summary = {"ecmp_seeds": ECMP_SEEDS, "sources": {}}
    strategies = (
        "cold_start_reconfig",
        "uniform_frozen",
        "profiled_frozen",
        "ecmp",
        "spray",
        "oracle_least_loaded",
    )
    for bundle in bundles:
        name = bundle["name"]
        window = int(bundle["default_window"])
        all_ckpts = [_ckpt_key(c) for c in bundle["ckpts"]]
        pre = all_ckpts[:window]
        post = all_ckpts[window:]
        block = {
            "profile_window": pre,
            "held_out": post,
            "reconfiguration_planning": plan_cost[name],
            "route_identity_audit": route_audit[name],
            "strategies": {},
            "by_checkpoint": {},
        }
        for strategy in strategies:
            entry = {}
            for phase, ckpts in (("all", all_ckpts), ("profile", pre), ("held_out", post)):
                vals = _values(rows, name, ckpts, strategy)
                entry[phase + "_mean_s"] = float(np.mean(vals))
                entry[phase + "_std_s"] = float(np.std(vals))
            block["strategies"][strategy] = entry
        for ckpt in all_ckpts:
            block["by_checkpoint"][ckpt] = {}
            for strategy in strategies:
                vals = _values(rows, name, [ckpt], strategy)
                block["by_checkpoint"][ckpt][strategy] = {
                    "mean_s": float(np.mean(vals)),
                    "std_s": float(np.std(vals)),
                }

        post_cold = block["strategies"]["cold_start_reconfig"]["held_out_mean_s"]
        comparisons = {}
        for strategy in ("uniform_frozen", "ecmp", "spray", "oracle_least_loaded"):
            other = block["strategies"][strategy]["held_out_mean_s"]
            comparisons["speedup_vs_" + strategy] = other / post_cold
        saving = (
            block["strategies"]["uniform_frozen"]["held_out_mean_s"] - post_cold
        )
        comparisons["saved_s_per_iteration_vs_uniform"] = saving
        if saving > 0:
            cpu = float(plan_cost[name]["median_s"])
            comparisons["payback_iterations_planner_cpu"] = cpu / saving
            comparisons["payback_iterations_cpu_plus_100ms_install"] = (cpu + 0.1) / saving
        else:
            comparisons["payback_iterations_planner_cpu"] = None
            comparisons["payback_iterations_cpu_plus_100ms_install"] = None
        block["held_out_comparisons"] = comparisons
        summary["sources"][name] = block
    return summary


def _enqueue_astrasim(
    jobs: List[Dict],
    metadata: List[Dict],
    matrices: Sequence[np.ndarray],
    *,
    work: Path,
    cache: Path,
    network: Path,
    group: str,
    source: str,
    checkpoint: str,
    arm: str,
) -> None:
    phase = 0
    for matrix in matrices:
        for direction in (matrix, matrix.T.copy()):
            name = "{}_p{}".format(group, phase)
            jobs.append(
                {
                    "phases": [direction],
                    "work": str(work / name),
                    "name": name,
                    "network": str(network),
                    "bytes_per_slot": BYTES,
                    "cache_dir": str(cache),
                    "bandwidth": SWITCH_BW_GBPS,
                    "latency": SWITCH_LATENCY_NS,
                }
            )
            metadata.append(
                {
                    "group": group,
                    "source": source,
                    "checkpoint": checkpoint,
                    "arm": arm,
                    "phase": phase,
                }
            )
            phase += 1


def _astrasim_rows(bank: MatrixBank, bundles: Sequence[Dict], out: Path) -> List[Dict]:
    work = out / "astrasim_work"
    work.mkdir(parents=True, exist_ok=True)
    network = write_switch_yaml(
        work / "network_switch8.yml",
        npus=8,
        bandwidth=SWITCH_BW_GBPS,
        latency=SWITCH_LATENCY_NS,
    )
    shared_cache = out.parent / "claim-v1" / "cache" / "astrasim"
    cache = shared_cache if shared_cache.is_dir() else out / "cache" / "astrasim"
    jobs: List[Dict] = []
    metadata: List[Dict] = []
    metric_rows: List[Dict] = []

    for bundle in bundles:
        source = bundle["name"]
        window = int(bundle["default_window"])
        uniform = {
            cf: topology_only_uniform(bundle, cf)
            for cf in UNIFORM_CAPACITY_FACTORS
        }
        p95 = {
            layer: collapse_history(
                [bank.M(bundle, layer, c) for c in bundle["ckpts"][:window]], "p95"
            )
            for layer in bundle["layers"]
        }
        for t, checkpoint in enumerate(bundle["ckpts"]):
            ckpt = _ckpt_key(checkpoint)
            actual = [bank.M(bundle, layer, checkpoint) for layer in bundle["layers"]]
            arms: Dict[str, Sequence[np.ndarray]] = {"live": actual}
            for cf, envelope in uniform.items():
                arms["uniform_cf{:g}".format(cf)] = [
                    admit_slots(matrix, envelope) for matrix in actual
                ]
            if t >= window:
                arms["profiled_p95"] = [
                    admit_slots(matrix, p95[layer])
                    for layer, matrix in zip(bundle["layers"], actual)
                ]

            for arm, admitted in arms.items():
                group = "{}_{}_{}".format(source, arm.replace(".", "p"), ckpt)
                _enqueue_astrasim(
                    jobs,
                    metadata,
                    admitted,
                    work=work,
                    cache=cache,
                    network=network,
                    group=group,
                    source=source,
                    checkpoint=ckpt,
                    arm=arm,
                )
                actual_total = float(sum(matrix.sum() for matrix in actual))
                admitted_total = float(sum(matrix.sum() for matrix in admitted))
                overflow = actual_total - admitted_total
                idle_num = 0.0
                idle_den = 0.0
                if arm.startswith("uniform"):
                    cf = float(arm.removeprefix("uniform_cf"))
                    for matrix in actual:
                        idle_num += float(np.maximum(uniform[cf] - matrix, 0).sum())
                        idle_den += float(uniform[cf].sum())
                elif arm == "profiled_p95":
                    for layer, matrix in zip(bundle["layers"], actual):
                        idle_num += float(np.maximum(p95[layer] - matrix, 0).sum())
                        idle_den += float(p95[layer].sum())
                metric_rows.append(
                    {
                        "source": source,
                        "checkpoint": ckpt,
                        "arm": arm,
                        "cycles": None,
                        "actual_slots": actual_total,
                        "admitted_slots": admitted_total,
                        "admitted_fraction": admitted_total / actual_total,
                        "overflow_ratio": overflow / actual_total,
                        "idle_ratio": idle_num / idle_den if idle_den else 0.0,
                    }
                )
            print("cold-start ASTRA queued", source, ckpt, flush=True)

    workers = min(6, os.cpu_count() or 2)
    print("cold-start ASTRA-sim jobs", len(jobs), "workers", workers, flush=True)
    finished = run_phased_jobs(jobs, workers=workers)
    cycles: Dict[str, int] = {}
    for meta, result in zip(metadata, finished):
        cycles[meta["group"]] = cycles.get(meta["group"], 0) + int(result["cycles"])
    for row in metric_rows:
        group = "{}_{}_{}".format(
            row["source"], row["arm"].replace(".", "p"), row["checkpoint"]
        )
        row["cycles"] = cycles[group]

    for bundle in bundles:
        source = bundle["name"]
        window = int(bundle["default_window"])
        for t, checkpoint in enumerate(bundle["ckpts"]):
            ckpt = _ckpt_key(checkpoint)
            arm = "uniform_cf1p25" if t < window else "profiled_p95"
            base_arm = arm.replace("p", ".") if arm.startswith("uniform") else arm
            base = next(
                row
                for row in metric_rows
                if row["source"] == source
                and row["checkpoint"] == ckpt
                and row["arm"] == base_arm
            )
            cold = dict(base)
            cold["arm"] = "cold_start_reconfig"
            metric_rows.append(cold)
    return metric_rows


def _astrasim_summary(rows: Sequence[Dict], bundles: Sequence[Dict]) -> Dict:
    summary = {
        "backend": "ASTRA-sim analytical congestion-aware Switch 8x100 GB/s",
        "path_aware": False,
        "sources": {},
    }
    for bundle in bundles:
        source = bundle["name"]
        window = int(bundle["default_window"])
        all_ckpts = [_ckpt_key(c) for c in bundle["ckpts"]]
        phases = {
            "all": set(all_ckpts),
            "profile": set(all_ckpts[:window]),
            "held_out": set(all_ckpts[window:]),
        }
        arms = sorted({r["arm"] for r in rows if r["source"] == source})
        block = {"arms": {}, "by_checkpoint": {}}
        for arm in arms:
            entry = {}
            for phase, ckpts in phases.items():
                selected = [
                    r
                    for r in rows
                    if r["source"] == source
                    and r["arm"] == arm
                    and r["checkpoint"] in ckpts
                ]
                if not selected:
                    continue
                for metric in ("cycles", "admitted_fraction", "overflow_ratio", "idle_ratio"):
                    entry[phase + "_mean_" + metric] = float(
                        np.mean([float(r[metric]) for r in selected])
                    )
            block["arms"][arm] = entry
        for ckpt in all_ckpts:
            block["by_checkpoint"][ckpt] = {
                r["arm"]: {
                    "cycles": int(r["cycles"]),
                    "admitted_fraction": float(r["admitted_fraction"]),
                    "overflow_ratio": float(r["overflow_ratio"]),
                    "idle_ratio": float(r["idle_ratio"]),
                }
                for r in rows
                if r["source"] == source and r["checkpoint"] == ckpt
            }
        summary["sources"][source] = block
    return summary


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def _draw(out: Path, path_summary: Dict, astra_summary: Dict) -> None:
    _style()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {
        "cold_start_reconfig": "#1f5f99",
        "uniform_frozen": "#666666",
        "profiled_frozen": "#61a5c2",
        "ecmp": "#c65f2f",
        "spray": "#248f57",
        "oracle_least_loaded": "#6f4c9b",
    }
    labels = {
        "cold_start_reconfig": "least-loaded (uniform = profiled = current-matrix)",
        "uniform_frozen": "uniform frozen",
        "profiled_frozen": "profiled from start (non-causal)",
        "ecmp": "ECMP (32-seed mean)",
        "spray": "spray",
        "oracle_least_loaded": "current-matrix least-loaded",
    }
    for source, block in path_summary["sources"].items():
        ckpts = block["profile_window"] + block["held_out"]
        x = np.arange(len(ckpts))
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        order = (
            "cold_start_reconfig",
            "ecmp",
            "spray",
        )
        for strategy in order:
            mean = [block["by_checkpoint"][c][strategy]["mean_s"] * 1e3 for c in ckpts]
            style = "-" if strategy in ("cold_start_reconfig", "ecmp", "spray") else "--"
            width = 2.6 if strategy == "cold_start_reconfig" else 1.4
            ax.plot(
                x,
                mean,
                marker="o" if strategy == "cold_start_reconfig" else None,
                linewidth=width,
                linestyle=style,
                color=colors[strategy],
                label=labels[strategy],
            )
            if strategy == "ecmp":
                std = [block["by_checkpoint"][c][strategy]["std_s"] * 1e3 for c in ckpts]
                ax.fill_between(
                    x,
                    np.asarray(mean) - np.asarray(std),
                    np.asarray(mean) + np.asarray(std),
                    color=colors[strategy],
                    alpha=0.12,
                    linewidth=0,
                )
        boundary = len(block["profile_window"]) - 0.5
        ax.axvline(boundary, color="#111111", linewidth=1.0)
        ax.text(
            boundary + 0.08,
            0.98,
            "profiling completes; quota updated",
            transform=ax.get_xaxis_transform(),
            va="top",
            fontsize=9,
        )
        ax.set_xticks(x, ckpts, rotation=30)
        ax.set_xlabel("Observed training checkpoint")
        ax.set_ylabel("Whole-model dispatch + combine (ms)")
        ax.set_title("Profiling leaves the least-loaded paths unchanged ({})".format(source.upper()))
        ax.legend(fontsize=9)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figures / "path_transition_{}.{}".format(source, suffix), dpi=180)
        plt.close(fig)

        order = (
            "cold_start_reconfig",
            "ecmp",
            "spray",
        )
        vals = [block["strategies"][s]["held_out_mean_s"] * 1e3 for s in order]
        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        bars = ax.bar(
            np.arange(len(order)), vals, color=[colors[s] for s in order], width=0.68
        )
        ax.set_xticks(
            np.arange(len(order)),
            ["least-loaded\n(all variants)", "ECMP", "spray"],
        )
        ax.set_ylabel("Held-out whole-model time (ms)")
        ax.set_title("Held-out routing time after the path-identity audit ({})".format(source.upper()))
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                "{:.0f}".format(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figures / "heldout_paths_{}.{}".format(source, suffix), dpi=180)
        plt.close(fig)

        astra = astra_summary["sources"][source]["arms"]
        arms = ("uniform_cf1", "uniform_cf1.25", "profiled_p95")
        overflow = [astra[a]["held_out_mean_overflow_ratio"] * 100 for a in arms]
        idle = [astra[a]["held_out_mean_idle_ratio"] * 100 for a in arms]
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
        names = ["uniform\n1.00×", "uniform\n1.25×", "profiled\nP95"]
        for ax, values, title, ylabel in (
            (axes[0], overflow, "Demand above quota", "Overflow slots (%)"),
            (axes[1], idle, "Reserved capacity left idle", "Idle reservation (%)"),
        ):
            bars = ax.bar(names, values, color=("#666666", "#9a9a9a", "#1f5f99"))
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    "{:.2f}".format(value),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        fig.suptitle("Quota trade-off after profiling ({})".format(source.upper()), fontweight="bold")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figures / "quota_tradeoff_{}.{}".format(source, suffix), dpi=180)
        plt.close(fig)


def _report(summary: Dict) -> str:
    path = summary["path_comparison"]
    astra = summary["astrasim"]
    lines = [
        "# Cold start: uniform paths, then one profiled reconfiguration",
        "",
        "## Question and method",
        "",
        "The system starts without router traces. It derives one row-uniform matrix from the known token count, top-k, and eight-rank job shape, and installs the corresponding least-loaded Clos paths. It profiles live source--destination matrices for the established window, then replaces the paths once with per-layer paths built from the observed mean. The same transition replaces the initial 1.25x uniform quota with a P95 quota.",
        "",
        "Puppeteer carries every live byte and compares paths on the 2-pod Clos. ASTRA-sim independently evaluates admitted sizes on an analytical 8x100 GB/s Switch; it cannot distinguish the Clos path policies.",
        "",
        "## Results",
        "",
        "| Trace | Reconfigured | Uniform frozen | ECMP | Spray | Current-matrix LL | vs uniform | vs ECMP | planner CPU | payback vs uniform |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, block in path["sources"].items():
        s = block["strategies"]
        c = block["held_out_comparisons"]
        payback = c["payback_iterations_planner_cpu"]
        lines.append(
            "| {source} | {cold:.1f} ms | {uniform:.1f} ms | {ecmp:.1f} ms | {spray:.1f} ms | {oracle:.1f} ms | {vu:.3f}x | {ve:.3f}x | {cpu:.3f} s | {payback} |".format(
                source=source,
                cold=s["cold_start_reconfig"]["held_out_mean_s"] * 1e3,
                uniform=s["uniform_frozen"]["held_out_mean_s"] * 1e3,
                ecmp=s["ecmp"]["held_out_mean_s"] * 1e3,
                spray=s["spray"]["held_out_mean_s"] * 1e3,
                oracle=s["oracle_least_loaded"]["held_out_mean_s"] * 1e3,
                vu=c["speedup_vs_uniform_frozen"],
                ve=c["speedup_vs_ecmp"],
                cpu=block["reconfiguration_planning"]["median_s"],
                payback="{:.1f} iterations".format(payback) if payback is not None else "no saving",
            )
        )
    lines += [
        "",
        "Times are held-out means of whole-model dispatch plus combine. ECMP is averaged over 32 seeds. Current-matrix least-loaded is a non-causal lower reference; spray is a fluid multipath reference and does not preserve a single path per pair.",
        "",
        "### Route-table identity audit",
        "",
        "| Trace | Profiled entries changed | Current-matrix entries changed | Distinct path tables |",
        "|---|---:|---:|---:|",
    ]
    for source, block in path["sources"].items():
        audit = block["route_identity_audit"]
        lines.append(
            "| {} | {} / {} | {} / {} | {} |".format(
                source,
                audit["profiled_route_entries_changed_from_uniform"],
                audit["profiled_route_entries_checked"],
                audit["current_matrix_route_entries_changed_from_uniform"],
                audit["current_matrix_route_entries_checked"],
                audit["n_distinct_path_tables"],
            )
        )
    lines += [
        "",
        "The audit compares every route entry with the topology-only uniform table. Each direction contains 56 off-diagonal source--destination routes.",
        "",
        "### Size-side check",
        "",
        "| Trace | Quota | Overflow | Idle reserve | ASTRA-sim admitted/live cycles |",
        "|---|---|---:|---:|---:|",
    ]
    for source, block in astra["sources"].items():
        arms = block["arms"]
        live = arms["live"]["held_out_mean_cycles"]
        for arm, label in (
            ("uniform_cf1", "uniform 1.00x"),
            ("uniform_cf1.25", "uniform 1.25x"),
            ("profiled_p95", "profiled P95"),
        ):
            entry = arms[arm]
            lines.append(
                "| {} | {} | {:.2f}% | {:.2f}% | {:.3f}x |".format(
                    source,
                    label,
                    entry["held_out_mean_overflow_ratio"] * 100,
                    entry["held_out_mean_idle_ratio"] * 100,
                    entry["held_out_mean_cycles"] / live,
                )
            )
    lines += [
        "",
        "A shorter admitted time is not a speedup: it means that quota overflow was omitted from this ASTRA-sim arm. The Puppeteer table above is the path result and carries the complete live matrices.",
        "",
        "## Interpretation",
        "",
        "On this dense 8x8 workload, reconfiguration changes the quota table but not the path table. Puppeteer's greedy Clos router balances the number of flows on each choice, rather than weighting occupancy by flow bytes. Every source--destination pair is present before and after profiling, so the uniform, profiled, and current-matrix runs produce the same 56-path table. The persistent advantage over ECMP comes from that deterministic balanced table; it does not require trace knowledge in this topology.",
        "",
        "The useful reconfiguration is therefore quota sizing. The 1.25x uniform quota is conservative, while P95 releases most of its idle reservation at the cost of some overflow. The experiment does not model packet queues, congestion control, path-install disruption, or training quality under quota overflow.",
        "",
        "Figures and machine-readable results are under `plane/output/cold-start-reconfig/`.",
        "",
    ]
    return "\n".join(lines)


def run_cold_start_reconfig(
    out: Path, sources: Sequence[str] = ("flame", "olmoe")
) -> Dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bundles = [source_bundle(source) for source in sources]
    bank = MatrixBank()
    for bundle in bundles:
        bank.load(bundle, keep_tokens=False)

    path_rows, plan_cost, route_audit = _path_rows(bank, bundles)
    whole_rows = _whole_model_rows(path_rows, bundles)
    path_summary = _path_summary(whole_rows, bundles, plan_cost, route_audit)
    astra_rows = _astrasim_rows(bank, bundles, out)
    astra_summary = _astrasim_summary(astra_rows, bundles)
    summary = {
        "experiment": "uniform cold start followed by one causal profiled reconfiguration",
        "path_comparison": path_summary,
        "astrasim": astra_summary,
        "method": {
            "uniform_input": "known token count, top-k, and rank count only",
            "profiled_paths": "per-layer mean over the causal profile window",
            "profiled_quota": "per-layer P95 over the same profile window",
            "path_traffic": "complete unadmitted live matrix",
            "bytes_per_slot": BYTES,
            "topology": str(DEFAULT_TOPOLOGY),
        },
    }
    _write_csv(out / "path_layers.csv", path_rows)
    _write_csv(out / "path_whole_model.csv", whole_rows)
    _write_csv(out / "astrasim.csv", astra_rows)
    _write_json(out / "summary.json", summary)
    _draw(out, path_summary, astra_summary)
    report = _report(summary)
    (out / "REPORT.md").write_text(report)
    reports = Path(__file__).resolve().parents[1] / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "cold-start-reconfig.md").write_text(report)
    print(json.dumps(summary, indent=2)[:5000])
    return summary
