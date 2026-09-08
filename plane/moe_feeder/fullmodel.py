"""Whole-model FLAME MoE forward: freeze per-layer reservations, ASTRA-sim the rest."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .astrasim import run_analytical, write_switch_yaml
from .config import FeederConfig, flame_checkpoints, flame_layers
from .et import write_comm_et
from .loaders import load_tokens
from .matrix import build_matrix, maps, profile_window
from .policy import collapse_history, score_heldout

POLICIES = ("uniform", "mean", "p95", "worst")
BYTES = 2048
WINDOW = 4


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cfg(layer: str, n_tokens: int) -> FeederConfig:
    return FeederConfig(
        source="flame",
        model="flame-moe-290m",
        layer=layer,
        n_tokens=n_tokens,
        traffic_lens="membership",
        deflection="none",
        window=WINDOW,
        bytes_per_slot=BYTES,
    )


def _sim(M: np.ndarray, work: Path, name: str, network: Path) -> int:
    write_comm_et(M, str(work), name=name, bytes_per_slot=BYTES)
    prefix = str(work / "et" / name)
    return run_analytical(prefix, str(network))


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#222",
        "axes.grid": True,
        "grid.color": "#ddd",
        "grid.linewidth": 0.6,
        "font.size": 11,
        "axes.titlesize": 13,
        "legend.frameon": False,
    })


def run_fullmodel_astrasim(out: Path, n_tokens: int = 250_000) -> List[Dict]:
    out = Path(out)
    work = out / "astrasim"
    work.mkdir(parents=True, exist_ok=True)
    network = write_switch_yaml(work / "network_switch8.yml")

    layers = flame_layers("flame-moe-290m")
    ckpts = flame_checkpoints("flame-moe-290m")
    window = list(ckpts[:WINDOW])
    held = list(ckpts[WINDOW:])
    print("layers", layers)
    print("profile", window, "held-out", held)

    reservations: Dict[str, Dict[str, np.ndarray]] = {p: {} for p in POLICIES}
    for layer in layers:
        cfg = _cfg(layer, n_tokens)
        history, _ = profile_window(cfg, window)
        for name in POLICIES:
            reservations[name][layer] = collapse_history(history, name)
        print("profiled", layer)

    plan_rows = []
    for name in POLICIES:
        total = 0
        for layer in layers:
            E = reservations[name][layer]
            tag = "{}_{}".format(name, layer.replace("layer_", "L"))
            disp = _sim(E, work, tag + "_disp", network)
            comb = _sim(E.T, work, tag + "_comb", network)
            layer_t = disp + comb
            total += layer_t
            plan_rows.append({
                "kind": "plan",
                "reservation": name,
                "layer": layer,
                "checkpoint": "frozen:{}-{}".format(window[0], window[-1]),
                "dispatch_cycles": disp,
                "combine_cycles": comb,
                "layer_cycles": layer_t,
                "reserved_slots": float(E.sum()),
            })
            print("  plan", name, layer, layer_t)
        print("plan total", name, total)

    actual_rows = []
    score_rows = []
    for ckpt in held:
        layer_cycles = {}
        for layer in layers:
            cfg = _cfg(layer, n_tokens)
            _, indices = load_tokens(cfg, ckpt)
            place, src = maps(cfg, len(indices))
            M = build_matrix(cfg, indices, place, src)
            tag = "act{}_{}".format(ckpt, layer.replace("layer_", "L"))
            disp = _sim(M, work, tag + "_disp", network)
            comb = _sim(M.T, work, tag + "_comb", network)
            layer_t = disp + comb
            layer_cycles[layer] = layer_t
            actual_rows.append({
                "kind": "actual",
                "reservation": "live",
                "layer": layer,
                "checkpoint": int(ckpt),
                "dispatch_cycles": disp,
                "combine_cycles": comb,
                "layer_cycles": layer_t,
                "actual_slots": float(M.sum()),
            })
            for name in POLICIES:
                env = score_heldout(M, reservations[name][layer])
                score_rows.append({
                    "checkpoint": int(ckpt),
                    "layer": layer,
                    "reservation": name,
                    **env,
                })
            print("  actual", ckpt, layer, layer_t)
        print("actual total", ckpt, sum(layer_cycles.values()))

    _write_csv(out / "plan_by_layer.csv", plan_rows)
    _write_csv(out / "actual_by_layer.csv", actual_rows)
    _write_csv(out / "heldout_overflow.csv", score_rows)

    plan_tot = {}
    for name in POLICIES:
        plan_tot[name] = int(sum(r["layer_cycles"] for r in plan_rows if r["reservation"] == name))
    actual_tot = {}
    for ckpt in held:
        actual_tot[str(ckpt)] = int(
            sum(r["layer_cycles"] for r in actual_rows if r["checkpoint"] == ckpt)
        )
    summary = {
        "model": "flame-moe-290m",
        "layers": layers,
        "profile_window": window,
        "held_out": held,
        "n_tokens": n_tokens,
        "bytes_per_slot": BYTES,
        "topology": "Switch 8x 100GB/s",
        "plan_total_cycles": plan_tot,
        "actual_total_cycles": actual_tot,
        "note": (
            "ASTRA-sim analytical congestion-aware. Whole-model comm = "
            "sum of 8 layers x (dispatch + combine). Plan sizes are the "
            "frozen reservation; actual sizes are the live membership matrix."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    drop_rows = score_drops(reservations, layers, held, n_tokens)
    _write_csv(out / "token_drops.csv", drop_rows)
    _draw(out, layers, held, plan_rows, actual_rows, plan_tot, actual_tot, drop_rows)
    print(json.dumps({"plan_total_cycles": plan_tot, "actual_total_cycles": actual_tot}, indent=2))
    return plan_rows + actual_rows


def membership_drops(indices: np.ndarray, place: np.ndarray, src: np.ndarray, E: np.ndarray) -> dict:
    """Admit top-k dests in order. Overflowing slots are dropped (no deflection)."""
    dests = place[indices]
    n, k = dests.shape
    sent = np.zeros_like(E, dtype=np.float64)
    any_drop = 0
    full_drop = 0
    slots_drop = 0
    slots = 0
    D = int(E.shape[0])
    for t in range(n):
        s = int(src[t])
        kept = 0
        lost = 0
        for j in range(k):
            d = int(dests[t, j])
            slots += 1
            if 0 <= s < D and 0 <= d < D and sent[s, d] + 1.0 <= E[s, d] + 1e-9:
                sent[s, d] += 1.0
                kept += 1
            else:
                slots_drop += 1
                lost += 1
        if lost:
            any_drop += 1
        if kept == 0:
            full_drop += 1
    return {
        "n_tokens": n,
        "frac_tokens_dropped": any_drop / n if n else 0.0,
        "frac_tokens_fully_dropped": full_drop / n if n else 0.0,
        "frac_slots_dropped": slots_drop / slots if slots else 0.0,
    }


def score_drops(reservations, layers, held, n_tokens: int) -> List[Dict]:
    rows = []
    for layer in layers:
        cfg = _cfg(layer, n_tokens)
        for ckpt in held:
            _, indices = load_tokens(cfg, ckpt)
            place, src = maps(cfg, len(indices))
            for name in POLICIES:
                stats = membership_drops(indices, place, src, reservations[name][layer])
                rows.append({
                    "checkpoint": int(ckpt),
                    "layer": layer,
                    "reservation": name,
                    **stats,
                })
        print("drops", layer)
    return rows


def run_overhead_and_drops(out: Path, n_tokens: int = 250_000) -> None:
    """Rebuild reservations and add overhead / drop charts without re-running ASTRA-sim."""
    out = Path(out)
    summary = json.loads((out / "summary.json").read_text())
    layers = summary["layers"]
    window = summary["profile_window"]
    held = summary["held_out"]
    plan_tot = {k: int(v) for k, v in summary["plan_total_cycles"].items()}
    actual_tot = {str(k): int(v) for k, v in summary["actual_total_cycles"].items()}

    reservations: Dict[str, Dict[str, np.ndarray]] = {p: {} for p in POLICIES}
    for layer in layers:
        cfg = _cfg(layer, n_tokens)
        history, _ = profile_window(cfg, window)
        for name in POLICIES:
            reservations[name][layer] = collapse_history(history, name)
        print("profiled", layer)

    drop_rows = score_drops(reservations, layers, held, n_tokens)
    _write_csv(out / "token_drops.csv", drop_rows)

    import csv as _csv
    plan_rows = list(_csv.DictReader((out / "plan_by_layer.csv").open()))
    actual_rows = list(_csv.DictReader((out / "actual_by_layer.csv").open()))
    for r in plan_rows:
        r["layer_cycles"] = int(float(r["layer_cycles"]))
    for r in actual_rows:
        r["layer_cycles"] = int(float(r["layer_cycles"]))
        r["checkpoint"] = int(r["checkpoint"])
    _draw(out, layers, held, plan_rows, actual_rows, plan_tot, actual_tot, drop_rows)


def _draw(out, layers, held, plan_rows, actual_rows, plan_tot, actual_tot, drop_rows=None) -> None:
    _style()
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"uniform": "#4a4a4a", "mean": "#2f6fed", "p95": "#c45c26", "worst": "#1a7f4b", "live": "#111"}

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    names = list(POLICIES)
    vals = [plan_tot[n] / 1e6 for n in names]
    bars = ax.bar(names, vals, color=[colors[n] for n in names])
    ax.set_ylabel("ASTRA-sim comm time (million cycles)")
    ax.set_title("Whole-model MoE forward — frozen reservation (8 layers × dispatch+combine)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, "{:.2f}".format(v),
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "plan_totals.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    labels = [str(c) for c in held]
    live = [actual_tot[str(c)] / 1e6 for c in held]
    ax.plot(range(len(held)), live, marker="o", color="#111", label="live held-out M")
    for name in POLICIES:
        ax.hlines(
            plan_tot[name] / 1e6, -0.2, len(held) - 0.8,
            colors=colors[name], linestyles="--",
            label="{} plan ({:.2f}M)".format(name, plan_tot[name] / 1e6),
        )
    ax.set_xticks(range(len(held)), labels, rotation=30)
    ax.set_xlabel("Held-out checkpoint")
    ax.set_ylabel("Whole-model comm (million cycles)")
    ax.set_title("Later iterations vs the frozen whole-model reservation")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "actual_vs_plan_over_training.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(layers))
    w = 0.25
    short = [l.replace("layer_", "L") for l in layers]
    p95 = [next(r["layer_cycles"] for r in plan_rows if r["reservation"] == "p95" and r["layer"] == ly) / 1e6 for ly in layers]
    last = held[-1]
    live_l = [next(r["layer_cycles"] for r in actual_rows if r["checkpoint"] == last and r["layer"] == ly) / 1e6 for ly in layers]
    mean_p = [next(r["layer_cycles"] for r in plan_rows if r["reservation"] == "mean" and r["layer"] == ly) / 1e6 for ly in layers]
    ax.bar(x - w, mean_p, w, label="mean plan", color=colors["mean"])
    ax.bar(x, p95, w, label="p95 plan", color=colors["p95"])
    ax.bar(x + w, live_l, w, label="live ckpt {}".format(last), color="#111")
    ax.set_xticks(x, short)
    ax.set_ylabel("Layer comm (million cycles)")
    ax.set_title("Per-layer ASTRA-sim comm — plan vs last held-out iteration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "per_layer_plan_vs_actual.png", dpi=160)
    plt.close(fig)

    live_avg = float(np.mean([actual_tot[str(c)] for c in held]))
    overhead = {n: 100.0 * (plan_tot[n] - live_avg) / live_avg for n in POLICIES}
    per_step = {n: [100.0 * (plan_tot[n] - actual_tot[str(c)]) / actual_tot[str(c)] for c in held] for n in POLICIES}

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    vals = [overhead[n] for n in POLICIES]
    bars = ax.bar(list(POLICIES), vals, color=[colors[n] for n in POLICIES])
    ax.axhline(0.0, color="#888", linewidth=1)
    ax.set_ylabel("Latency overhead vs mean live comm (%)")
    ax.set_title("How much extra whole-model comm each frozen plan budgets")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, "{:+.2f}%".format(v),
                ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "latency_overhead.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    for name in POLICIES:
        ax.plot(range(len(held)), per_step[name], marker="o", color=colors[name], label=name)
    ax.axhline(0.0, color="#888", linestyle=":", linewidth=1)
    ax.set_xticks(range(len(held)), [str(c) for c in held], rotation=30)
    ax.set_xlabel("Held-out checkpoint")
    ax.set_ylabel("Plan comm / live comm − 1 (%)")
    ax.set_title("Latency overhead of the frozen plan at each later iteration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "latency_overhead_over_training.png", dpi=160)
    plt.close(fig)

    if drop_rows:
        drop_avg = {}
        full_avg = {}
        slot_avg = {}
        drop_step = {n: [] for n in POLICIES}
        for name in POLICIES:
            xs = [r for r in drop_rows if r["reservation"] == name]
            drop_avg[name] = 100.0 * float(np.mean([r["frac_tokens_dropped"] for r in xs]))
            full_avg[name] = 100.0 * float(np.mean([r["frac_tokens_fully_dropped"] for r in xs]))
            slot_avg[name] = 100.0 * float(np.mean([r["frac_slots_dropped"] for r in xs]))
            for c in held:
                step = [r for r in xs if int(r["checkpoint"]) == int(c)]
                drop_step[name].append(100.0 * float(np.mean([r["frac_tokens_dropped"] for r in step])))

        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        x = np.arange(len(POLICIES))
        w = 0.38
        b1 = ax.bar(x - w / 2, [drop_avg[n] for n in POLICIES], w, label="tokens losing ≥1 expert slot", color="#c45c26")
        b2 = ax.bar(x + w / 2, [slot_avg[n] for n in POLICIES], w, label="expert-slots dropped", color="#4a4a4a")
        ax.set_xticks(x, list(POLICIES))
        ax.set_ylabel("Drop rate if overflow is discarded (%)")
        ax.set_title("If overflowing traffic is dropped — tokens vs slots")
        ax.legend()
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h, "{:.2f}".format(h),
                        ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / "token_drop_if_overflow_dropped.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.4, 4.6))
        for name in POLICIES:
            ax.plot(range(len(held)), drop_step[name], marker="o", color=colors[name], label=name)
        ax.set_xticks(range(len(held)), [str(c) for c in held], rotation=30)
        ax.set_xlabel("Held-out checkpoint")
        ax.set_ylabel("Tokens with at least one dropped expert slot (%)")
        ax.set_title("Token drop rate over later iterations (overflow discarded, no deflection)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "token_drop_over_training.png", dpi=160)
        plt.close(fig)

        (out / "overhead_drops_summary.json").write_text(json.dumps({
            "live_avg_cycles": live_avg,
            "latency_overhead_pct_vs_live": overhead,
            "token_drop_pct": drop_avg,
            "token_fully_dropped_pct": full_avg,
            "slot_drop_pct": slot_avg,
        }, indent=2) + "\n")
