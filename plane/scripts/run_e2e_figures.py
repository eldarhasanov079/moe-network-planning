"""Sample end-to-end: freeze a reservation, then score every later checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "plane") not in sys.path:
    sys.path.insert(0, str(ROOT / "plane"))

from moe_feeder.config import FeederConfig, flame_checkpoints, olmoe_checkpoints
from moe_feeder.loaders import load_tokens
from moe_feeder.matrix import build_matrix, maps, profile_window
from moe_feeder.planner import plan_matrix
from moe_feeder.policy import collapse_history, score_heldout
from moe_feeder.runtime import replay_against_plan

POLICIES = ("uniform", "mean", "p95", "worst")
BYTES = 2048


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
        "axes.labelsize": 11,
        "legend.frameon": False,
    })


def _save(fig, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=160)
    plt.close(fig)
    return dest


def _setup(source: str) -> tuple[FeederConfig, list, int, Path, str]:
    if source == "olmoe":
        cfg = FeederConfig(
            source="olmoe",
            model="olmoe",
            layer="layer_0",
            n_tokens=205_000,
            traffic_lens="membership",
            deflection="none",
            window=3,
            bytes_per_slot=BYTES,
        )
        ckpts = olmoe_checkpoints()
        window_n = 3
        out = ROOT / "plane" / "output" / "e2e" / "olmoe"
        xlabel = "Training checkpoint (OLMoE-1B-7B, layer 0)"
    else:
        cfg = FeederConfig(
            source="flame",
            model="flame-moe-290m",
            layer="layer_02",
            n_tokens=250_000,
            traffic_lens="membership",
            deflection="none",
            window=4,
            bytes_per_slot=BYTES,
        )
        ckpts = flame_checkpoints(cfg.model)
        window_n = 4
        out = ROOT / "plane" / "output" / "e2e"
        xlabel = "Training checkpoint (FLAME-MoE 290M, layer_02)"
    return cfg, ckpts, window_n, out, xlabel


def run(source: str) -> None:
    cfg, ckpts, window_n, out, xlabel = _setup(source)
    out.mkdir(parents=True, exist_ok=True)
    window = list(ckpts[:window_n])
    print(source, "profile window", window, "held-out", ckpts[window_n:])

    history, _ = profile_window(cfg, window)
    reservations = {name: collapse_history(history, name) for name in POLICIES}

    plans = {}
    mean = reservations["mean"]
    for name, reserved in reservations.items():
        planned = plan_matrix(
            reserved,
            cfg.topology,
            bytes_per_slot=BYTES,
            envelope_slot=name,
            out_dir=str(out / "plans" / name),
            route_slots=None if name == "mean" else mean,
        )
        plans[name] = {
            "iteration_time_s": planned["iteration_time_s"],
            "exposed_comm_s": planned["exposed_comm_s"],
            "ideal_iteration_time_s": planned["ideal_iteration_time_s"],
            "plan_wall_s": planned["plan_wall_s"],
            "n_flows": planned["n_flows"],
            "flow_bytes": planned["flow_bytes"],
            "reserved_slots": float(reserved.sum()),
        }
        print("planned", name, plans[name]["iteration_time_s"], "s")

    step_rows = []
    replay_rows = []
    last_actual = None
    for ckpt in ckpts:
        scores, indices = load_tokens(cfg, ckpt)
        place, src = maps(cfg, len(indices))
        actual = build_matrix(cfg, indices, place, src)
        last_actual = actual
        phase = "profile" if ckpt in window else "held-out"
        for name, reserved in reservations.items():
            env = score_heldout(actual, reserved)
            base = float(np.minimum(actual, reserved).sum())
            overflow = float(np.maximum(actual - reserved, 0).sum())
            refund = float(np.maximum(reserved - actual, 0).sum())
            step_rows.append({
                "checkpoint": str(ckpt),
                "phase": phase,
                "reservation": name,
                "actual_slots": float(actual.sum()),
                "reserved_slots": float(reserved.sum()),
                "covered_slots": base,
                "overflow_slots": overflow,
                "refund_slots": refund,
                **env,
            })
            replay = replay_against_plan(
                cfg, indices, scores, place, src, reserved, deflect=False
            )
            replay_rows.append({
                "checkpoint": str(ckpt),
                "phase": phase,
                "reservation": name,
                **replay.as_dict(),
            })
        print("scored", ckpt)

    _write_csv(out / "per_step.csv", step_rows)
    _write_csv(out / "replay.csv", replay_rows)
    (out / "plans.json").write_text(json.dumps(plans, indent=2) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "source": source,
        "model": cfg.model,
        "layer": cfg.layer,
        "n_tokens": cfg.n_tokens,
        "traffic_lens": cfg.traffic_lens,
        "profile_window": [str(c) for c in window],
        "held_out": [str(c) for c in ckpts[window_n:]],
        "bytes_per_slot": BYTES,
    }, indent=2) + "\n")

    if last_actual is not None:
        np.save(out / "last_actual.npy", last_actual)
        (out / "reservations").mkdir(parents=True, exist_ok=True)
        for name, reserved in reservations.items():
            np.save(out / "reservations" / "{}.npy".format(name), reserved)

    _style()
    figures = _draw(
        ckpts, window_n, step_rows, plans, reservations, last_actual,
        out / "figures", xlabel,
    )
    print(json.dumps({"out": str(out), "figures": [str(p) for p in figures]}, indent=2))


def _series(rows, policy, key):
    return [r[key] for r in rows if r["reservation"] == policy]


def _draw(ckpts, window_n, rows, plans, reservations, last_actual, fig_dir, xlabel):
    labels = [str(c) for c in ckpts]
    freeze_x = window_n - 0.5
    colors = {
        "uniform": "#4a4a4a",
        "mean": "#2f6fed",
        "p95": "#c45c26",
        "worst": "#1a7f4b",
    }
    saved = []

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for name in POLICIES:
        ax.plot(
            range(len(ckpts)),
            [100 * v for v in _series(rows, name, "overflow_ratio")],
            marker="o",
            color=colors[name],
            label=name,
        )
    ax.axvline(freeze_x, color="#888", linestyle="--", linewidth=1, label="plan frozen")
    ax.set_xticks(range(len(ckpts)), labels, rotation=30)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Overflow (% of actual dispatch slots)")
    ax.set_title("Each later iteration vs the frozen reservation")
    ax.legend(ncol=5, loc="upper right")
    saved.append(_save(fig, fig_dir / "overflow_over_training.png"))

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for name in POLICIES:
        ax.plot(
            range(len(ckpts)),
            [100 * v for v in _series(rows, name, "waste_ratio")],
            marker="o",
            color=colors[name],
            label=name,
        )
    ax.axvline(freeze_x, color="#888", linestyle="--", linewidth=1, label="plan frozen")
    ax.set_xticks(range(len(ckpts)), labels, rotation=30)
    ax.set_xlabel("Training checkpoint")
    ax.set_ylabel("Waste (% of reserved slots unused)")
    ax.set_title("Reserved-but-unused capacity after the plan is frozen")
    ax.legend(ncol=5, loc="upper right")
    saved.append(_save(fig, fig_dir / "waste_over_training.png"))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    for ax, name in zip(axes, ("mean", "p95")):
        covered = _series(rows, name, "covered_slots")
        overflow = _series(rows, name, "overflow_slots")
        ax.bar(range(len(ckpts)), covered, color="#2f6fed", label="fits the plan")
        ax.bar(range(len(ckpts)), overflow, bottom=covered, color="#c45c26", label="overflow")
        ax.axvline(freeze_x, color="#888", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(ckpts)), labels, rotation=40, fontsize=8)
        ax.set_title("{} reservation".format(name))
        ax.set_xlabel("Checkpoint")
    axes[0].set_ylabel("Dispatch slots")
    axes[1].legend(loc="upper right")
    fig.suptitle("What each iteration actually sent, split into planned vs overflow")
    saved.append(_save(fig, fig_dir / "covered_vs_overflow.png"))

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    names = list(POLICIES)
    times = [1000 * plans[n]["iteration_time_s"] for n in names]
    bars = ax.bar(names, times, color=[colors[n] for n in names])
    ax.set_ylabel("Planned iteration comm time (ms)")
    ax.set_title("Puppeteer Clos plan for the frozen matrix (one plan, reused)")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, t, "{:.2f}".format(t),
                ha="center", va="bottom", fontsize=10)
    saved.append(_save(fig, fig_dir / "planned_comm_time.png"))

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    actual = _series(rows, "mean", "actual_slots")
    ax.plot(range(len(ckpts)), actual, marker="o", color="#111", label="actual M")
    for name in POLICIES:
        reserved = reservations[name].sum()
        ax.hlines(
            reserved,
            xmin=-0.3,
            xmax=len(ckpts) - 0.7,
            colors=colors[name],
            linestyles="--",
            label="{} reserved ({:.0f})".format(name, reserved),
        )
    ax.axvline(freeze_x, color="#888", linestyle=":", linewidth=1)
    ax.set_xticks(range(len(ckpts)), labels, rotation=30)
    ax.set_xlabel("Training checkpoint")
    ax.set_ylabel("Total dispatch slots")
    ax.set_title("Actual traffic volume vs the one-shot reservation")
    ax.legend(loc="upper right", fontsize=8)
    saved.append(_save(fig, fig_dir / "volume_vs_reservation.png"))

    if last_actual is not None:
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8))
        last_ckpt = ckpts[-1]
        for ax, name in zip(axes, ("mean", "p95", "worst")):
            reserved = reservations[name]
            delta = last_actual - reserved
            vmax = max(abs(delta.min()), abs(delta.max()), 1.0)
            im = ax.imshow(delta, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title("{}: actual − reserved".format(name))
            ax.set_xlabel("dest rank")
            ax.set_ylabel("src rank")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Last checkpoint ({}) vs frozen pair reservations".format(last_ckpt))
        saved.append(_save(fig, fig_dir / "pair_delta_heatmaps.png"))

    return saved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("flame", "olmoe"), default="flame")
    args = parser.parse_args(argv)
    run(args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
