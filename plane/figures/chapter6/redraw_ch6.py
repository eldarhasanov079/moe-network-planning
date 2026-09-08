"""Chapter 6 figures: one claim per file, PDF + SVG + PNG."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "moe-trace-experiments"))

from common import plotting  # noqa: E402

P = plotting.PALETTE
S2 = ROOT / "plane/output/stage2/summary.json"
HEAD = {"flame": 125_000, "olmoe": 102_400}
LAB = {"flame": "FLAME-MoE-290M", "olmoe": "OLMoE-1B-7B"}
SIZE = (6.4, 4.3)


def save(fig, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        plotting.save(fig, OUT / f"{stem}.{ext}", close=False)
    plt.close(fig)


def load_s2() -> dict:
    return json.loads(S2.read_text())


def leftover_vs_step(s: dict) -> None:
    g = s["granularity"]
    for src in ("flame", "olmoe"):
        per = g[src]["aligned"]
        Bs = sorted(int(b) for b in per)
        fig, ax = plt.subplots(figsize=SIZE)
        ax.plot(Bs, [per[str(B)]["p95_rate"]["overflow"] * 100 for B in Bs],
                "o-", color=P[0], lw=2.0, label="leftover, rate-sized $E$")
        ax.plot(Bs, [per[str(B)]["p95_step"]["overflow"] * 100 for B in Bs],
                "o--", color=P[0], lw=1.6, alpha=0.7, label="leftover, step-sized $E$")
        ax.plot(Bs, [per[str(B)]["p95_rate"]["waste"] * 100 for B in Bs],
                "s-", color=P[3], lw=2.0, label="idle, rate-sized $E$")
        ax.plot(Bs, [per[str(B)]["p95_step"]["waste"] * 100 for B in Bs],
                "s--", color=P[3], lw=1.6, alpha=0.7, label="idle, step-sized $E$")
        ax.axvline(HEAD[src], color="#666", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Tokens per step")
        ax.set_ylabel("% of slots")
        ax.set_title(f"Per-step leftover of a frozen P95 ({LAB[src]})")
        ax.legend(fontsize=8)
        save(fig, f"fig6_1_leftover_{src}")


def tightness(s: dict) -> None:
    t_all = s["tightness"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    marks = {"flame": "o", "olmoe": "s"}
    nudge = {
        ("flame", "mean_rate"): (6, 6),
        ("flame", "mean_step"): (6, -12),
        ("flame", "p95_rate"): (6, 4),
        ("flame", "p95_step"): (6, 4),
        ("flame", "worst_rate"): (6, -12),
        ("flame", "worst_step"): (6, -12),
        ("olmoe", "mean_rate"): (6, 6),
        ("olmoe", "mean_step"): (6, -12),
        ("olmoe", "p95_rate"): (6, 4),
        ("olmoe", "p95_step"): (6, 4),
        ("olmoe", "worst_rate"): (6, -12),
        ("olmoe", "worst_step"): (6, -12),
    }
    for src in ("flame", "olmoe"):
        t = t_all[src]
        for name, e in t.items():
            if name.startswith("mean") and name.endswith("step"):
                continue  # coincides with mean_rate
            face = P[0] if src == "flame" else P[1]
            m = "^" if name.endswith("step") else marks[src]
            ax.scatter(e["idle_reserve"] * 100, e["extension_rel"] * 100,
                       marker=m, s=70, color=face, zorder=3)
            short = name.replace("_", " ").replace("mean rate", "mean")
            dx, dy = nudge.get((src, name), (5, 3))
            ax.annotate(f"{src[0].upper()} {short}",
                        (e["idle_reserve"] * 100, e["extension_rel"] * 100),
                        textcoords="offset points", xytext=(dx, dy), fontsize=7, color=face)
    ax.set_xlabel("Idle reserve (% of $E$)")
    ax.set_ylabel("Tail past the reserved schedule (%)")
    ax.set_title("Reservation tightness: idle headroom vs tail lateness")
    ax.plot([], [], "o", color=P[0], label="FLAME, rate-sized")
    ax.plot([], [], "^", color=P[0], label="FLAME, step-sized")
    ax.plot([], [], "s", color=P[1], label="OLMoE, rate-sized")
    ax.plot([], [], "^", color=P[1], label="OLMoE, step-sized")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "fig6_1_tightness")


def leftover_worth(s: dict) -> None:
    sh = s["shedding"]["flame"][str(HEAD["flame"])]["p95"]
    arms = ["arrival-drop", "rank-drop", "score-drop"]
    labels = ["arrival\n(Stage 1)", "rank order", "score order"]
    drop = [sh[a]["weight_mass_dropped_per_token"] * 1e3 for a in arms]
    fig, ax = plt.subplots(figsize=SIZE)
    ax.bar(np.arange(3), drop, color=[P[8], P[1], P[0]], width=0.55)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Router weight lost per token ($\\times 10^{-3}$)")
    ax.set_title("Cost of dropping the leftover (FLAME-MoE-290M)")
    for i, v in enumerate(drop):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    save(fig, "fig6_2_drop_cost_flame")

    blk = s["shedding"]["olmoe"][str(HEAD["olmoe"])]["p95"]
    k = len(blk["arrival-drop"]["rank_hist_over"])
    x = np.arange(k)
    fig, ax = plt.subplots(figsize=SIZE)
    ax.bar(x - 0.2, np.array(blk["arrival-drop"]["rank_hist_over"]) * 100, 0.4,
           color=P[8], label="arrival order")
    ax.bar(x + 0.2, np.array(blk["rank-drop"]["rank_hist_over"]) * 100, 0.4,
           color=P[1], label="rank order")
    ax.set_xticks(x, [str(i + 1) for i in range(k)])
    ax.set_xlabel("Slot rank in the recorded top-8")
    ax.set_ylabel("% of leftover slots")
    ax.set_title("Where the leftover sits (OLMoE-1B-7B)")
    ax.legend(fontsize=8)
    save(fig, "fig6_2_leftover_rank_olmoe")


def deadline(s: dict) -> None:
    st = s["step"]
    for src in ("flame", "olmoe"):
        S = st[src]
        alloc = S.get("primary_allocator", "max-min")
        key = "aligned|{}|r={}".format(alloc, S["anchor_ratio"])
        curve = S["deadline"].get(key)
        if not curve:
            continue
        fracs = sorted(float(f) for f in curve)
        x = [f * 100 for f in fracs]
        fig, ax = plt.subplots(figsize=SIZE)
        if src == "flame":
            y = [(curve[str(f)]["orders"]["score_best"]["cost_per_token"] or 0) * 1e3 for f in fracs]
            ax.plot(x, y, "o-", color=P[0], lw=2.0, label="score order, lightest cut")
            y2 = [(curve[str(f)]["orders"]["arrival_random"]["cost_per_token"] or 0) * 1e3 for f in fracs]
            ax.plot(x, y2, "s--", color=P[8], lw=1.8, label="arrival order, random cut")
            stage1 = s["shedding"]["flame"][str(HEAD["flame"])]["p95"]["arrival-drop"]["weight_mass_dropped_per_token"]
            ax.axhline(stage1 * 1e3, color="#222", ls=":", lw=1, label="drop entire leftover")
            ax.set_ylabel("Router weight lost per token ($\\times 10^{-3}$)")
        else:
            y = [curve[str(f)]["orders"]["rank_best"]["dropped_frac_of_all_slots"] * 100 for f in fracs]
            ax.plot(x, y, "o-", color=P[1], lw=2.0, label="rank order, lightest cut")
            ax.set_ylabel("Slots cut (% of all top-8 slots)")
        ax.set_xlabel("Allowed lateness (% of reserved step)")
        ax.set_title(f"Sender-side deadline quota ({LAB[src]})")
        ax.legend(fontsize=8)
        save(fig, f"fig6_3_deadline_{src}")


def kernel_vs_r(s: dict) -> None:
    st = s["step"]
    for src in ("flame", "olmoe"):
        S = st[src]
        rs = sorted(float(r) for r in S["sweep"])
        fig, ax = plt.subplots(figsize=SIZE)
        xs = [max(r, 0.004) for r in rs]
        ax.plot(xs, [S["sweep"][str(r)]["ext_monolithic"] * 100 for r in rs],
                "o-", color=P[1], lw=2.0, label="monolithic kernel")
        ax.plot(xs, [S["sweep"][str(r)]["ext_chunked"] * 100 for r in rs],
                "s-", color=P[0], lw=2.0, label="chunked kernel")
        a = s["step"]["assumptions"][src]
        ax.axvline(a["ratio"], color="#222", ls=":", lw=1, label=f"anchor $r$={a['ratio']:.2f}")
        ax.set_xscale("log")
        ax.set_xlabel("Compute / wire time per slot ($r$)")
        ax.set_ylabel("Step time above reserved-only (%)")
        ax.set_title(f"Can the tail hide behind expert compute? ({LAB[src]})")
        ax.legend(fontsize=8)
        save(fig, f"fig6_4_kernel_{src}")


def step_bars(s: dict) -> None:
    st = s["step"]
    for src in ("flame", "olmoe"):
        S = st[src]
        tp = S["anchor"][S.get("primary_allocator", "max-min")]
        A = S["anchor"]
        labels = ["reserved\nonly", "+ tail\nmonolithic", "+ tail\nchunked", "all $M$\nplan", "all $M$\nECMP"]
        vals = [
            tp["reserved_only_s"] * 1e3,
            tp["monolithic_s"] * 1e3,
            tp["chunked_s"] * 1e3,
            A["controls"]["one_class_plan_s"] * 1e3,
            A["controls"]["full_ecmp_s"] * 1e3,
        ]
        fig, ax = plt.subplots(figsize=SIZE)
        ax.bar(np.arange(5), vals, color=[P[0], P[1], P[2], P[8], P[3]], width=0.6)
        ax.set_xticks(np.arange(5), labels, fontsize=8)
        ax.set_ylabel("Whole-model forward step (ms)")
        ax.set_title(f"Full step at the compute anchor ({LAB[src]})")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.14)
        save(fig, f"fig6_5_step_{src}")


def spare_resolved() -> None:
    models = ["FLAME\n290M", "FLAME\n721M", "FLAME\n1.7B*", "OLMoE"]
    arrival = [43, 43, 38, 31]
    score = [58, 59, 51, 58]
    two = [71, 69, 65, 56]
    x = np.arange(4)
    w = 0.25
    fig, ax = plt.subplots(figsize=SIZE)
    ax.bar(x - w, arrival, w, color=P[0], label="one spare, arrival")
    ax.bar(x, score, w, color=P[2], label="one spare, score/rank sort")
    ax.bar(x + w, two, w, color=P[4], label="two spares, arrival")
    ax.set_xticks(x, models)
    ax.set_ylabel("Overflow slots resolved (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Spare-expert deflection (top-$(k-1)$ emulation)")
    ax.legend(fontsize=8)
    save(fig, "fig6_6_spare_resolved")


def leftover_vs_mechanisms() -> None:
    labels = ["drop leftover\n(rate-sized $E$)", "one spare\n(emulated $k$)", "two spares", "deadline cut\n($L{=}0$)"]
    flame = [0.67, 0.42, 0.25, 0.085]
    olmoe = [2.19, 1.59, 1.09, 0.48]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=SIZE)
    ax.bar(x - 0.18, flame, 0.36, color=P[0], label="FLAME-290M")
    ax.bar(x + 0.18, olmoe, 0.36, color=P[1], label="OLMoE-1B-7B")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("Slots still dropped (%)")
    ax.set_title("Remaining drops (order of magnitude; $k$ differs for spare)")
    ax.legend(fontsize=8)
    save(fig, "fig6_7_remaining_drops")


def reconfig_stable() -> None:
    rows = [
        ("freeze / guard", 1.36, 3.5, 3.85, 7.3, (8, 4), (8, 6)),
        ("periodic slide", 1.44, 3.2, 4.03, 5.9, (8, 4), (-70, 8)),
        ("EMA", 0.34, 5.2, 2.01, 7.3, (8, 4), (8, -12)),
        ("margin 1.05", 0.31, 7.1, 2.34, 10.3, (8, -12), (8, 4)),
        ("oracle", 0.13, 6.0, 0.40, 12.8, (8, 4), (-40, 8)),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, fo, fi, oo, oi, fn, on in rows:
        ax.scatter(fi, fo, marker="o", s=70, color=P[0], zorder=3)
        ax.scatter(oi, oo, marker="s", s=70, color=P[1], zorder=3)
        ax.annotate(name, (fi, fo), textcoords="offset points", xytext=fn, fontsize=7, color=P[0])
        ax.annotate(name, (oi, oo), textcoords="offset points", xytext=on, fontsize=7, color=P[1])
    ax.set_xlabel("Idle reserve (% of $E$)")
    ax.set_ylabel("Per-step overflow (% of slots)")
    ax.set_title("Stable traces: overflow vs idle headroom")
    ax.plot([], [], "o", color=P[0], label="FLAME-290M")
    ax.plot([], [], "s", color=P[1], label="OLMoE-1B-7B")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(2.5, 15.0)
    save(fig, "fig6_8_reconfig_stable")


def collapse_timetable() -> None:
    xs = ["through\n2200", "3300\nsteps 1–3", "3300\nstep 4 (UP)", "4400\nsteps 1–2", "4400\nstep 3 (DOWN)"]
    ms = [63, 63, 114, 115, 55]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.step(np.arange(len(xs)), ms, where="mid", color=P[1], lw=2.2)
    ax.scatter(np.arange(len(xs)), ms, color=P[1], zorder=3)
    ax.set_xticks(np.arange(len(xs)), xs, fontsize=8)
    ax.set_ylabel("Installed timetable (ms), layer 02")
    ax.set_title("1.7B collapse: guard inflate then release")
    save(fig, "fig6_9_collapse_timetable")


def collapse_layer_times() -> None:
    layers = ["02", "03", "04", "05", "06", "07"]
    bound = [86, 62, 84, 88, 75, 74]
    tail = [116, 75, 110, 113, 87, 99]
    rec = [125, 82, 123, 128, 98, 108]
    x = np.arange(len(layers))
    w = 0.25
    fig, ax = plt.subplots(figsize=SIZE)
    ax.bar(x - w, bound, w, color=P[8], label="hot-NIC bound")
    ax.bar(x, tail, w, color=P[0], label="clean $E$ + Stage-2 tail")
    ax.bar(x + w, rec, w, color=P[1], label="reconfigured $E$")
    ax.set_xticks(x, layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer step (ms)")
    ax.set_title("Collapse: reconfig does not buy time (FLAME-1.7B, iter. 3300)")
    ax.legend(fontsize=8)
    save(fig, "fig6_9_collapse_layer_times")


def main() -> None:
    plotting.apply_style()
    plt.rcParams["savefig.dpi"] = 200
    s = load_s2()
    leftover_vs_step(s)
    tightness(s)
    leftover_worth(s)
    deadline(s)
    kernel_vs_r(s)
    step_bars(s)
    spare_resolved()
    leftover_vs_mechanisms()
    reconfig_stable()
    collapse_timetable()
    collapse_layer_times()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
