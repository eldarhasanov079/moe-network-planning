"""Stage 2 figures from plane/output/stage2/summary.json (+ CSVs). No resims."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "plane"))
from moe_feeder.claim_v1 import COLORS, _style  # noqa: E402
from moe_feeder.stage2 import HEADLINE_B  # noqa: E402

OUT = ROOT / "plane/output/stage2"
FIG = OUT / "figures"
SOURCES = ("flame", "olmoe")
LABEL = {"flame": "FLAME-MoE-290M", "olmoe": "OLMoE-1B-7B"}
C = {"mean": COLORS["mean"], "p95": COLORS["p95"], "worst": COLORS["worst"],
     "reserved": COLORS["planned"], "tail": COLORS["ecmp"], "spray": "#1a7f4b", "grey": "#888888"}


def _load():
    return json.loads((OUT / "summary.json").read_text())


def _csv(name):
    path = OUT / "{}.csv".format(name)
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def fig1_granularity(s):
    g = s.get("granularity")
    if not g:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, src in zip(axes, SOURCES):
        per_B = g.get(src, {}).get("aligned", {})
        if not per_B:
            continue
        Bs = sorted(int(b) for b in per_B)
        for p in ("mean", "p95"):
            for sizing, ls in (("rate", "-"), ("step", "--")):
                key = "{}_{}".format(p, sizing)
                ov = [per_B[str(B)].get(key, {}).get("overflow", np.nan) * 100 for B in Bs]
                ws = [per_B[str(B)].get(key, {}).get("waste", np.nan) * 100 for B in Bs]
                ax.plot(Bs, ov, ls, marker="o", color=C[p], label="{} {}: leftover".format(p, sizing))
                ax.plot(Bs, ws, ls, marker="s", color=C[p], alpha=0.45, label="{} {}: idle reserve".format(p, sizing))
        ax.axvline(HEADLINE_B[src], color=C["grey"], lw=0.8, ls=":")
        ax.text(HEADLINE_B[src], 0.0, " ~1 microbatch", fontsize=8, color=C["grey"], va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("tokens per All-to-All step (log)")
        ax.set_ylabel("% of slots")
        ax.set_title("{} — frozen E per step".format(LABEL[src]))
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Leftover (solid markers) and idle reserve (faded) vs step size: rate-sized E (—) vs step-sized E (- -)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_granularity.png", dpi=170)
    plt.close(fig)


def fig2_shedding(s):
    sh = s.get("shedding")
    if not sh:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    src = "flame"
    blk = sh.get(src, {}).get(str(HEADLINE_B[src]), {}).get("p95", {})
    arms = [a for a in ("arrival-drop", "rank-drop", "score-drop", "tail-gate-0.05", "tail-gate-0.10", "tail-gate-0.15", "tail-all") if a in blk]
    if arms:
        drop = [blk[a].get("weight_mass_dropped_per_token", 0.0) * 1e3 for a in arms]
        tail = [blk[a].get("weight_mass_tail_per_token", 0.0) * 1e3 for a in arms]
        x = np.arange(len(arms))
        ax.bar(x, drop, color=C["tail"], label="dropped weight (x1e-3 / token)")
        ax.bar(x, tail, bottom=drop, color=C["reserved"], alpha=0.5, label="weight carried in tail")
        ax.set_xticks(x, arms, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, max(d + t for d, t in zip(drop, tail)) * 1.25)
        ax.set_ylabel("router weight per token (x1e-3)")
        ax.set_title("FLAME — what the leftover is worth (P95, ~1 microbatch)")
        ax.legend(fontsize=8)
    ax = axes[1]
    src = "olmoe"
    blk = sh.get(src, {}).get(str(HEADLINE_B[src]), {}).get("p95", {})
    if blk:
        k = len(blk.get("arrival-drop", {}).get("rank_hist_over", []))
        x = np.arange(k)
        w = 0.38
        for i, (arm, col) in enumerate((("arrival-drop", C["grey"]), ("rank-drop", C["tail"]))):
            h = blk.get(arm, {}).get("rank_hist_over", [0] * k)
            ax.bar(x + (i - 0.5) * w, np.array(h) * 100, w, color=col, label=arm)
        ax.set_xticks(x, ["slot {}".format(i + 1) for i in range(k)], fontsize=8)
        ax.set_ylabel("% of leftover slots")
        ax.set_title("OLMoE — which slot position is left over (P95)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_shedding.png", dpi=170)
    plt.close(fig)


def fig3_links_astrasim(s):
    ln = s.get("links")
    az = s.get("astrasim")
    if not ln and not az:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    if ln:
        rows = _csv("links")
        x_labels, cov_frozen, cov_spray, uncovered = [], [], [], []
        for src in SOURCES:
            B = HEADLINE_B[src]
            for tail, sink in (("frozen", cov_frozen), ("spray", cov_spray)):
                by_layer = defaultdict(list)
                for r in rows:
                    if r["source"] == src and int(r["B"]) == B and r["policy"] == "p95" and r["tail_routing"] == tail:
                        by_layer[r["layer"]].append(float(r["coverage"]))
                sink.extend(np.mean(v) * 100 for _, v in sorted(by_layer.items()))
                if tail == "frozen":
                    x_labels.extend("{}:{}".format(src[0].upper(), l.replace("layer_", "")) for l, _ in sorted(by_layer.items()))
        x = np.arange(len(x_labels))
        ax.bar(x - 0.2, cov_frozen, 0.4, color=C["reserved"], label="tail on frozen path")
        ax.bar(x + 0.2, cov_spray, 0.4, color=C["spray"], label="tail sprayed")
        ax.set_xticks(x, x_labels, rotation=90, fontsize=7)
        ax.set_ylabel("% of tail bytes covered by idle reserve on the same link")
        ax.set_title("Where the tail lands (P95, ~1 microbatch)")
        ax.legend(fontsize=8)
    ax = axes[1]
    if az:
        labels, vals, cols = [], [], []
        for src in SOURCES:
            B = str(HEADLINE_B[src])
            ent = az.get(src, {}).get(B, {}).get("p95")
            if not ent:
                continue
            base = ent["reserved_only"] or 1.0
            for arm, col in (("reserved_only", C["reserved"]), ("one_class", C["grey"]), ("serialized", C["tail"])):
                labels.append("{}\n{}".format(src, arm.replace("_", " ")))
                vals.append(ent[arm] / base)
                cols.append(col)
        x = np.arange(len(labels))
        ax.bar(x, vals, color=cols)
        ax.axhline(1.0, color="#222", lw=0.8)
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylabel("ASTRA-sim Switch time / reserved-only")
        ax.set_title("ASTRA-sim Switch bracket (P95, ~1 microbatch)")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.003, "{:.3f}".format(v), ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_links_astrasim.png", dpi=170)
    plt.close(fig)


def fig4_two_class(s):
    cl = s.get("clos")
    if not cl:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, src in zip(axes, SOURCES):
        B = str(HEADLINE_B[src])
        ent = cl.get(src, {}).get(B)
        if not ent:
            continue
        env = ent["envelopes"].get("p95_rate", {})
        bars = [
            ("reserved only\n(Stage 1)", env.get("reserved_only_s", 0), C["reserved"]),
            ("plan(E)\ntimetable", env.get("plan_E_s", 0), C["p95"]),
            ("reserved +\ntail ECMP", env.get("two_class_ecmp_total_s", 0), C["tail"]),
            ("reserved +\ntail spray", env.get("two_class_spray_total_s", 0), C["spray"]),
            ("all M, one class,\nplan paths", ent.get("full_one_class_frozen_s", 0), C["grey"]),
            ("all M\nECMP", ent.get("full_ecmp_s", 0), "#c9a27e"),
            ("all M\nspray", ent.get("full_spray_s", 0), "#8fbf9f"),
        ]
        x = np.arange(len(bars))
        vals = [b[1] * 1e3 for b in bars]
        ax.bar(x, vals, color=[b[2] for b in bars], hatch=["", "", "", "", "//", "", ""])
        res = env.get("two_class_ecmp_reserved_s", 0) * 1e3
        if res:
            ax.axhline(res, color=C["reserved"], lw=0.8, ls=":")
        ax.set_xticks(x, [b[0] for b in bars], fontsize=7)
        ax.set_ylabel("whole-model Clos time (ms), ~1 microbatch step")
        ax.set_title("{} — two classes on one Clos (P95)".format(LABEL[src]))
        for i, v in enumerate(vals):
            ax.text(i, v, "{:.0f}".format(v), ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_two_class.png", dpi=170)
    plt.close(fig)


def fig5_tightness(s):
    t_all = s.get("tightness")
    if not t_all:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, src in zip(axes, SOURCES):
        t = t_all.get(src, {})
        for env_name, e in sorted(t.items()):
            p = env_name.split("_")[0]
            marker = "o" if env_name.endswith("rate") else "^"
            ax.scatter(e["idle_reserve"] * 100, e["extension_rel"] * 100, color=C.get(p, C["grey"]), marker=marker, s=60)
            ax.annotate("{}\n{:.2f}% guaranteed".format(env_name, e["guaranteed_share"] * 100),
                        (e["idle_reserve"] * 100, e["extension_rel"] * 100), fontsize=7, xytext=(5, 3), textcoords="offset points")
        ax.set_xlabel("idle reserve (% of E)")
        ax.set_ylabel("tail past the reserved schedule (%)")
        ax.set_title("{} — reservation policy trade (~1 microbatch)".format(LABEL[src]))
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    fig.suptitle("circle = Stage 1 E sized as a rate; triangle = E rebuilt per step", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_tightness.png", dpi=170)
    plt.close(fig)


def main() -> None:
    _style()
    FIG.mkdir(parents=True, exist_ok=True)
    s = _load()
    fig1_granularity(s)
    fig2_shedding(s)
    fig3_links_astrasim(s)
    fig4_two_class(s)
    fig5_tightness(s)
    fig6_deadline(s)
    fig7_compute(s)
    fig8_step_bars(s)
    print("figures ->", FIG)


# ------------------------------------------------------------------ Stage 2b figures


def fig6_deadline(s):
    st = s.get("step")
    if not st:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, src in zip(axes, SOURCES):
        S = st.get(src)
        if not S:
            continue
        anchor = S["anchor_ratio"]
        alloc = S.get("primary_allocator", "max-min")
        key = "aligned|{}|r={}".format(alloc, anchor)
        curve = S["deadline"].get(key) or next((v for k, v in S["deadline"].items() if k.startswith("aligned|" + alloc)), None)
        if not curve:
            continue
        fracs = sorted(float(f) for f in curve)
        x = [f * 100 for f in fracs]
        if src == "flame":
            stage1 = s.get("shedding", {}).get("flame", {}).get(str(HEADLINE_B["flame"]), {}).get("p95", {}).get("arrival-drop", {}).get("weight_mass_dropped_per_token")
            for order, attr, col, ls, lab in (("arrival", "random", C["grey"], "-", "arrival admission, random cut (Stage 1 lens)"),
                                              ("rank", "best", C["tail"], "-", "rank admission, heaviest-first send"),
                                              ("score", "best", C["reserved"], "-", "score admission, heaviest-first send"),
                                              ("score", "random", C["reserved"], "--", "score admission, random cut")):
                y = [(curve[str(f)]["orders"].get("{}_{}".format(order, attr), {}).get("cost_per_token") or 0.0) * 1e3 for f in fracs]
                sd = [(curve[str(f)]["orders"].get("{}_{}".format(order, attr), {}).get("cost_per_token_seed_std") or 0.0) * 1e3 for f in fracs]
                ax.errorbar(x, y, yerr=sd, fmt="o" + ls, color=col, capsize=2, label=lab, ms=4)
            if stage1:
                ax.axhline(stage1 * 1e3, color="#222", lw=0.8, ls=":", label="drop the whole leftover (Stage 1)")
            ax.set_ylabel("router weight lost per token (x1e-3)")
            ax.set_title("FLAME — quality cost of cutting the tail at a deadline")
        else:
            y_all = [curve[str(f)]["orders"].get("rank_best", {}).get("dropped_frac_of_all_slots", 0.0) * 100 for f in fracs]
            y_tail = [curve[str(f)]["orders"].get("rank_best", {}).get("dropped_frac_of_tail_slots", 0.0) * 100 for f in fracs]
            ax.plot(x, y_all, "o-", color=C["tail"], label="slots dropped, % of all slots")
            ax2 = ax.twinx()
            ax2.plot(x, y_tail, "s--", color=C["grey"], label="% of the leftover dropped")
            ax2.set_ylabel("% of the leftover that is dropped")
            ax2.legend(loc="center right", fontsize=8)
            ax.set_ylabel("slots dropped (% of all top-8 slots)")
            ax.set_title("OLMoE — slots cut at a deadline (rank order, no magnitudes)")
        sw = S["sweep"].get(str(anchor), {})
        res = sw.get("reserved_only_s") or 0.0
        if res and sw.get("dispatch_side_lateness_s") is not None:
            ax.axvline(max(0.0, sw["dispatch_side_lateness_s"]) / res * 100, color=C["spray"], lw=0.8, ls="--")
            ax.text(max(0.0, sw["dispatch_side_lateness_s"]) / res * 100, ax.get_ylim()[1] * 0.92, " dispatch tail\n lateness at anchor", fontsize=6, color=C["spray"])
        ax.set_xlabel("allowed step lateness, % of the reserved step")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Deadline as a sender-side byte quota: what is lost when the tail must not delay the step", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_deadline.png", dpi=170)
    plt.close(fig)


def fig7_compute(s):
    st = s.get("step")
    if not st:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, src in zip(axes, SOURCES):
        S = st.get(src)
        if not S:
            continue
        rs = sorted(float(r) for r in S["sweep"])
        ext_m = [S["sweep"][str(r)]["ext_monolithic"] * 100 for r in rs]
        ext_c = [S["sweep"][str(r)]["ext_chunked"] * 100 for r in rs]
        xs = [max(r, 0.004) for r in rs]
        ax.plot(xs, ext_m, "o-", color=C["tail"], label="monolithic kernel (today): waits for the tail")
        ax.plot(xs, ext_c, "s-", color=C["reserved"], label="chunked kernel: reserved GEMM first")
        ax.set_xscale("log")
        a = S["assumptions"] if "assumptions" in S else s["step"]["assumptions"][src]
        a = s["step"]["assumptions"][src]
        ax.axvspan(min(a["ratio_band"].values()), max(a["ratio_band"].values()), color=C["grey"], alpha=0.18, label="anchor band (H100..A100, 100 Gbps)")
        ax.axvline(a["ratio"], color="#222", lw=0.8, ls=":")
        ax.axvline(a["ratio_400G"], color=C["spray"], lw=0.8, ls="--", label="400 Gbps marker (x4)")
        if src == "olmoe":
            ax.axvline(a["ratio_true_bytes"], color=C["p95"], lw=0.8, ls="-.", label="OLMoE at true 4096-B slots")
        ax.set_xlabel("expert compute per slot / wire time per slot (log; r=0 drawn at left edge)")
        ax.set_ylabel("step time past reserved-only (%)")
        ax.set_title("{} — can the tail hide behind expert compute?".format(LABEL[src]))
        ax.legend(fontsize=7)
    fig.suptitle("Pre-stated ceiling: only the dispatch half of the tail can hide; nothing computes after the combine in a per-layer step", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_compute.png", dpi=170)
    plt.close(fig)


def fig8_step_bars(s):
    st = s.get("step")
    if not st:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, src in zip(axes, SOURCES):
        S = st.get(src)
        if not S:
            continue
        A = S["anchor"]
        tp = A[S.get("primary_allocator", "max-min")]
        alt = A["tte-priority" if S.get("primary_allocator", "max-min") == "max-min" else "max-min"]
        bars = [
            ("reserved only", tp["reserved_only_s"], C["reserved"], ""),
            ("+ tail, monolithic\n(today's kernel)", tp["monolithic_s"], C["tail"], ""),
            ("+ tail, chunked", tp["chunked_s"], C["spray"], ""),
            ("all M one class,\nplan paths", A["controls"]["one_class_plan_s"], C["grey"], "//"),
            ("all M ECMP", A["controls"]["full_ecmp_s"], "#c9a27e", ""),
            ("all M spray", A["controls"]["full_spray_s"], "#8fbf9f", ""),
            ("chunked,\nslack-tiered alloc", alt["chunked_s"], C["spray"], ".."),
            ("chunked,\nRDMA semantics", A["rdma"]["chunked_s"], C["spray"], "xx"),
        ]
        x = np.arange(len(bars))
        vals = [b[1] * 1e3 for b in bars]
        ax.bar(x, vals, color=[b[2] for b in bars], hatch=[b[3] for b in bars])
        ax.set_xticks(x, [b[0] for b in bars], fontsize=6.5, rotation=25, ha="right")
        ax.set_ylabel("whole-model step time (ms), forward")
        ax.set_title("{} — full step at the compute anchor (r={})".format(LABEL[src], S["anchor_ratio"]), fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v, "{:.0f}".format(v), ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, max(vals) * 1.12)
    fig.tight_layout()
    fig.savefig(FIG / "fig8_step_bars.png", dpi=170)
    plt.close(fig)

if __name__ == "__main__":
    main()
