"""Figures for the two Stage 3 experiments from plane/output/stage3/*.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "plane"))
from moe_feeder.claim_v1 import COLORS, _style  # noqa: E402

OUT = ROOT / "plane/output/stage3"
FIG = OUT / "figures"
MODELS = ("flame-moe-290m", "flame-moe-721m", "flame-moe-1.7b", "olmoe")
LABEL = {"flame-moe-290m": "FLAME 290M", "flame-moe-721m": "FLAME 721M", "flame-moe-1.7b": "FLAME 1.7B", "olmoe": "OLMoE 1B-7B"}
HEAD = {"flame-moe-290m": "125000", "flame-moe-721m": "125000", "flame-moe-1.7b": "125000", "olmoe": "102400"}
C = {"drop": "#888888", "spare": COLORS["p95"], "any": COLORS["mean"], "spare_any": COLORS["worst"],
     "freeze": COLORS["planned"], "freeze_robust": "#5b8ff9", "margin_1.05": "#9ecae1", "margin_1.10": "#6baed6",
     "uniform_cf1.25": "#bbbbbb", "periodic_slide": COLORS["ecmp"], "periodic_expand": "#e6a06b", "ema": "#7b2d8e",
     "oracle": "#222222", "guard_loo_1": "#1a7f4b", "guard_loo_1.5": "#2ca02c", "guard_loo_2": "#98df8a",
     "cap_2": "#d62728", "cap_5": "#ff9896", "cosine_0.99": "#bcbd22"}
MAIN_POLICIES = ("freeze", "margin_1.05", "periodic_slide", "periodic_expand", "ema", "guard_loo_1", "cap_2", "cosine_0.99", "oracle")


def _key(M, placement="contiguous", spares=1, B=None):
    return "{}|spares{}|B{}".format(placement, spares, B)


def _load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


EXCLUDE = {"flame-moe-1.7b": {"3300"}}
DLABEL = dict(LABEL)
_KEYS = ("frac_slots_over", "frac_tokens_hit", "resolved_spare_frac", "resolved_any_frac", "dropped_frac_of_over",
         "frac_slots_dropped", "frac_tokens_dropped", "spare_cell_full_frac", "E_utilisation",
         "cost_pref_per_token", "cost_pess_per_token", "cost_drop_per_token")


def _mean(vals):
    v = [float(x) for x in vals if x not in (None, "")]
    return sum(v) / len(v) if v else float("nan")


def _dist(vals):
    v = sorted(float(x) for x in vals)
    if not v:
        return {}
    return {"min": v[0], "median": v[len(v) // 2], "max": v[-1], "mean": sum(v) / len(v)}


def _reaggregate_without_collapse(S):
    """Rebuild the D summary for models in EXCLUDE from the per-row CSV, dropping those checkpoints."""
    import csv as _csv
    path = OUT / "deflect.csv"
    if not path.exists():
        return S
    rows = list(_csv.DictReader(path.open()))
    for model, drop_cks in EXCLUDE.items():
        sel_m = [r for r in rows if r["model"] == model and r["checkpoint"] not in drop_cks]
        if not sel_m or model not in S:
            continue
        M = {}
        for blk_key in S[model]:
            pl, sp, B = blk_key.split("|")
            n_sp, Bv = int(sp[6:]), int(B[1:])
            block = {}
            for ent_key in S[model][blk_key]:
                order, pol = ent_key.split("_", 1)
                sel = [r for r in sel_m if r["placement"] == pl and int(r["n_spares"]) == n_sp
                       and int(float(r["B"])) == Bv and r["order"] == order and r["policy"] == pol]
                if not sel:
                    continue
                ent = {k: _mean(r.get(k) for r in sel) for k in _KEYS
                       if any(r.get(k) not in (None, "") for r in sel)}
                ent["n"] = len(sel)
                res = lambda r: float(r["resolved_spare_frac"] or 0) + float(r["resolved_any_frac"] or 0)
                cells, per_layer = [], {}
                for layer in sorted({r["layer"] for r in sel}):
                    ls = [r for r in sel if r["layer"] == layer]
                    per_layer[layer] = _mean(res(r) for r in ls)
                    for ck in sorted({r["checkpoint"] for r in ls}):
                        cells.append(_mean(res(r) for r in ls if r["checkpoint"] == ck))
                if cells:
                    ent["resolved_dist"] = _dist(cells)
                ent["resolved_per_layer"] = per_layer
                block[ent_key] = ent
            M[blk_key] = block
        S[model] = M
        DLABEL[model] = LABEL[model] + "*"
    return S


# ------------------------------------------------------------------ Experiment D


def figD1_resolved(S):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    x_labels, spare_v, any_v, full_v = [], [], [], []
    for m in MODELS:
        M = S.get(m)
        if not M:
            continue
        order = "arrival"
        Bs = sorted({int(k.split("B")[-1]) for k in M if k.startswith("contiguous|spares1|")})
        for B in Bs:
            blk = M.get(_key(M, "contiguous", 1, B), {})
            e = blk.get("{}_spare_any".format(order))
            if not e:
                continue
            x_labels.append("{}\n{}".format(DLABEL[m], "{}k".format(B // 1000)))
            spare_v.append(e["resolved_spare_frac"] * 100)
            any_v.append(e["resolved_any_frac"] * 100)
            full_v.append(blk.get("{}_spare".format(order), {}).get("spare_cell_full_frac", 0) * 100)
    x = np.arange(len(x_labels))
    ax.bar(x, spare_v, color=C["spare"], label="resolved by the spare expert")
    ax.bar(x, any_v, bottom=spare_v, color=C["any"], label="resolved by any device with room")
    ax.plot(x, full_v, "kx", label="spare's own cell also full (%)")
    ax.set_xticks(x, x_labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("% of overflow slots")
    ax.set_title("What deflection resolves (arrival order, per step size)")
    ax.legend(fontsize=7)
    ax = axes[1]
    labels, vals = [], []
    for m in MODELS:
        M = S.get(m)
        if not M:
            continue
        B = int(HEAD[m])
        for pl, sp in (("contiguous", 1), ("contiguous", 2), ("roundrobin", 1), ("random0", 1), ("random1", 1)):
            e = M.get(_key(M, pl, sp, B), {}).get("arrival_spare")
            if e and "resolved_dist" in e:
                labels.append("{} {}{}".format(DLABEL[m].split()[-1], {"contiguous": "contig", "roundrobin": "rrobin", "random0": "rand0", "random1": "rand1"}[pl], " 2sp" if sp == 2 else ""))
                d = e["resolved_dist"]
                vals.append((d["median"] * 100, d["min"] * 100, d["max"] * 100))
    x = np.arange(len(labels))
    med = [v[0] for v in vals]
    ax.bar(x, med, color=C["spare"], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], capsize=3)
    ax.set_xticks(x, labels, fontsize=6, rotation=60, ha="right")
    ax.set_ylabel("% of overflow resolved by the spare\n(median, min-max over layer x ckpt)")
    ax.set_title("Spare resolution by placement and spares (~1 microbatch)")
    fig.tight_layout()
    fig.savefig(FIG / "figD1_resolved.png", dpi=170)
    plt.close(fig)


def figD2_cost(S):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    labels, drop, spare_pref, spare_pess, any_pref = [], [], [], [], []
    for m in ("flame-moe-290m", "flame-moe-721m", "flame-moe-1.7b"):
        M = S.get(m)
        if not M:
            continue
        B = int(HEAD[m])
        blk = M.get(_key(M, "contiguous", 1, B), {})
        for order in ("arrival", "score"):
            e_d = blk.get("{}_drop".format(order)); e_s = blk.get("{}_spare".format(order)); e_a = blk.get("{}_any".format(order))
            if not (e_d and e_s and e_a):
                continue
            labels.append("{}\n{}".format(DLABEL[m], order))
            drop.append(e_d["cost_drop_per_token"] * 1e3)
            spare_pref.append(e_s["cost_pref_per_token"] * 1e3)
            spare_pess.append(e_s["cost_pess_per_token"] * 1e3)
            any_pref.append(e_a["cost_pref_per_token"] * 1e3)
    x = np.arange(len(labels)); w = 0.2
    ax.bar(x - 1.5 * w, drop, w, color=C["drop"], label="drop (Stage 1)")
    ax.bar(x - 0.5 * w, spare_pref, w, color=C["spare"], label="spare: preference gap")
    ax.bar(x + 0.5 * w, spare_pess, w, color=C["spare"], alpha=0.45, hatch="//", label="spare: pessimistic bound")
    ax.bar(x + 1.5 * w, any_pref, w, color=C["any"], label="any device: lower bound")
    ax.set_xticks(x, labels, fontsize=6.5)
    ax.set_ylabel("router weight lost per token (x1e-3)")
    ax.set_title("Quality proxy per token (~1 microbatch)")
    ax.annotate("* excludes the 1.7B collapse checkpoint (3300)", (0.02, 0.03), xycoords="axes fraction", fontsize=6, color="#555")
    ax.legend(fontsize=7)
    ax = axes[1]
    for m, col in (("flame-moe-290m", COLORS["p95"]), ("flame-moe-721m", COLORS["mean"]), ("flame-moe-1.7b", COLORS["worst"]), ("olmoe", "#222222")):
        M = S.get(m)
        if not M:
            continue
        e = M.get(_key(M, "contiguous", 1, int(HEAD[m])), {}).get("arrival_spare")
        if not e:
            continue
        per = e["resolved_per_layer"]
        ax.plot(range(len(per)), [v * 100 for v in per.values()], "o-", color=col, label=DLABEL[m], ms=3)
    ax.set_xlabel("layer index")
    ax.set_ylabel("% of overflow resolved by the spare (arrival)")
    ax.set_title("Per-layer spread of spare resolution")
    ax.annotate("* excludes the 1.7B collapse checkpoint (3300)", (0.02, 0.03), xycoords="axes fraction", fontsize=6, color="#555")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "figD2_cost.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ Experiment R


def figR1_trajectories(S):
    keys = [k for k in S["models"] if k.startswith("flame-moe-290m") or k.startswith("olmoe") or k.startswith("flame-moe-1.7b")]
    keys = sorted(keys)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.2), squeeze=False)
    for ax, key in zip(axes[0], keys):
        B = list(S["models"][key].keys())[0]
        agg = S["models"][key][B]
        model = key.split("|")[0]
        for pol in ("freeze", "margin_1.05", "periodic_slide", "ema", "guard_loo_1", "cap_2", "cosine_0.99", "oracle"):
            ent = agg.get(pol)
            if not ent:
                continue
            ck = ent["by_ckpt"]
            xs = list(range(len(ck)))
            ys = [v["overflow"] * 100 for v in ck.values()]
            ax.plot(xs, ys, "o-", color=C[pol], label=pol, ms=3, lw=1.2 if pol != "freeze" else 2)
            fires = [i for i, v in enumerate(ck.values()) if v["fires"]]
            if fires:
                ax.plot([xs[i] for i in fires], [ys[i] for i in fires], "v", color=C[pol], ms=7)
        ax.set_xticks(list(range(len(ck))), list(ck.keys()), rotation=60, fontsize=6)
        ax.set_ylabel("per-step overflow (% of slots), mean over layers")
        ax.set_title("{} (K={})".format(LABEL[model], key.split("K")[1]), fontsize=9)
        ax.set_yscale("log")
        ax.legend(fontsize=6)
    fig.suptitle("Overflow along training under each policy (triangles = reconfigurations); step 32k tokens", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "figR1_trajectories.png", dpi=170)
    plt.close(fig)


def figR2_pareto(S):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    from matplotlib.lines import Line2D

    marks = {}
    for i, key in enumerate(sorted(S["models"])):
        marks[key] = ("o", "s", "^", "v", "D", "P")[i % 6]
    for ax, metric, lab in ((axes[0], "overflow", "post-window overflow (% of slots)"), (axes[1], "idle", "post-window idle reserve (% of E)")):
        for key in sorted(S["models"]):
            B = list(S["models"][key].keys())[0]
            agg = S["models"][key][B]
            for pol, ent in agg.items():
                if pol not in MAIN_POLICIES:
                    continue
                ax.scatter(ent["n_reconfig_per_layer"], ent[metric] * 100, color=C.get(pol, "#999"),
                           marker=marks[key], s=34, edgecolors="none", alpha=0.9)
        ax.set_xlabel("reconfigurations per layer over training (log)")
        ax.set_xscale("symlog")
        ax.set_ylabel(lab)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(handles=[Line2D([], [], color="#444", marker=marks[k], ls="", ms=5,
                                   label="{} (window K={})".format(LABEL[k.split("|")[0]], k.split("|K")[-1]))
                            for k in sorted(S["models"])], fontsize=6, title="trace", title_fontsize=6, loc="best")
    axes[1].legend(handles=[Line2D([], [], color=C.get(p, "#999"), marker="o", ls="", ms=5, label=p)
                            for p in MAIN_POLICIES], fontsize=6, ncol=2, title="policy", title_fontsize=6, loc="best")
    fig.suptitle("What reconfiguring buys, against how often it happens", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "figR2_pareto.png", dpi=170)
    plt.close(fig)


def figR3_collapse(S):
    col = S.get("collapse") or {}
    if not col:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    key = next((k for k in S["models"] if k.startswith("flame-moe-1.7b|K2")), None)
    if key:
        B = list(S["models"][key].keys())[0]
        agg = S["models"][key][B]
        for pol in ("freeze", "periodic_slide", "periodic_expand", "ema", "guard_loo_1", "cosine_0.99"):
            ck = agg[pol]["by_ckpt"]
            ax.plot(list(ck.keys()), [v["idle"] * 100 for v in ck.values()], "s-", color=C[pol], label=pol, ms=3)
        ax.set_ylabel("idle reserve (% of E), mean over layers")
        ax.set_title("1.7B (collapse held out): idle reserve after the collapse", fontsize=9)
        ax.tick_params(axis="x", rotation=60, labelsize=6)
        ax.legend(fontsize=6)
    ax = axes[1]
    k2 = next((k for k in col if k.endswith("K2")), None)
    if k2:
        mkey = next((k for k in S["models"] if k.startswith("flame-moe-1.7b|K2")), None)
        ck_order = list(S["models"][mkey][list(S["models"][mkey].keys())[0]]["freeze"]["by_ckpt"]) if mkey else []
        ck_ix = {c: i for i, c in enumerate(ck_order)}
        for layer, pols in list(col[k2].items())[:1]:
            for pol, hist in pols["installed"].items():
                xs, ys, seen = [], [], {}
                for h in hist:
                    if h["ckpt"] not in ck_ix:
                        continue
                    n = seen.get(h["ckpt"], 0); seen[h["ckpt"]] = n + 1
                    xs.append(ck_ix[h["ckpt"]] + 0.18 * n)
                    ys.append(h["timetable_s"] * 1e3)
                if not xs:
                    continue
                xs.append(len(ck_order) - 1 + 0.5); ys.append(ys[-1])
                style = {"freeze": "-", "freeze_robust": ":", "guard_loo_1": "-", "cosine_0.99": (0, (5, 3)),
                         "periodic_slide": (0, (1.5, 1.5))}.get(pol, "-")
                ax.step(xs, ys, where="post", color=C.get(pol, "#999"), label=pol, lw=1.6, ls=style, alpha=0.9)
                ax.plot(xs[:-1], ys[:-1], "o", color=C.get(pol, "#999"), ms=3.5)
        ax.set_xticks(range(len(ck_order)), ck_order)
        ax.set_xlabel("checkpoint (iteration) at which a reservation is installed")
        ax.set_ylabel("timetable of the installed E (ms, one layer)")
        ax.set_title("What each installed reservation costs in schedule length", fontsize=9)
        ax.tick_params(axis="x", rotation=60, labelsize=6)
        ax.legend(fontsize=6)
    ax = axes[2]
    if k2:
        rows = []
        for layer, pols in col[k2].items():
            case = pols.get("case", {}).get("3300")
            if case:
                rows.append((layer, case["overflow_under_clean_E"] * 100, case["step_clean_E_plus_tail_s"] * 1e3,
                             case["step_E_accommodates_s"] * 1e3, case["downlink_bound_s"] * 1e3))
        if rows:
            x = np.arange(len(rows)); w = 0.27
            ax.bar(x - w, [r[2] for r in rows], w, color=COLORS["ecmp"], label="clean E + tail carried (no reconfig)")
            ax.bar(x, [r[3] for r in rows], w, color=COLORS["planned"], label="E accommodates the collapse (reconfig)")
            ax.bar(x + w, [r[4] for r in rows], w, color="#888", label="hot-NIC bound (inter-host)")
            ax.set_xticks(x, [r[0] for r in rows], fontsize=6, rotation=30)
            ax.set_ylabel("one-layer step time at iter 3300 (ms)")
            ax.set_title("Collapse step: does reconfiguring buy time?", fontsize=9)
            ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "figR3_collapse.png", dpi=170)
    plt.close(fig)


def main() -> None:
    _style()
    FIG.mkdir(parents=True, exist_ok=True)
    D = _load("deflect_summary.json")
    D = _reaggregate_without_collapse(D) if D else D
    R = _load("reconfig_summary.json")
    if D:
        figD1_resolved(D)
        figD2_cost(D)
    if R:
        figR1_trajectories(R)
        figR2_pareto(R)
        figR3_collapse(R)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
