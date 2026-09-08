"""Prefix vs full-file sanity: overflow + token-drop, no ASTRA-sim."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "plane"))
sys.path.insert(0, str(ROOT / "moe-trace-experiments"))

from moe_feeder.admit import token_drop_stats
from moe_feeder.claim_v1 import POLICIES, make_cfg, source_bundle
from moe_feeder.matrix import build_matrix, maps
from moe_feeder.policy import collapse_history, cosine, score_heldout

SAMPLE_OLMOE = 205_000
SAMPLE_FLAME = 250_000
FLAME_LAYER = "layer_02"
FLAME_CKPT = 5473
MOVED_OVERFLOW_PP = 0.3
MOVED_DROP_PP = 1.0
OUT = ROOT / "plane/output/full-trace-sanity"
REPORT = ROOT / "plane/reports/full-trace-sanity.md"


def _mean(xs):
    return float(np.mean(xs)) if xs else 0.0


def _dest_share(M: np.ndarray) -> np.ndarray:
    col = np.asarray(M, dtype=np.float64).sum(axis=0)
    tot = col.sum()
    return col / tot if tot else col


def _score_source(bundle, n_tokens: int | None, label: str):
    from common.olmoe_loader import load_checkpoint

    cfg0 = make_cfg(bundle, bundle["layers"][0])
    cfg0.n_tokens = n_tokens or SAMPLE_OLMOE
    packed = {}
    for ckpt in bundle["ckpts"]:
        arr = load_checkpoint(str(ckpt), n_tokens, verbose=True)
        packed[str(ckpt)] = arr
        print("  {} {} tokens={:,}".format(label, ckpt, arr.shape[0]), flush=True)

    n_used = min(arr.shape[0] for arr in packed.values())
    matrices = {}
    tokens = {}
    for layer_i, layer in enumerate(bundle["layers"]):
        cfg = make_cfg(bundle, layer)
        cfg.n_tokens = n_used
        for ckpt in bundle["ckpts"]:
            indices = packed[str(ckpt)][:n_used, layer_i, :]
            place, src = maps(cfg, len(indices))
            matrices[(layer, str(ckpt))] = build_matrix(cfg, indices, place, src)
            tokens[(layer, str(ckpt))] = (indices, place, src)

    window = bundle["default_window"]
    reserved = {p: {} for p in POLICIES}
    for layer in bundle["layers"]:
        hist = [matrices[(layer, str(c))] for c in bundle["ckpts"][:window]]
        for policy in POLICIES:
            reserved[policy][layer] = collapse_history(hist, policy)

    overflow = {p: [] for p in POLICIES}
    drops = {p: [] for p in POLICIES}
    for ckpt in bundle["ckpts"][window:]:
        for layer in bundle["layers"]:
            actual = matrices[(layer, str(ckpt))]
            indices, place, src = tokens[(layer, str(ckpt))]
            for policy in POLICIES:
                overflow[policy].append(
                    100.0 * score_heldout(actual, reserved[policy][layer])["overflow_ratio"]
                )
                drop = token_drop_stats(indices, place, src, reserved[policy][layer])
                drops[policy].append(100.0 * drop["frac_tokens_dropped"])
    return {
        "label": label,
        "n_tokens": n_used,
        "overflow_pct": {p: _mean(overflow[p]) for p in POLICIES},
        "token_drop_pct": {p: _mean(drops[p]) for p in POLICIES},
    }


def run_olmoe():
    print("=== OLMoE full file vs 205k prefix ===", flush=True)
    bundle = source_bundle("olmoe")
    sample = _score_source(bundle, SAMPLE_OLMOE, "sample")
    full = _score_source(bundle, None, "full")
    deltas = {
        p: {
            "overflow_pp": full["overflow_pct"][p] - sample["overflow_pct"][p],
            "token_drop_pp": full["token_drop_pct"][p] - sample["token_drop_pct"][p],
        }
        for p in POLICIES
    }
    return {"sample": sample, "full": full, "delta_pp": deltas}


def _flame_E(bundle, layer: str):
    from common.flame_loader import load_layer

    cfg = make_cfg(bundle, layer)
    hist = []
    for ckpt in bundle["ckpts"][: bundle["default_window"]]:
        _, indices = load_layer(int(ckpt), layer, SAMPLE_FLAME, model=bundle["model"], verbose=False)
        place, src = maps(cfg, len(indices))
        hist.append(build_matrix(cfg, indices, place, src))
        print("  profiled {} {} n={:,}".format(ckpt, layer, len(indices)), flush=True)
    return {policy: collapse_history(hist, policy) for policy in POLICIES}, cfg


def run_flame_probe():
    from common.flame_loader import iter_layer_indices, load_layer
    from common.traffic_matrix import build_dispatch_matrix

    print("=== FLAME {} ckpt {} : 250k blocks vs rest of file ===".format(
        FLAME_LAYER, FLAME_CKPT), flush=True)
    bundle = source_bundle("flame")
    reserved, cfg = _flame_E(bundle, FLAME_LAYER)
    place, _ = maps(cfg, SAMPLE_FLAME)

    _, idx0 = load_layer(
        FLAME_CKPT, FLAME_LAYER, SAMPLE_FLAME, model=bundle["model"], verbose=False
    )
    place0, src0 = maps(cfg, len(idx0))
    M0 = build_matrix(cfg, idx0, place0, src0)
    block0 = {
        "n_tokens": len(idx0),
        "overflow_pct": {
            p: 100.0 * score_heldout(M0, reserved[p])["overflow_ratio"] for p in POLICIES
        },
        "token_drop_pct": {
            p: 100.0 * token_drop_stats(idx0, place0, src0, reserved[p])["frac_tokens_dropped"]
            for p in POLICIES
        },
    }
    dest0 = _dest_share(M0)

    later_overflow = {p: [] for p in POLICIES}
    later_dest = []
    drop_blocks = {}
    buf = []
    seen = 0
    block_i = 0
    last_full = None

    for batch in iter_layer_indices(
        FLAME_CKPT, FLAME_LAYER, model=bundle["model"], verbose=True
    ):
        buf.append(batch)
        seen += len(batch)
        while sum(len(x) for x in buf) >= SAMPLE_FLAME:
            take = SAMPLE_FLAME
            pieces = []
            got = 0
            rest = []
            for chunk in buf:
                need = take - got
                if got >= take:
                    rest.append(chunk)
                    continue
                if len(chunk) <= need:
                    pieces.append(chunk)
                    got += len(chunk)
                else:
                    pieces.append(chunk[:need])
                    rest.append(chunk[need:])
                    got += need
            buf = rest
            indices = np.concatenate(pieces, axis=0)
            if block_i == 0:
                block_i += 1
                print("  skip prefix block (already scored from cache)", flush=True)
                continue
            src = (np.arange(len(indices)) // int(np.ceil(len(indices) / cfg.num_src_ranks))).astype(np.int64)
            M = build_dispatch_matrix(
                indices, place, src, cfg.num_src_ranks, cfg.num_devices, dedup_device=False
            )
            later_dest.append(_dest_share(M))
            for policy in POLICIES:
                later_overflow[policy].append(
                    100.0 * score_heldout(M, reserved[policy])["overflow_ratio"]
                )
            last_full = (block_i, indices, src, M)
            block_i += 1
            if block_i % 20 == 0:
                print("  block {}  tokens {:,}".format(block_i, seen), flush=True)

    n_later = len(later_overflow["p95"])
    drop_ids = []
    if last_full is not None:
        mid_i = max(1, n_later // 2)
        bi, indices, src, _M = last_full
        drop_blocks["last"] = {
            "block": bi,
            "token_drop_pct": {
                p: 100.0 * token_drop_stats(indices, place, src, reserved[p])["frac_tokens_dropped"]
                for p in POLICIES
            },
        }
        drop_ids.append(bi)

    later = {
        "n_blocks": n_later,
        "n_tokens_streamed": seen,
        "overflow_pct_mean": {p: _mean(later_overflow[p]) for p in POLICIES},
        "overflow_pct_min": {p: float(np.min(later_overflow[p])) if later_overflow[p] else 0.0 for p in POLICIES},
        "overflow_pct_max": {p: float(np.max(later_overflow[p])) if later_overflow[p] else 0.0 for p in POLICIES},
        "dest_cosine_vs_prefix": (
            float(np.mean([cosine(dest0, d) for d in later_dest])) if later_dest else 1.0
        ),
        "token_drop_last_block": drop_blocks.get("last"),
    }
    deltas = {
        p: {
            "overflow_pp": later["overflow_pct_mean"][p] - block0["overflow_pct"][p],
            "token_drop_pp": (
                later["token_drop_last_block"]["token_drop_pct"][p] - block0["token_drop_pct"][p]
                if later["token_drop_last_block"]
                else 0.0
            ),
        }
        for p in POLICIES
    }
    return {
        "layer": FLAME_LAYER,
        "checkpoint": FLAME_CKPT,
        "prefix_block": block0,
        "later_blocks": later,
        "delta_pp": deltas,
    }


def _moved(delta_pp) -> bool:
    for policy in POLICIES:
        if abs(delta_pp[policy]["overflow_pp"]) > MOVED_OVERFLOW_PP:
            return True
        if abs(delta_pp[policy]["token_drop_pp"]) > MOVED_DROP_PP:
            return True
    return False


def _render(payload) -> str:
    olmoe = payload["olmoe"]
    flame = payload["flame"]
    lines = [
        "# Prefix vs full-file sanity",
        "",
        "Overflow + token-drop only. No ASTRA-sim, no Clos. "
        "Moved if |Δ overflow| > {:.1f} pp or |Δ token-drop| > {:.1f} pp.".format(
            MOVED_OVERFLOW_PP, MOVED_DROP_PP
        ),
        "",
        "## OLMoE — 205k prefix vs whole JSONL",
        "",
        "Freeze after 2, all 16 layers, held-out mean. Full file uses every sequence in "
        "`allenai/analysis_olmoe`.",
        "",
        "| Policy | overflow 205k | overflow full | Δ pp | tokens≥1 dest 205k | full | Δ pp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in POLICIES:
        s, f, d = olmoe["sample"], olmoe["full"], olmoe["delta_pp"][p]
        lines.append(
            "| {} | {:.2f}% | {:.2f}% | {:+.2f} | {:.2f}% | {:.2f}% | {:+.2f} |".format(
                p,
                s["overflow_pct"][p],
                f["overflow_pct"][p],
                d["overflow_pp"],
                s["token_drop_pct"][p],
                f["token_drop_pct"][p],
                d["token_drop_pp"],
            )
        )
    lines += [
        "",
        "Tokens per file: sample **{:,}**, full **{:,}**.".format(
            olmoe["sample"]["n_tokens"], olmoe["full"]["n_tokens"]
        ),
        "",
        "**OLMoE verdict:** {}.".format(
            "MOVED" if payload["olmoe_moved"] else "stable — prefix is enough"
        ),
        "",
        "## FLAME — {} ckpt {} , 250k blocks".format(FLAME_LAYER, FLAME_CKPT),
        "",
        "E is the Stage 1 freeze (first 4 ckpts × 250k). Later 250k blocks of the "
        "same file are scored against that E. Same scale, not a 210× slot blow-up.",
        "",
        "Later blocks streamed: **{}**. Dest-share cosine vs prefix: **{:.4f}**.".format(
            flame["later_blocks"]["n_blocks"],
            flame["later_blocks"]["dest_cosine_vs_prefix"],
        ),
        "",
        "| Policy | overflow prefix | overflow later mean [min, max] | Δ pp | token-drop prefix | last block | Δ pp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    later = flame["later_blocks"]
    last = (later["token_drop_last_block"] or {}).get("token_drop_pct", {p: 0.0 for p in POLICIES})
    for p in POLICIES:
        d = flame["delta_pp"][p]
        lines.append(
            "| {} | {:.2f}% | {:.2f}% [{:.2f}, {:.2f}] | {:+.2f} | {:.2f}% | {:.2f}% | {:+.2f} |".format(
                p,
                flame["prefix_block"]["overflow_pct"][p],
                later["overflow_pct_mean"][p],
                later["overflow_pct_min"][p],
                later["overflow_pct_max"][p],
                d["overflow_pp"],
                flame["prefix_block"]["token_drop_pct"][p],
                last[p],
                d["token_drop_pp"],
            )
        )
    lines += [
        "",
        "Tokens streamed: **{:,}**.".format(later["n_tokens_streamed"]),
        "",
        "**FLAME verdict:** {}.".format(
            "MOVED" if payload["flame_moved"] else "stable — skip the other 87 files"
        ),
        "",
        "## Takeaway",
        "",
    ]
    if payload["olmoe_moved"] or payload["flame_moved"]:
        lines.append(
            "At least one probe moved. Do not treat 250k/205k as interchangeable "
            "with the full dump without a closer look. Still do **not** download "
            "all 88 FLAME files until that look says the other layers/ckpts matter."
        )
    else:
        lines.append(
            "Neither probe moved past the gates. Stage 1 numbers stay on the 205k / "
            "250k prefix. Do not pay for the other 87 FLAME parquet files."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    olmoe = run_olmoe()
    flame = run_flame_probe()
    payload = {
        "olmoe": olmoe,
        "flame": flame,
        "olmoe_moved": _moved(olmoe["delta_pp"]),
        "flame_moved": _moved(flame["delta_pp"]),
        "gates_pp": {"overflow": MOVED_OVERFLOW_PP, "token_drop": MOVED_DROP_PP},
    }
    payload["any_moved"] = payload["olmoe_moved"] or payload["flame_moved"]
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    report = _render(payload)
    (OUT / "REPORT.md").write_text(report)
    REPORT.write_text(report)
    print(report)
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
