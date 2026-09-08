"""Structural side-statistics behind Experiment D's "why only ~40%" paragraph."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "plane"))

from moe_feeder.admit import admit_by_slot  # noqa: E402
from moe_feeder.policy import collapse_history  # noqa: E402
from moe_feeder.reconfig import MODELS, load_tokens, maps  # noqa: E402
from moe_feeder.stage2_blocks import block_matrix, scale_reservation, step_blocks  # noqa: E402
from moe_feeder.stage3 import HEADLINE, PLACEMENTS  # noqa: E402

OUT = ROOT / "plane/output/stage3/deflect_neighbourhood.json"
SAMPLE = 20_000


def _dist(vals) -> dict:
    v = [float(x) for x in vals]
    return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n_layers": len(v)}


def main() -> None:
    models = {}
    for model in ("flame-moe-290m", "flame-moe-721m", "flame-moe-1.7b", "olmoe"):
        spec = MODELS[model]
        N, K, B = spec["n_tokens"], spec["window"], HEADLINE[spec["source"]]
        ckpt = spec["ckpts"][-1]
        entry = {"same_device": {}}
        for placement, seed in PLACEMENTS:
            place, _ = maps(model, placement, seed)
            vals = []
            for layer in spec["layers"]:
                ind, _ = load_tokens(model, layer, ckpt)
                dev = place[ind]
                k = ind.shape[1]
                vals.append(float(np.mean((dev[:, :k - 1] == dev[:, k - 1:k]).any(axis=1))))
            entry["same_device"]["{}{}".format(placement, seed if placement == "random" else "")] = _dist(vals)

        place, src_full = maps(model)
        chance, multi = [], []
        for layer in spec["layers"]:
            ind, _ = load_tokens(model, layer, ckpt)
            k = ind.shape[1]
            prim = place[ind][:SAMPLE, :k - 1]
            chance.append(float(np.mean([len(set(r)) for r in prim]) / 8.0))
            E = collapse_history(
                [block_matrix(load_tokens(model, layer, c)[0][:, :k - 1], place, src_full, 8) for c in spec["ckpts"][:K]],
                "p95")
            idx, srcb = step_blocks(N, B, 8)[0]
            adm = admit_by_slot(ind[idx][:, :k - 1], place, srcb, scale_reservation(E, B, N),
                                order="arrival", overflow="drop")
            over = np.asarray(adm.dropped)
            per_tok = over.sum(axis=1)
            tot = float(per_tok.sum())
            multi.append(float((per_tok - 1)[per_tok >= 2].sum() / tot) if tot else 0.0)
        entry["chance_level"] = _dist(chance)
        entry["multi_overflow"] = _dist(multi)
        models[model] = entry
        print("{:16s} same_device={:.3f}  chance={:.3f}  multi_overflow={:.3f}".format(
            model, entry["same_device"]["contiguous"]["mean"], entry["chance_level"]["mean"], entry["multi_overflow"]["mean"]))

    OUT.write_text(json.dumps({
        "note": "structural side-statistics for Experiment D; see plane/scripts/deflect_neighbourhood.py",
        "checkpoint": "last of each model", "step_tokens": HEADLINE, "models": models}, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
