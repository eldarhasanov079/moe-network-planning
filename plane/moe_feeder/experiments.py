"""Experiment series from the plan: sanity, deflection, envelope, coupled."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from .config import (
    FeederConfig,
    flame_checkpoints,
    olmoe_checkpoints,
    window_and_heldout,
)
from .loaders import load_tokens
from .matrix import build_matrix, maps, profile_window
from .planner import plan_matrix
from .policy import collapse_history, cosine, rank1_residual, score_heldout
from .runtime import replay_against_plan


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _base(n_tokens: int, **kwargs) -> FeederConfig:
    cfg = FeederConfig(n_tokens=n_tokens, **kwargs)
    return cfg


def run_sanity(out: Path, n_tokens: int) -> List[Dict]:
    """Uniform all-to-all vs heterogeneous mean matrix on one FLAME layer."""
    cfg = _base(n_tokens, layer="layer_02", deflection="none", traffic_lens="membership")
    ckpts = flame_checkpoints(cfg.model)
    window, held = window_and_heldout(ckpts, cfg.window)
    history, _ = profile_window(cfg, window)
    rows = []
    for name in ("uniform", "mean"):
        reserved = collapse_history(history, name)
        planned = plan_matrix(
            reserved, cfg.topology, cfg.bytes_per_slot, envelope_slot=name,
            out_dir=str(out / "sanity" / name),
        )
        scores, indices = load_tokens(cfg, held)
        place, src = maps(cfg, len(indices))
        actual = build_matrix(cfg, indices, place, src)
        env = score_heldout(actual, reserved)
        rows.append({
            "suite": "sanity",
            "reservation": name,
            "heldout": held,
            "n_flows": planned["n_flows"],
            "flow_bytes": planned["flow_bytes"],
            "iteration_time_s": planned["iteration_time_s"],
            "exposed_comm_s": planned["exposed_comm_s"],
            "ideal_iteration_time_s": planned["ideal_iteration_time_s"],
            "plan_wall_s": planned["plan_wall_s"],
            **env,
            "rank1_residual": rank1_residual(actual),
        })
    _write_csv(out / "sanity" / "results.csv", rows)
    return rows


def run_deflection(out: Path, n_tokens: int) -> List[Dict]:
    """none / online_global / puppeteer_choose at a few tau values."""
    cfg0 = _base(n_tokens, layer="layer_02", traffic_lens="primary")
    ckpts = flame_checkpoints(cfg0.model)
    window, held = window_and_heldout(ckpts, cfg0.window)
    rows = []
    for arm in ("none", "online_global", "puppeteer_choose"):
        taus = (0.0,) if arm == "none" else (0.01, 0.03, 0.05)
        for tau in taus:
            cfg = _base(
                n_tokens, layer="layer_02", traffic_lens="primary",
                deflection=arm, tau=tau,
            )
            history, _ = profile_window(cfg, window)
            reserved = collapse_history(history, "mean")
            planned = plan_matrix(
                reserved, cfg.topology, cfg.bytes_per_slot, envelope_slot="mean",
                out_dir=str(out / "deflection" / "{}_t{}".format(arm, tau)),
            )
            scores, indices = load_tokens(cfg, held)
            place, src = maps(cfg, len(indices))
            from .deflection import apply_deflection

            dec = apply_deflection(cfg, indices, scores, place, src)
            actual = build_matrix(cfg, dec.indices, place, src)
            env = score_heldout(actual, reserved)
            rows.append({
                "suite": "deflection",
                "arm": arm,
                "tau": tau,
                "frac_deflected": dec.frac_deflected,
                "frac_eligible": dec.frac_eligible,
                "cost": dec.cost,
                "n_flows": planned["n_flows"],
                "iteration_time_s": planned["iteration_time_s"],
                "exposed_comm_s": planned["exposed_comm_s"],
                "plan_wall_s": planned["plan_wall_s"],
                **env,
            })
    _write_csv(out / "deflection" / "results.csv", rows)
    return rows


def run_envelope(out: Path, n_tokens: int) -> List[Dict]:
    """mean / p95 / worst on FLAME and OLMoE (no deflection)."""
    rows = []
    jobs = [
        ("flame", "flame-moe-290m", "layer_02", flame_checkpoints("flame-moe-290m")),
        ("olmoe", "olmoe", "layer_0", olmoe_checkpoints()),
    ]
    for source, model, layer, ckpts in jobs:
        n = 205_000 if source == "olmoe" else n_tokens
        cfg = _base(
            n, source=source, model=model, layer=layer,
            deflection="none", traffic_lens="membership",
        )
        window, held = window_and_heldout(ckpts, min(cfg.window, max(len(ckpts) - 1, 1)))
        history, _ = profile_window(cfg, window)
        scores, indices = load_tokens(cfg, held)
        place, src = maps(cfg, len(indices))
        actual = build_matrix(cfg, indices, place, src)
        mean = collapse_history(history, "mean")
        for name in ("mean", "p95", "worst"):
            reserved = collapse_history(history, name)
            route = None if name == "mean" else mean
            planned = plan_matrix(
                reserved, cfg.topology, cfg.bytes_per_slot, envelope_slot=name,
                out_dir=str(out / "envelope" / source / name),
                route_slots=route,
            )
            env = score_heldout(actual, reserved)
            rows.append({
                "suite": "envelope",
                "source": source,
                "reservation": name,
                "heldout": str(held),
                "window_cosine": cosine(history[-1], actual) if history else 0.0,
                "n_flows": planned["n_flows"],
                "iteration_time_s": planned["iteration_time_s"],
                "exposed_comm_s": planned["exposed_comm_s"],
                "plan_wall_s": planned["plan_wall_s"],
                **env,
            })
    _write_csv(out / "envelope" / "results.csv", rows)
    return rows


def run_coupled(out: Path, n_tokens: int) -> List[Dict]:
    """Deflect on the profile window, then envelope-plan the deflected matrices."""
    rows = []
    ckpts = flame_checkpoints("flame-moe-290m")
    window, held = window_and_heldout(ckpts, 4)
    for arm, tau in (("none", 0.0), ("online_global", 0.03)):
        cfg = _base(
            n_tokens, layer="layer_02", traffic_lens="primary",
            deflection=arm, tau=tau,
        )
        history, _ = profile_window(cfg, window)
        mean = collapse_history(history, "mean")
        for name in ("mean", "p95"):
            reserved = collapse_history(history, name)
            planned = plan_matrix(
                reserved, cfg.topology, cfg.bytes_per_slot, envelope_slot=name,
                out_dir=str(out / "coupled" / "{}_{}".format(arm, name)),
                route_slots=None if name == "mean" else mean,
            )
            scores, indices = load_tokens(cfg, held)
            place, src = maps(cfg, len(indices))
            replay = replay_against_plan(
                cfg, indices, scores, place, src, reserved, deflect=arm != "none"
            )
            actual = build_matrix(cfg, indices, place, src)
            env = score_heldout(actual, reserved)
            if replay.sent is not None:
                from .et import write_comm_et

                write_comm_et(
                    replay.sent,
                    str(out / "coupled" / "{}_{}".format(arm, name)),
                    name="decided",
                    bytes_per_slot=cfg.bytes_per_slot,
                )
            rows.append({
                "suite": "coupled",
                "arm": arm,
                "tau": tau,
                "reservation": name,
                "iteration_time_s": planned["iteration_time_s"],
                "plan_wall_s": planned["plan_wall_s"],
                "reserved_slots": float(reserved.sum()),
                **replay.as_dict(),
                **env,
            })
    _write_csv(out / "coupled" / "results.csv", rows)
    return rows


def run_suite(suite: str, out: Path, n_tokens: int = 250_000, stages=None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    collected: Dict[str, List[Dict]] = {}
    if suite == "packet-spray":
        from .packet_spray_experiment import run_packet_spray_experiment

        collected[suite] = [run_packet_spray_experiment(out / "packet-spray")]
        _write_json(out / "summary.json", {k: len(v) for k, v in collected.items()})
        return 0
    if suite == "cold-start-reconfig":
        from .cold_start import run_cold_start_reconfig

        collected[suite] = [run_cold_start_reconfig(out / "cold-start-reconfig")]
        _write_json(out / "summary.json", {k: len(v) for k, v in collected.items()})
        return 0
    if suite in ("deflect", "reconfig"):
        from .stage3 import run_deflect, run_reconfig

        fn = run_deflect if suite == "deflect" else run_reconfig
        collected[suite] = [fn(out / "stage3")]
        _write_json(out / "summary.json", {k: len(v) for k, v in collected.items()})
        return 0
    if suite in ("stage2-runtime",):
        from .stage2 import ALL_STAGES, run_stage2

        collected["stage2-runtime"] = [run_stage2(out / "stage2", stages=stages or ALL_STAGES)]
        _write_json(out / "summary.json", {k: len(v) for k, v in collected.items()})
        return 0
    if suite in ("sanity", "all"):
        collected["sanity"] = run_sanity(out, n_tokens)
    if suite in ("deflection", "all"):
        collected["deflection"] = run_deflection(out, n_tokens)
    if suite in ("envelope", "all"):
        collected["envelope"] = run_envelope(out, n_tokens)
    if suite in ("coupled", "all"):
        collected["coupled"] = run_coupled(out, n_tokens)
    if suite in ("fullmodel-astrasim",):
        from .fullmodel import run_fullmodel_astrasim

        collected["fullmodel-astrasim"] = run_fullmodel_astrasim(
            out / "fullmodel-astrasim", n_tokens
        )
    if suite in ("claim-v1", "admitted-live", "freeze-sweep", "clos-ecmp"):
        from .claim_v1 import run_claim_v1

        collected["claim-v1"] = [run_claim_v1(out / "claim-v1")]
    if suite in ("clos-baselines",):
        from .clos_baselines import run_clos_baselines

        collected["clos-baselines"] = [run_clos_baselines(out / "clos-baselines")]
    _write_json(out / "summary.json", collected)
    print(json.dumps({k: len(v) for k, v in collected.items()}, indent=2))
    return 0
