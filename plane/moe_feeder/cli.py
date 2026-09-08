"""moe-feeder profile | plan | replay | experiment"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import (
    DEFAULT_TOPOLOGY,
    DEFLECTION_ARMS,
    RESERVATION_POLICIES,
    FeederConfig,
    flame_checkpoints,
    olmoe_checkpoints,
    window_and_heldout,
)
from .loaders import load_tokens
from .matrix import build_matrix, maps, profile_window
from .planner import plan_matrix
from .policy import collapse_history, score_heldout
from .runtime import replay_against_plan


def _cfg(args) -> FeederConfig:
    data = {k: getattr(args, k) for k in FeederConfig.__dataclass_fields__ if hasattr(args, k)}
    return FeederConfig(**data)


def _add_shared(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", choices=("flame", "olmoe"), default="flame")
    p.add_argument("--model", default="flame-moe-290m")
    p.add_argument("--layer", default="layer_02")
    p.add_argument("--n-tokens", dest="n_tokens", type=int, default=250_000)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--reservation", choices=RESERVATION_POLICIES, default="mean")
    p.add_argument("--deflection", choices=DEFLECTION_ARMS, default="none")
    p.add_argument("--tau", type=float, default=0.03)
    p.add_argument("--traffic-lens", dest="traffic_lens", default="membership")
    p.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    p.add_argument("--bytes-per-slot", dest="bytes_per_slot", type=int, default=2048)
    p.add_argument("-o", "--out", default="plane/output")


def cmd_profile(args) -> int:
    cfg = _cfg(args)
    ckpts = flame_checkpoints(cfg.model) if cfg.source == "flame" else olmoe_checkpoints()
    window, held = window_and_heldout(ckpts, cfg.window, cfg.freeze)
    history, used = profile_window(cfg, window)
    reserved = collapse_history(history, cfg.reservation)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "reservation.npy", reserved)
    for i, M in enumerate(history):
        np.save(dest / "hist_{}.npy".format(i), M)
    payload = {
        "config": cfg.as_dict(),
        "window": [str(c) for c in used],
        "heldout": str(held),
        "reservation": cfg.reservation,
        "shape": list(reserved.shape),
        "reserved_slots": float(reserved.sum()),
    }
    (dest / "profile.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


def cmd_plan(args) -> int:
    cfg = _cfg(args)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    if args.matrix:
        reserved = np.load(args.matrix)
    else:
        ckpts = flame_checkpoints(cfg.model) if cfg.source == "flame" else olmoe_checkpoints()
        window, _ = window_and_heldout(ckpts, cfg.window, cfg.freeze)
        history, _ = profile_window(cfg, window)
        reserved = collapse_history(history, cfg.reservation)
        np.save(dest / "reservation.npy", reserved)
    route_slots = np.load(args.route_matrix) if args.route_matrix else None
    result = plan_matrix(
        reserved,
        cfg.topology,
        bytes_per_slot=1 if args.already_bytes else cfg.bytes_per_slot,
        envelope_slot=cfg.reservation,
        out_dir=str(dest),
        route_slots=route_slots,
    )
    summary = {k: result[k] for k in result if k != "plan"}
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_profile_et(args) -> int:
    """Build a PLANE reservation from one or more Chakra workload directories."""
    from .chakra import matrix_from_chakra

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    history = []
    trace_stats = []
    expected_shape = None
    for index, trace in enumerate(args.trace):
        matrix, stats = matrix_from_chakra(
            trace,
            topology=args.topology,
            strict=args.strict,
        )
        if expected_shape is None:
            expected_shape = matrix.shape
        elif matrix.shape != expected_shape:
            raise ValueError(
                "all Chakra profile traces must use the same ranks: expected {}, got "
                "{} for {!r}".format(expected_shape, matrix.shape, trace)
            )
        history.append(matrix)
        trace_stats.append(stats)
        np.save(dest / "hist_{}_bytes.npy".format(index), matrix)

    reservation = collapse_history(history, args.reservation)
    mean = np.stack(history, axis=0).astype(np.float64).mean(axis=0)
    np.save(dest / "reservation_bytes.npy", reservation)
    np.save(dest / "mean_bytes.npy", mean)
    payload = {
        "input_format": "chakra-et",
        "traces": list(args.trace),
        "reservation": args.reservation,
        "shape": list(reservation.shape),
        "reserved_bytes": int(np.rint(reservation).sum()),
        "trace_stats": trace_stats,
    }
    (dest / "profile.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


def cmd_replay(args) -> int:
    cfg = _cfg(args)
    reserved = np.load(args.reservation_file)
    ckpts = flame_checkpoints(cfg.model) if cfg.source == "flame" else olmoe_checkpoints()
    _, held = window_and_heldout(ckpts, cfg.window, cfg.freeze)
    scores, indices = load_tokens(cfg, held)
    place, src = maps(cfg, len(indices))
    stats = replay_against_plan(
        cfg, indices, scores, place, src, reserved,
        deflect=not args.drop_only,
        drop=args.drop,
    )
    env = score_heldout(build_matrix(cfg, indices, place, src), reserved)
    out = {"heldout": str(held), "replay": stats.as_dict(), "envelope": env}
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "replay.json").write_text(json.dumps(out, indent=2) + "\n")
    if stats.sent is not None:
        np.save(dest / "sent.npy", stats.sent)
    print(json.dumps(out, indent=2))
    return 0


def cmd_export_et(args) -> int:
    from .et import write_comm_et

    matrix = np.load(args.matrix)
    n = write_comm_et(
        matrix,
        args.out,
        name=args.name,
        bytes_per_slot=None if args.already_bytes else args.bytes_per_slot,
    )
    print(json.dumps({"ranks": n, "out": args.out, "name": args.name}, indent=2))
    return 0


def cmd_experiment(args) -> int:
    from .experiments import run_suite

    stages = tuple(x for x in (args.stages or "").split(",") if x) or None
    return run_suite(args.suite, Path(args.out), n_tokens=args.n_tokens, stages=stages)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plane")
    sub = parser.add_subparsers(dest="command", required=True)

    profile = sub.add_parser("profile", help="build a reservation from M checkpoints")
    _add_shared(profile)
    profile.set_defaults(func=cmd_profile)

    plan = sub.add_parser("plan", help="hand the reservation to Puppeteer")
    _add_shared(plan)
    plan.add_argument("--matrix", help="path to a saved reservation.npy")
    plan.add_argument(
        "--route-matrix",
        dest="route_matrix",
        help="optional mean matrix whose Clos paths are frozen onto --matrix sizes",
    )
    plan.add_argument(
        "--already-bytes",
        action="store_true",
        help="--matrix and --route-matrix contain bytes rather than token slots",
    )
    plan.set_defaults(func=cmd_plan)

    profile_et = sub.add_parser(
        "profile-et",
        help="build a reservation from Chakra ET workload directories",
    )
    profile_et.add_argument(
        "--trace",
        action="append",
        required=True,
        help="Chakra workload directory; repeat in profiling order",
    )
    profile_et.add_argument(
        "--topology",
        help="topology YAML required when the Chakra traces contain collectives",
    )
    profile_et.add_argument(
        "--reservation",
        choices=RESERVATION_POLICIES,
        default="p95",
    )
    profile_et.add_argument("--strict", action="store_true")
    profile_et.add_argument("-o", "--out", default="plane/output/et-profile")
    profile_et.set_defaults(func=cmd_profile_et)

    replay = sub.add_parser("replay", help="token-level deflect/drop against a reservation")
    _add_shared(replay)
    replay.add_argument("--reservation-file", required=True)
    replay.add_argument("--drop", action="store_true", help="drop tokens that cannot fit")
    replay.add_argument("--drop-only", action="store_true", help="disable deflection")
    replay.set_defaults(func=cmd_replay)

    exp = sub.add_parser("experiment", help="run the planned experiment suite")
    exp.add_argument(
        "--suite",
        choices=(
            "sanity", "deflection", "envelope", "coupled", "fullmodel-astrasim",
            "claim-v1", "admitted-live", "freeze-sweep", "clos-ecmp",
            "clos-baselines", "stage2-runtime", "deflect", "reconfig",
            "cold-start-reconfig", "packet-spray", "all",
        ),
        default="all",
    )
    exp.add_argument("-o", "--out", default="plane/output")
    exp.add_argument(
        "--stages",
        default="",
        help="stage2-runtime only: comma list of granularity,shedding,links,astrasim,clos,tightness,step,ns3 (default: all)",
    )
    exp.add_argument("--n-tokens", dest="n_tokens", type=int, default=250_000)
    exp.set_defaults(func=cmd_experiment)

    export = sub.add_parser("export-et", help="write a decided matrix as Chakra ET")
    export.add_argument("--matrix", required=True, help="npy of slots or bytes")
    export.add_argument("-o", "--out", default="plane/output/et")
    export.add_argument("--name", default="moe_dispatch")
    export.add_argument("--bytes-per-slot", dest="bytes_per_slot", type=int, default=2048)
    export.add_argument(
        "--already-bytes",
        action="store_true",
        help="matrix is already in bytes; do not scale by --bytes-per-slot",
    )
    export.set_defaults(func=cmd_export_et)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
