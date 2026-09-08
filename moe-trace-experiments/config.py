"""Central configuration for all MoE trace experiments."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_CACHE = ROOT / "data_cache"
DATA_CACHE.mkdir(exist_ok=True)

FLAME_REPO = "CMU-FLAME/FLAME-MoE-Traces"
FLAME_REPO_TYPE = "dataset"

FLAME_MODELS = {
    "flame-moe-290m": {
        "layers": [f"layer_{i:02d}" for i in range(2, 10)],   # layer_02..09 (8)
        "checkpoints": [540, 1080, 1620, 2160, 2700, 3240, 3780, 4320, 4860, 5400, 5473],
        "top_k": 6,
        "num_experts": 64,
    },
    "flame-moe-721m": {
        "layers": [f"layer_{i:02d}" for i in range(2, 13)],   # layer_02..12 (11)
        "checkpoints": [880, 1760, 2640, 3520, 4400, 5280, 6160, 7040, 7920, 8800, 8815],
        "top_k": 6,
        "num_experts": 64,
    },
    "flame-moe-1.7b": {
        "layers": [f"layer_{i:02d}" for i in range(2, 19)],   # layer_02..18 (17)
        "checkpoints": [1100, 2200, 3300, 4400, 5500, 6600, 7700, 8800, 9900, 11000, 11029],
        "top_k": 6,
        "num_experts": 64,
    },
}

FLAME_MODEL = os.environ.get("FLAME_MODEL", "flame-moe-290m")
_active = FLAME_MODELS[FLAME_MODEL]

FLAME_LAYERS = _active["layers"]
FLAME_CHECKPOINTS = _active["checkpoints"]
FLAME_FINAL_CHECKPOINT = FLAME_CHECKPOINTS[-1]
FLAME_TOP_K = _active["top_k"]
FLAME_NUM_EXPERTS = _active["num_experts"]

OLMOE_REPO = "allenai/analysis_olmoe"
OLMOE_REPO_TYPE = "dataset"
OLMOE_TOP_K = 8
OLMOE_NUM_EXPERTS = 64
OLMOE_NUM_LAYERS = 16

SAMPLE_MARGIN_FINAL = 500_000
SAMPLE_MARGIN_TREND = 250_000
SAMPLE_STABILIZATION = 250_000
SAMPLE_OLMOE = 205_000

MARGIN_THRESHOLDS = [0.01, 0.03, 0.05, 0.10]

NET_DEVICES = 8
NET_SRC_RANKS = 8
NET_PLACEMENT = "contiguous"
NET_SHARDING = "contiguous"
NET_NODE_SIZE = 4
NET_DEDUP_DEVICE = False
NET_RANDOM_SEED = 0

ENVELOPE_PERCENTILES = [90, 95, 99]

NET_RAILS = 4
SIM_ECMP_SEEDS = 256
SIM_RAIL_SWEEP = [2, 4, 8]

HETERO_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
HETERO_OFFSET = 1
