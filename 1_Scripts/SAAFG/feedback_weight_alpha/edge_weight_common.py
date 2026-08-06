#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared helpers for feedback_weight_alpha sensitivity scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SAAFG_SCRIPT_DIR = SCRIPT_DIR.parent
BASE_DIR = SCRIPT_DIR.parents[2]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"
KB_DIR = BASE_DIR / "0_Data" / "5_Knowledge_Base"
DEFAULT_EXPERIMENT_ROOT = SAAFG_ROOT / "6_Experiment_Result"
DEFAULT_ALPHA_KG_ROOT = KB_DIR / "recevograph_rag_evo_v0_2"
DEFAULT_ALPHAS = [0.0, 0.3, 0.5, 0.8]

if str(SAAFG_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SAAFG_SCRIPT_DIR))


def alpha_label(alpha: float) -> str:
    value = f"{float(alpha):.3f}".rstrip("0").rstrip(".")
    if "." not in value:
        value = f"{value}.0"
    return "alpha_" + value.replace("-", "m").replace(".", "p")


def alpha_tag(alpha: float) -> str:
    return alpha_label(alpha).replace("alpha_", "alpha")


def validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"feedback_weight_alpha must be in [0.0, 1.0], got {alpha}")
    return value


def validate_alphas(alphas: Sequence[float]) -> list[float]:
    values: list[float] = []
    seen: set[str] = set()
    for alpha in alphas:
        value = validate_alpha(alpha)
        label = alpha_label(value)
        if label in seen:
            continue
        seen.add(label)
        values.append(value)
    if not values:
        raise ValueError("At least one feedback_weight_alpha value is required.")
    return values


def parse_feedback_alpha_from_argv(default: float = 0.0) -> float:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--feedback-weight-alpha", type=float, default=default)
    alpha_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return validate_alpha(alpha_args.feedback_weight_alpha)


def feedback_alpha_kg_dir(alpha: float) -> Path:
    return DEFAULT_ALPHA_KG_ROOT / alpha_label(alpha)


def require_feedback_alpha_kg(alpha: float) -> Path:
    kg_dir = feedback_alpha_kg_dir(alpha)
    required = [
        kg_dir / "graph_nodes.json",
        kg_dir / "graph_edges.json",
        kg_dir / "graph_metadata.json",
        kg_dir / "networkx_graph.pkl",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"feedback_weight_alpha KG is not ready for {alpha_tag(alpha)}: {missing_text}. "
            "Run build_edge_weight_graphs.py first."
        )
    return kg_dir


def experiment_dir(alpha: float, run_tag: str, team: str) -> Path:
    return DEFAULT_EXPERIMENT_ROOT / f"ma_RecEvoGraphRAG_{alpha_tag(alpha)}_{run_tag}" / team


def write_json_direct(path: Path, payload: Any) -> None:
    """Write JSON directly for alpha runs when os.replace is blocked on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
