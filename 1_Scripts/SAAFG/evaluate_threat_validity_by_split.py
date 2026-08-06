#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate Red Team Task A on a benchmark split with direct JSON writes.

This wrapper keeps the original evaluator unchanged, but avoids Windows
os.replace locking issues and auto-selects case ids by split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, List

import evaluate_threat_validity as evaluator


BASE_DIR = Path(__file__).resolve().parents[2]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"
DEFAULT_REGISTRY_PATH = SAAFG_ROOT / "7_Benchmark_Package_v0_2" / "case_registry_test_1.json"
PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Red Team Task A on a selected split.")
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="test")
    parser.add_argument("--case-registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def clear_dead_local_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        value = os.getenv(key)
        if value and "127.0.0.1:9" in value:
            os.environ.pop(key, None)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def split_case_ids(registry_path: Path, split: str) -> List[str]:
    rows = read_json(registry_path).get("cases") or []
    if split == "all":
        return [row["use_case_id"] for row in rows]
    return [row["use_case_id"] for row in rows if row.get("split") == split]


def write_json_direct(path: Path, payload: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(windows_write_path(path), "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def windows_write_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and len(resolved) > 240 and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def write_csv_direct(path: Path, rows: Any, fieldnames: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(windows_write_path(path), "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main() -> None:
    args = parse_args()
    clear_dead_local_proxy_env()
    evaluator.write_json_atomic = write_json_direct
    evaluator.write_csv = write_csv_direct
    case_ids = split_case_ids(args.case_registry_path, args.split)
    if not case_ids:
        raise ValueError(f"No case ids found for split={args.split}")
    sys.argv = [
        "evaluate_threat_validity.py",
        "--predictions-path",
        str(args.predictions_path.resolve()),
        "--output-json",
        str(args.output_json.resolve()),
        "--output-csv",
        str(args.output_csv.resolve()),
        "--log-path",
        str(args.log_path.resolve()),
        "--run-tag",
        args.run_tag,
        "--case-registry-path",
        str(args.case_registry_path),
        "--case-id",
        *case_ids,
    ]
    if args.skip_probe:
        sys.argv.append("--skip-probe")
    if args.resume:
        sys.argv.append("--resume")
    evaluator.main()


if __name__ == "__main__":
    main()
