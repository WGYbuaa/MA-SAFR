#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build a deterministic split_2.0 for SAAFG v0.2.

The original test split is structurally skewed because it consists of contiguous
source blocks. This script creates a second split without random shuffling:

- test_2.0 is selected only from the original train+dev pool.
- train_2.0/dev_2.0 are selected from the remaining original train+dev cases plus
  all original test cases.
- Selection is stratified by source group and silver threat count.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parents[2]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"
PACKAGE_DIR = SAAFG_ROOT / "7_Benchmark_Package_v0_2"
OUTPUT_DIR = SAAFG_ROOT / "7_Benchmark_Package_v0_2"

REGISTRY_PATH = PACKAGE_DIR / "case_registry_test_1.json"
THREAT_PATH = SAAFG_ROOT / "2_RedTeam_Threat_Records" / "threat_records.json"

OUT_SPLITS_PATH = OUTPUT_DIR / "splits_test_2.json"
OUT_REGISTRY_JSON_PATH = OUTPUT_DIR / "case_registry_test_2.json"
OUT_REGISTRY_CSV_PATH = OUTPUT_DIR / "case_registry_test_2.csv"
OUT_SUMMARY_PATH = OUTPUT_DIR / "split_test_2_summary.md"

TEST_SIZE = 57
DEV_SIZE = 20
SEED = "saafg_split_2_0_v20260709"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{text}".encode("utf-8")).hexdigest(), 16)


def source_group(case: Dict[str, Any]) -> str:
    if case.get("dataset") == "atlas":
        return "atlas"
    return str(case.get("source_knowledge_id"))


def largest_remainder_quotas(group_sizes: Dict[str, int], target_total: int) -> Dict[str, int]:
    total = sum(group_sizes.values())
    if total <= 0:
        return {}
    raw: Dict[str, float] = {key: value * target_total / total for key, value in group_sizes.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = target_total - sum(quotas.values())
    ranked = sorted(
        group_sizes,
        key=lambda key: (raw[key] - quotas[key], group_sizes[key], -stable_hash(key)),
        reverse=True,
    )
    for key in ranked[:remaining]:
        quotas[key] += 1
    return quotas


def target_threat_distribution(cases: Sequence[Dict[str, Any]], target_total: int) -> Dict[int, int]:
    sizes = Counter(int(case["silver_threat_count"]) for case in cases)
    return largest_remainder_quotas({str(key): value for key, value in sizes.items()}, target_total)  # type: ignore[return-value]


def normalize_threat_distribution(dist: Dict[Any, int]) -> Dict[int, int]:
    return {int(key): int(value) for key, value in dist.items()}


def select_stratified_cases(
    candidates: Sequence[Dict[str, Any]],
    target_size: int,
    *,
    quota_key: str,
) -> List[Dict[str, Any]]:
    """Select cases by group quota while matching threat-count distribution."""

    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in candidates:
        by_group[str(case[quota_key])].append(case)

    group_sizes = {group: len(items) for group, items in by_group.items()}
    group_quotas = largest_remainder_quotas(group_sizes, target_size)
    threat_targets = normalize_threat_distribution(target_threat_distribution(candidates, target_size))
    threat_remaining = dict(threat_targets)
    group_remaining = dict(group_quotas)
    selected: List[Dict[str, Any]] = []
    selected_ids = set()

    sorted_candidates = sorted(
        candidates,
        key=lambda case: (
            -int(case["silver_threat_count"]),
            str(case["source_knowledge_id"]),
            stable_hash(str(case["use_case_id"])),
        ),
    )

    while len(selected) < target_size:
        best_case = None
        best_score: Tuple[int, int, int, int, int] | None = None
        for case in sorted_candidates:
            case_id = str(case["use_case_id"])
            group = str(case[quota_key])
            threat_count = int(case["silver_threat_count"])
            if case_id in selected_ids or group_remaining.get(group, 0) <= 0:
                continue

            threat_deficit = threat_remaining.get(threat_count, 0)
            group_deficit = group_remaining.get(group, 0)
            score = (
                1 if threat_deficit > 0 else 0,
                threat_deficit,
                group_deficit,
                threat_count,
                -stable_hash(case_id),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_case = case

        if best_case is None:
            raise RuntimeError("No selectable case found before reaching target size.")

        selected.append(best_case)
        selected_ids.add(str(best_case["use_case_id"]))
        selected_group = str(best_case[quota_key])
        selected_threat_count = int(best_case["silver_threat_count"])
        group_remaining[selected_group] -= 1
        threat_remaining[selected_threat_count] = threat_remaining.get(selected_threat_count, 0) - 1

    return sorted(selected, key=lambda case: str(case["use_case_id"]))


def summarize_split(cases: Sequence[Dict[str, Any]], ids: Iterable[str]) -> Dict[str, Any]:
    id_set = set(ids)
    selected = [case for case in cases if case["use_case_id"] in id_set]
    return {
        "case_count": len(selected),
        "silver_threat_total": sum(int(case["silver_threat_count"]) for case in selected),
        "dataset_distribution": dict(Counter(case["dataset"] for case in selected)),
        "source_group_distribution": dict(Counter(case["source_group"] for case in selected)),
        "source_knowledge_distribution": dict(Counter(case["source_knowledge_id"] for case in selected)),
        "silver_threat_count_distribution": dict(
            sorted(Counter(int(case["silver_threat_count"]) for case in selected).items())
        ),
        "old_split_distribution": dict(Counter(case["old_split"] for case in selected)),
    }


def write_registry_csv(path: Path, cases: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "use_case_id",
        "dataset",
        "old_split",
        "split",
        "split_version",
        "source_knowledge_id",
        "source_title",
        "silver_threat_count",
        "basic_flow_step_count",
        "alternative_flow_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for case in sorted(cases, key=lambda item: item["use_case_id"]):
            writer.writerow({key: case.get(key, "") for key in fieldnames})


def main() -> None:
    registry = load_json(REGISTRY_PATH)["cases"]
    threat_cases = load_json(THREAT_PATH)["threat_record_cases"]
    threat_counts = {
        case["use_case_id"]: len(case.get("threat_records") or [])
        for case in threat_cases
    }

    enriched: List[Dict[str, Any]] = []
    for case in registry:
        new_case = dict(case)
        new_case["old_split"] = case["split"]
        new_case["split_version"] = "2.0"
        new_case["silver_threat_count"] = int(threat_counts.get(case["use_case_id"], 0))
        new_case["source_group"] = source_group(case)
        enriched.append(new_case)

    old_train_dev = [case for case in enriched if case["old_split"] in {"train", "dev"}]
    test_2 = select_stratified_cases(old_train_dev, TEST_SIZE, quota_key="source_group")
    test_2_ids = {case["use_case_id"] for case in test_2}

    train_dev_pool = [case for case in enriched if case["use_case_id"] not in test_2_ids]
    dev_2 = select_stratified_cases(train_dev_pool, DEV_SIZE, quota_key="source_group")
    dev_2_ids = {case["use_case_id"] for case in dev_2}
    train_2 = [
        case
        for case in enriched
        if case["use_case_id"] not in test_2_ids and case["use_case_id"] not in dev_2_ids
    ]
    train_2_ids = {case["use_case_id"] for case in train_2}

    assert len(test_2_ids) == TEST_SIZE
    assert len(dev_2_ids) == DEV_SIZE
    assert len(train_2_ids) == len(enriched) - TEST_SIZE - DEV_SIZE
    assert not (test_2_ids & dev_2_ids or test_2_ids & train_2_ids or dev_2_ids & train_2_ids)

    split_map = {
        "train": train_2_ids,
        "dev": dev_2_ids,
        "test": test_2_ids,
    }
    registry_2 = []
    for case in enriched:
        updated = dict(case)
        for split, ids in split_map.items():
            if case["use_case_id"] in ids:
                updated["split"] = split
                break
        registry_2.append(updated)

    splits_payload = {
        "meta": {
            "split_version": "2.0",
            "construction_rule": (
                "test_2.0 selected from original train+dev using deterministic stratification "
                "over source groups and silver threat counts; train_2.0/dev_2.0 selected from "
                "remaining cases plus all original test cases."
            ),
            "seed": SEED,
            "test_size": TEST_SIZE,
            "dev_size": DEV_SIZE,
            "train_size": len(train_2_ids),
        },
        "train": sorted(train_2_ids),
        "dev": sorted(dev_2_ids),
        "test": sorted(test_2_ids),
        "train_2.0": sorted(train_2_ids),
        "dev_2.0": sorted(dev_2_ids),
        "test_2.0": sorted(test_2_ids),
    }

    summary = {
        "train_2.0": summarize_split(registry_2, train_2_ids),
        "dev_2.0": summarize_split(registry_2, dev_2_ids),
        "test_2.0": summarize_split(registry_2, test_2_ids),
    }

    OUT_SPLITS_PATH.write_text(json.dumps(splits_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_REGISTRY_JSON_PATH.write_text(
        json.dumps({"meta": splits_payload["meta"], "cases": registry_2}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_registry_csv(OUT_REGISTRY_CSV_PATH, registry_2)

    lines = [
        "# SAAFG split_2.0 Summary",
        "",
        "## Construction Rule",
        "",
        splits_payload["meta"]["construction_rule"],
        "",
        "The split is deterministic and does not use a random shuffle. `test_2.0` is drawn only from the original train+dev pool.",
        "",
    ]
    for split_name in ["train_2.0", "dev_2.0", "test_2.0"]:
        item = summary[split_name]
        lines.extend(
            [
                f"## {split_name}",
                "",
                f"- Cases: {item['case_count']}",
                f"- Silver threats: {item['silver_threat_total']}",
                f"- Dataset distribution: `{item['dataset_distribution']}`",
                f"- Old split distribution: `{item['old_split_distribution']}`",
                f"- Source group distribution: `{item['source_group_distribution']}`",
                f"- Silver threat count distribution: `{item['silver_threat_count_distribution']}`",
                "",
            ]
        )
    OUT_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"[Done] splits={OUT_SPLITS_PATH}")
    print(f"[Done] registry_json={OUT_REGISTRY_JSON_PATH}")
    print(f"[Done] registry_csv={OUT_REGISTRY_CSV_PATH}")
    print(f"[Done] summary={OUT_SUMMARY_PATH}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
