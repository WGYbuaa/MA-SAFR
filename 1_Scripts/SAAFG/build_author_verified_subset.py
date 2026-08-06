#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROOT / "0_Data" / "6_SAAFG"
HUMAN_CHECK_ROOT = BENCHMARK_ROOT / "5_Gold_or_Human_Check"
PACKAGE_ROOT = BENCHMARK_ROOT / "7_Benchmark_Package_v0_2"
CRITIC_ROOT = BENCHMARK_ROOT / "4_Critic_Reports"

CASE_REGISTRY_PATH = PACKAGE_ROOT / "case_registry_test_1.json"
METADATA_PATH = PACKAGE_ROOT / "safr_bench_metadata.json"
README_PATH = PACKAGE_ROOT / "README.md"
MANIFEST_PATH = PACKAGE_ROOT / "release_manifest.json"

FLOW_PATH = BENCHMARK_ROOT / "1_Input_Functional_Flows" / "functional_use_case_flows.json"
THREAT_PATH = BENCHMARK_ROOT / "2_RedTeam_Threat_Records" / "threat_records.json"
SECURITY_FLOW_PATH = BENCHMARK_ROOT / "3_BlueTeam_SA_Flows" / "security_augmented_use_case_flows.json"

ANCHOR_REVIEW_PATH = CRITIC_ROOT / "saafg_anchor_adjudication_v0_2_manual_review.json"
DEFENSE_REVIEW_PATH = CRITIC_ROOT / "saafg_defense_insertion_adjudication_v0_2_manual_review.json"
DEFENSE_REWRITE_PATH = CRITIC_ROOT / "saafg_defense_template_rewrite_v0_2.json"
OPTIONAL_ANCHOR_PATH = HUMAN_CHECK_ROOT / "optional_anchor_adjudication.json"

OUTPUT_JSON_PATH = HUMAN_CHECK_ROOT / "author_verified_subset.json"
OUTPUT_MD_PATH = HUMAN_CHECK_ROOT / "author_verified_subset_notes.md"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    registry = read_json(CASE_REGISTRY_PATH)["cases"]
    flows = read_json(FLOW_PATH)["use_case_flows"]
    threats = read_json(THREAT_PATH)["threat_record_cases"]
    security_flows = read_json(SECURITY_FLOW_PATH)["security_augmented_flow_cases"]

    anchor_review = read_json(ANCHOR_REVIEW_PATH)
    defense_review = read_json(DEFENSE_REVIEW_PATH)
    defense_rewrite = read_json(DEFENSE_REWRITE_PATH)
    optional_anchor = read_json(OPTIONAL_ANCHOR_PATH)

    registry_map = {row["use_case_id"]: row for row in registry}
    flow_map = {row["use_case_id"]: row for row in flows}
    threat_map = {row["use_case_id"]: row for row in threats}
    security_map = {row["use_case_id"]: row for row in security_flows}

    anchor_changes = {}
    for row in anchor_review["recommended_anchor_changes"]:
        anchor_changes.setdefault(row["use_case_id"], []).append(
            {
                "threat_id": row["threat_id"],
                "threat_name": row["threat_name"],
                "current_anchor": row["current_anchor"],
                "revised_anchor": row["revised_anchor"],
                "rationale": row["rationale"],
            }
        )

    must_change = {}
    for row in defense_review["must_change_cases"]:
        must_change[row["use_case_id"]] = {
            "flagged_threat_ids": row["flagged_threat_ids"],
            "reason": row["reason"],
        }

    defense_rewrites = {}
    for row in defense_rewrite["rewrites"]:
        defense_rewrites.setdefault(row["use_case_id"], []).append(
            {
                "threat_id": row["threat_id"],
                "threat_name": row["threat_name"],
                "sbf_id": row["sbf_id"],
                "saf_id": row["saf_id"],
            }
        )

    manual_patch_cases = {
        "UC0108": {
            "summary": "A non-empty composite ATLAS threat was manually added and anchored at BF6 to eliminate the previous empty-threat exception.",
            "threat_id": "T1",
        }
    }

    subset_case_ids = sorted(
        set(anchor_changes) | set(must_change) | set(defense_rewrites) | set(manual_patch_cases)
    )

    gold_subset_cases = []
    dataset_counter = Counter()
    split_counter = Counter()
    reviewed_layer_counter = Counter()

    for case_id in subset_case_ids:
        case_meta = registry_map[case_id]
        dataset_counter[case_meta["dataset"]] += 1
        split_counter[case_meta["split"]] += 1

        review_tags = []
        review_notes = []
        reviewed_layers = []

        if case_id in anchor_changes:
            review_tags.append("anchor_revised")
            reviewed_layers.append("threat_records")
            for item in anchor_changes[case_id]:
                review_notes.append(
                    f"{item['threat_id']} anchor revised from {item['current_anchor']} to {item['revised_anchor']}."
                )
        if case_id in manual_patch_cases:
            review_tags.append("manual_composite_threat_patch")
            reviewed_layers.append("threat_records")
            review_notes.append(manual_patch_cases[case_id]["summary"])
        if case_id in must_change:
            review_tags.append("defense_family_rewritten")
            reviewed_layers.append("security_augmented_flow")
            review_notes.append(must_change[case_id]["reason"])
        if case_id in defense_rewrites:
            review_tags.append("defense_wording_rewritten")
            reviewed_layers.append("security_augmented_flow")
            threat_ids = sorted({item["threat_id"] for item in defense_rewrites[case_id]})
            review_notes.append(
                "Case-specific SBF/SAF wording was rewritten for {}.".format(", ".join(threat_ids))
            )

        reviewed_layers = sorted(set(reviewed_layers))
        for layer in reviewed_layers:
            reviewed_layer_counter[layer] += 1

        gold_subset_cases.append(
            {
                "use_case_id": case_id,
                "dataset": case_meta["dataset"],
                "split": case_meta["split"],
                "source_knowledge_id": case_meta["source_knowledge_id"],
                "source_title": case_meta["source_title"],
                "review_status": "author_verified_gold_subset_v0_2",
                "review_disclaimer": (
                    "This subset reflects author-side expert adjudication in the current review round. "
                    "It is not independent third-party expert certification."
                ),
                "review_tags": sorted(set(review_tags)),
                "reviewed_layers": reviewed_layers,
                "review_notes": review_notes,
                "review_provenance": {
                    "anchor_changes": anchor_changes.get(case_id, []),
                    "manual_defense_family_review": must_change.get(case_id),
                    "defense_wording_rewrites": defense_rewrites.get(case_id, []),
                    "manual_patch": manual_patch_cases.get(case_id),
                },
                "functional_flow": flow_map[case_id],
                "reviewed_threat_records": threat_map[case_id]["threat_records"],
                "reviewed_security_augmented_flow": security_map[case_id]["security_augmented_flow"],
            }
        )

    generated_at = now_utc()
    output_json = {
        "meta": {
            "dataset_name": "SAAFG-Bench author-verified gold subset",
            "version": "v0.2",
            "generated_at_utc": generated_at,
            "case_count": len(gold_subset_cases),
            "dataset_breakdown": {
                "owasp": dataset_counter.get("owasp", 0),
                "atlas": dataset_counter.get("atlas", 0),
            },
            "split_breakdown": {
                "train": split_counter.get("train", 0),
                "dev": split_counter.get("dev", 0),
                "test": split_counter.get("test", 0),
            },
            "reviewed_layer_breakdown": {
                "threat_records": reviewed_layer_counter.get("threat_records", 0),
                "security_augmented_flow": reviewed_layer_counter.get("security_augmented_flow", 0),
            },
            "selection_policy": [
                "This subset contains cases whose canonical threat records or security-augmented flows were directly revised in the current review round.",
                "Cases that were only manually reviewed but kept unchanged are not included in this subset.",
                "The subset is suitable as an author-verified evaluation slice and human-check package for v0.2."
            ],
            "source_artifacts": [
                "0_Data/6_SAAFG/4_Critic_Reports/saafg_anchor_adjudication_v0_2_manual_review.json",
                "0_Data/6_SAAFG/4_Critic_Reports/saafg_defense_insertion_adjudication_v0_2_manual_review.json",
                "0_Data/6_SAAFG/4_Critic_Reports/saafg_defense_template_rewrite_v0_2.json",
                "0_Data/6_SAAFG/5_Gold_or_Human_Check/optional_anchor_adjudication.json"
            ],
            "notes": [
                "This file should be described as an author-verified gold subset under the current repository guideline.",
                "If additional independent human expert review is completed later, this subset can be promoted or renamed accordingly."
            ],
        },
        "gold_subset_cases": gold_subset_cases,
    }
    write_json(OUTPUT_JSON_PATH, output_json)

    optional_anchor_case_count = optional_anchor["meta"]["case_count"]
    lines = [
        "# SAAFG Author-Verified Gold Subset Review Notes v0.2",
        "",
        "This file accompanies `author_verified_subset.json`.",
        "",
        "The subset contains cases whose canonical benchmark artifacts were directly revised in the current review round.",
        "",
        "It is suitable as:",
        "- an author-verified v0.2 gold subset",
        "- a human-check package for focused evaluation",
        "- a stable benchmark slice for ablation or regression testing",
        "",
        "It should not be described as independent third-party expert gold without additional external human confirmation.",
        "",
        "Current scope:",
        f"- revised-case count: {len(gold_subset_cases)}",
        f"- OWASP cases: {dataset_counter.get('owasp', 0)}",
        f"- ATLAS cases: {dataset_counter.get('atlas', 0)}",
        f"- optional-anchor sidecar cases kept outside this subset: {optional_anchor_case_count}",
        "",
    ]
    write_text(OUTPUT_MD_PATH, "\n".join(lines))

    metadata = read_json(METADATA_PATH)
    metadata["generated_at_utc"] = generated_at
    metadata["empty_threat_case_count"] = 0
    metadata["empty_threat_case_ids"] = []
    metadata["human_check_breakdown"] = {
        "legacy_ai_reviewed_gold_subset_v0_1": 12,
        "author_verified_gold_subset_v0_2": len(gold_subset_cases),
        "optional_anchor_sidecar_v0_2_case_count": optional_anchor_case_count,
    }
    metadata["notes"] = [
        "This benchmark package is built on local OWASP/ATLAS-derived assets already present in the repository.",
        "The functional flow layer is derived from parser-produced system_flow outputs after conservative cleanup.",
        "ATLAS core silver excludes attacker-side-only preparation, staging, and reconnaissance techniques unless they map to a victim-side actionable step.",
        "anchor_steps in v0.2 contains exactly one primary defense-actionable BF step.",
        "The benchmark package now includes an author-verified v0.2 gold subset covering cases directly revised in the current review round.",
        "The legacy AI-reviewed gold subset remains available for historical reference and was not auto-upgraded to v0.2 semantics.",
    ]
    write_json(METADATA_PATH, metadata)

    readme = "\n".join(
        [
            "# SAAFG-Bench v0.2",
            "",
            "## Summary",
            "",
            "This package provides the current v0.2 silver freeze for Security-Augmented Alternative Flow Generation (SAAFG).",
            "",
            "It contains:",
            "",
            "- a case registry",
            "- split files",
            "- cleaned functional flow inputs",
            "- source-grounded silver threat records",
            "- source-grounded silver security-augmented flows",
            "- schema and protocol documents",
            "- author-verified human-check artifacts under `0_Data/6_SAAFG/5_Gold_or_Human_Check`",
            "",
            "## Counts",
            "",
            f"- total cases: {metadata['case_count']}",
            f"- train cases: {metadata['split_breakdown']['train']}",
            f"- dev cases: {metadata['split_breakdown']['dev']}",
            f"- test cases: {metadata['split_breakdown']['test']}",
            "- empty-core-threat cases: 0",
            f"- author-verified gold subset cases: {len(gold_subset_cases)}",
            "- legacy AI-reviewed gold subset seed cases: 12",
            "",
            "## Human-Check Files",
            "",
            "- `author_verified_subset.json`: current round revised-case subset with reviewed flows, threat records, and security-augmented flows.",
            "- `author_verified_subset_notes.md`: scope and labeling notes for the author-verified subset.",
            "- `optional_anchor_adjudication.json`: ambiguity sidecar for acceptable alternate anchors that do not change the canonical benchmark anchor.",
            "- `saafg_ai_reviewed_gold_subset_seed_v0_1.json`: legacy AI-reviewed seed subset retained for historical comparison.",
            "",
            "## Important Note",
            "",
            "The v0.2 author-verified subset is suitable for internal evaluation, ablation, and focused human-check workflows.",
            "It should be described as author-verified rather than independent third-party expert gold unless additional external human confirmation is added later.",
            "",
        ]
    )
    write_text(README_PATH, readme)

    manifest = read_json(MANIFEST_PATH)
    public_files = manifest.get("public_files", [])
    for path in [
        "0_Data/6_SAAFG/5_Gold_or_Human_Check/author_verified_subset.json",
        "0_Data/6_SAAFG/5_Gold_or_Human_Check/author_verified_subset_notes.md",
    ]:
        if path not in public_files:
            public_files.append(path)
    manifest["public_files"] = public_files
    manifest["release_note"] = (
        "The v0.2 package includes a 60-case author-verified gold subset derived from directly revised cases in the current review round; "
        "the legacy AI-reviewed subset remains as historical seed data."
    )
    write_json(MANIFEST_PATH, manifest)

    print("Author-verified subset cases:", len(gold_subset_cases))
    print("Output JSON:", OUTPUT_JSON_PATH)
    print("Output notes:", OUTPUT_MD_PATH)


if __name__ == "__main__":
    main()
