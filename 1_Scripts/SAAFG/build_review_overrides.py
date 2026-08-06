#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROOT / "0_Data" / "6_SAAFG"
CRITIC_ROOT = BENCHMARK_ROOT / "4_Critic_Reports"
HUMAN_CHECK_ROOT = BENCHMARK_ROOT / "5_Gold_or_Human_Check"

THREAT_PATH = BENCHMARK_ROOT / "2_RedTeam_Threat_Records" / "threat_records.json"
FLOW_PATH = BENCHMARK_ROOT / "3_BlueTeam_SA_Flows" / "security_augmented_use_case_flows.json"
ANCHOR_REVIEW_PATH = CRITIC_ROOT / "saafg_anchor_adjudication_v0_2_manual_review.json"
DEFENSE_REVIEW_PATH = CRITIC_ROOT / "saafg_defense_insertion_adjudication_v0_2_manual_review.json"
DEFENSE_REWRITE_PATH = CRITIC_ROOT / "saafg_defense_template_rewrite_v0_2.json"

THREAT_OVERRIDE_OUTPUT = HUMAN_CHECK_ROOT / "saafg_manual_threat_record_overrides_v0_2.json"
SECURITY_OVERRIDE_OUTPUT = HUMAN_CHECK_ROOT / "saafg_security_flow_overrides_v0_2.json"


def now_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    threats = read_json(THREAT_PATH)["threat_record_cases"]
    flows = read_json(FLOW_PATH)["security_augmented_flow_cases"]
    anchor_review = read_json(ANCHOR_REVIEW_PATH)
    defense_review = read_json(DEFENSE_REVIEW_PATH)
    defense_rewrite = read_json(DEFENSE_REWRITE_PATH)

    threat_map = {case["use_case_id"]: case for case in threats}
    flow_map = {case["use_case_id"]: case for case in flows}

    anchor_overrides = [
        {
            "use_case_id": row["use_case_id"],
            "threat_id": row["threat_id"],
            "revised_anchor": row["revised_anchor"],
            "rationale": row["rationale"],
        }
        for row in anchor_review["recommended_anchor_changes"]
    ]

    manual_cases = ["UC0108"]
    manual_threat_case_overrides = []
    for case_id in manual_cases:
        case = threat_map[case_id]
        manual_threat_case_overrides.append(
            {
                "use_case_id": case_id,
                "dataset": case["dataset"],
                "source_knowledge_id": case["source_knowledge_id"],
                "threat_records": case["threat_records"],
                "rationale": "Manual reviewed threat record override retained in the v0.2 freeze to avoid an empty-threat case and preserve reviewed anchor semantics.",
            }
        )

    write_json(
        THREAT_OVERRIDE_OUTPUT,
        {
            "meta": {
                "dataset_name": "SAAFG manual threat-record overrides",
                "version": "v0.2",
                "generated_at_utc": now_utc(),
                "anchor_override_count": len(anchor_overrides),
                "manual_threat_case_override_count": len(manual_threat_case_overrides),
                "notes": [
                    "anchor_overrides are applied to the silver threat record generation result after heuristic candidate selection.",
                    "manual_threat_case_overrides replace the full threat list for explicitly reviewed exceptions such as UC0108.",
                ],
            },
            "anchor_overrides": anchor_overrides,
            "manual_threat_case_overrides": manual_threat_case_overrides,
        },
    )

    branch_keys = set()
    for row in defense_rewrite["rewrites"]:
        branch_keys.add((row["use_case_id"], row["threat_id"]))
    for row in defense_review["must_change_cases"]:
        for threat_id in row["flagged_threat_ids"]:
            branch_keys.add((row["use_case_id"], threat_id))
    branch_keys.add(("UC0108", "T1"))

    branch_overrides = []
    for case_id, threat_id in sorted(branch_keys):
        case = flow_map[case_id]["security_augmented_flow"]
        saf = None
        for branch in case["security_alternative_flows"]:
            if threat_id in (branch.get("mitigates") or []):
                saf = branch
                break
        if saf is None:
            raise KeyError(f"Missing SAF for {case_id} {threat_id}")
        sbf_id = saf["entry_condition"].split()[0]
        sbf = None
        for item in case["security_basic_flow"]:
            if item["step_id"] == sbf_id:
                sbf = item
                break
        if sbf is None:
            raise KeyError(f"Missing SBF {sbf_id} for {case_id} {threat_id}")
        branch_overrides.append(
            {
                "use_case_id": case_id,
                "threat_id": threat_id,
                "sbf_id": sbf["step_id"],
                "saf_id": saf["saf_id"],
                "anchor_after": sbf["anchor_after"],
                "step_overrides": {
                    sbf["step_id"]: sbf["step_sentence"],
                    saf["steps"][0]["step_id"]: saf["steps"][0]["step_sentence"],
                    saf["steps"][1]["step_id"]: saf["steps"][1]["step_sentence"],
                },
            }
        )

    write_json(
        SECURITY_OVERRIDE_OUTPUT,
        {
            "meta": {
                "dataset_name": "SAAFG security-flow overrides",
                "version": "v0.2",
                "generated_at_utc": now_utc(),
                "branch_override_count": len(branch_overrides),
                "notes": [
                    "Each entry stores the reviewed SBF and first two SAF step sentences for a threat branch that differs from the default generator output.",
                    "The build script applies these branch overrides after generating baseline silver security-augmented flows.",
                ],
            },
            "branch_overrides": branch_overrides,
        },
    )

    print("Threat override file:", THREAT_OVERRIDE_OUTPUT)
    print("Security override file:", SECURITY_OVERRIDE_OUTPUT)


if __name__ == "__main__":
    main()
