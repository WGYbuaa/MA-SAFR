#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from safr_common import clean_source_text, first_sentence, normalize_text


ROOT = Path(__file__).resolve().parents[2]
THREAT_RECORDS_PATH = ROOT / "0_Data" / "6_SAAFG" / "2_RedTeam_Threat_Records" / "threat_records.json"
CASE_REGISTRY_PATH = ROOT / "0_Data" / "6_SAAFG" / "7_Benchmark_Package_v0_2" / "case_registry_test_1.json"
OWASP_SOURCE_PATH = ROOT / "0_Data" / "5_Knowledge_Base" / "source" / "owasp_knowledge.json"
ATLAS_SOURCE_PATH = ROOT / "0_Data" / "5_Knowledge_Base" / "source" / "mitre_atlas_knowledge.json"
OUTPUT_JSON_PATH = ROOT / "0_Data" / "6_SAAFG" / "4_Critic_Reports" / "saafg_original_threat_traceability_check_v0_2.json"
OUTPUT_MD_PATH = ROOT / "0_Data" / "6_SAAFG" / "4_Critic_Reports" / "saafg_original_threat_traceability_check_v0_2.md"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_owasp_expected_evidence(source: Dict[str, Any]) -> str:
    return normalize_text(clean_source_text(first_sentence(source.get("source_summary") or source.get("requirement_text") or "")))


def build_atlas_procedure_lookup(source: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    lookup: Dict[str, List[Dict[str, Any]]] = {}
    for proc_index, proc in enumerate(source.get("procedure", []), start=1):
        technique = proc.get("technique", {}) or {}
        tactic = proc.get("tactic", {}) or {}
        technique_id = technique.get("id")
        if not technique_id:
            continue
        procedure_text = clean_source_text(proc.get("description") or "")
        technique_text = clean_source_text(technique.get("description") or "")
        mechanism = normalize_text(first_sentence(procedure_text) or first_sentence(technique_text))
        lookup.setdefault(technique_id, []).append(
            {
                "procedure_index": proc_index,
                "mechanism": mechanism,
                "tactic_id": normalize_text(tactic.get("id") or ""),
                "tactic_name": normalize_text(tactic.get("name") or ""),
                "technique_name": normalize_text(technique.get("name") or technique_id),
            }
        )
    return lookup


def verify_owasp_threat_item(
    threat: Dict[str, Any],
    expected_source_id: str,
    expected_evidence_text: str,
) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if threat.get("source_knowledge_id") != expected_source_id:
        issues.append(
            f"threat source_knowledge_id {threat.get('source_knowledge_id')} != expected {expected_source_id}"
        )

    evidence_items = threat.get("source_evidence", [])
    if not evidence_items:
        issues.append("missing source_evidence")
    else:
        valid_evidence = False
        for evidence in evidence_items:
            evidence_ok = (
                evidence.get("evidence_type") == "owasp_category"
                and evidence.get("source_knowledge_id") == expected_source_id
                and normalize_text(evidence.get("evidence_text")) == expected_evidence_text
            )
            if evidence_ok:
                valid_evidence = True
                break
        if not valid_evidence:
            issues.append("no source_evidence item cleanly points back to the original OWASP source requirement/summary")
    return not issues, issues


def verify_atlas_threat_item(
    threat: Dict[str, Any],
    expected_case_source_id: str,
    procedure_lookup: Dict[str, List[Dict[str, Any]]],
) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    source_knowledge_id = threat.get("source_knowledge_id") or ""
    if not source_knowledge_id.startswith(expected_case_source_id + "#"):
        issues.append(
            f"threat source_knowledge_id {source_knowledge_id} does not point back to atlas case {expected_case_source_id}"
        )
        return False, issues

    technique_id = source_knowledge_id.split("#", 1)[1]
    candidates = procedure_lookup.get(technique_id, [])
    if not candidates:
        issues.append(f"technique {technique_id} not found in original ATLAS procedure list")
        return False, issues

    evidence_items = threat.get("source_evidence", [])
    if not evidence_items:
        issues.append("missing source_evidence")
        return False, issues

    valid_evidence = False
    for evidence in evidence_items:
        if evidence.get("atlas_case_id") != expected_case_source_id:
            continue
        if evidence.get("source_knowledge_id") != source_knowledge_id:
            continue
        evidence_text = normalize_text(evidence.get("evidence_text"))
        evidence_proc_index = evidence.get("procedure_index")
        for candidate in candidates:
            if candidate["procedure_index"] != evidence_proc_index:
                continue
            if candidate["mechanism"] != evidence_text:
                continue
            valid_evidence = True
            break
        if valid_evidence:
            break

    if not valid_evidence:
        issues.append("no source_evidence item cleanly maps to the original ATLAS procedure/technique")
    return not issues, issues


def build_report() -> Dict[str, Any]:
    registry = read_json(CASE_REGISTRY_PATH)["cases"]
    threat_cases = read_json(THREAT_RECORDS_PATH)["threat_record_cases"]
    owasp_sources = read_json(OWASP_SOURCE_PATH)
    atlas_sources = read_json(ATLAS_SOURCE_PATH)

    registry_map = {row["use_case_id"]: row for row in registry}
    owasp_by_requirement = {normalize_text(item["requirement_text"]): item for item in owasp_sources}
    atlas_by_id = {item["id"]: item for item in atlas_sources}

    case_results: List[OrderedDict] = []
    fail_case_ids: List[str] = []
    missing_original_threat_case_ids: List[str] = []
    total_threat_items = 0
    traceable_threat_items = 0

    for case in threat_cases:
        use_case_id = case["use_case_id"]
        registry_row = registry_map[use_case_id]
        issues: List[str] = []
        threat_results: List[OrderedDict] = []
        retains_original_threat = False
        all_threat_items_traceable = True

        if case.get("dataset") != registry_row.get("dataset"):
            issues.append(f"dataset mismatch: threat_records={case.get('dataset')} registry={registry_row.get('dataset')}")
        if case.get("source_knowledge_id") != registry_row.get("source_knowledge_id"):
            issues.append(
                f"case source_knowledge_id mismatch: threat_records={case.get('source_knowledge_id')} registry={registry_row.get('source_knowledge_id')}"
            )

        if registry_row["dataset"] == "owasp":
            source = owasp_by_requirement[normalize_text(registry_row["source_requirement_text"])]
            expected_evidence_text = build_owasp_expected_evidence(source)
            for threat in case.get("threat_records", []):
                total_threat_items += 1
                item_ok, item_issues = verify_owasp_threat_item(threat, registry_row["source_knowledge_id"], expected_evidence_text)
                if item_ok:
                    retains_original_threat = True
                    traceable_threat_items += 1
                else:
                    all_threat_items_traceable = False
                threat_results.append(
                    OrderedDict(
                        [
                            ("threat_id", threat["threat_id"]),
                            ("source_knowledge_id", threat.get("source_knowledge_id")),
                            ("traceable_to_original_threat", item_ok),
                            ("issues", item_issues),
                        ]
                    )
                )
        else:
            source = atlas_by_id[registry_row["source_knowledge_id"]]
            procedure_lookup = build_atlas_procedure_lookup(source)
            for threat in case.get("threat_records", []):
                total_threat_items += 1
                item_ok, item_issues = verify_atlas_threat_item(threat, registry_row["source_knowledge_id"], procedure_lookup)
                if item_ok:
                    retains_original_threat = True
                    traceable_threat_items += 1
                else:
                    all_threat_items_traceable = False
                threat_results.append(
                    OrderedDict(
                        [
                            ("threat_id", threat["threat_id"]),
                            ("source_knowledge_id", threat.get("source_knowledge_id")),
                            ("traceable_to_original_threat", item_ok),
                            ("issues", item_issues),
                        ]
                    )
                )

        threat_count = len(case.get("threat_records", []))
        if threat_count == 0:
            missing_original_threat_case_ids.append(use_case_id)
            all_threat_items_traceable = False

        if retains_original_threat and all_threat_items_traceable and not issues:
            status = "pass"
            notes = ["At least one retained threat cleanly points back to the original source attack/threat."]
        else:
            status = "fail"
            notes = ["Case does not cleanly retain its original threat trace under the current verification rule."]
            fail_case_ids.append(use_case_id)

        case_results.append(
            OrderedDict(
                [
                    ("use_case_id", use_case_id),
                    ("dataset", registry_row["dataset"]),
                    ("source_knowledge_id", registry_row["source_knowledge_id"]),
                    ("threat_count", threat_count),
                    ("retains_original_threat", retains_original_threat),
                    ("all_threat_items_traceable", all_threat_items_traceable),
                    ("status", status),
                    ("notes", notes),
                    ("case_issues", issues),
                    ("threat_results", threat_results),
                ]
            )
        )

    case_results.sort(key=lambda item: item["use_case_id"])
    pass_case_count = sum(1 for item in case_results if item["status"] == "pass")
    fail_case_count = sum(1 for item in case_results if item["status"] == "fail")

    return OrderedDict(
        [
            (
                "meta",
                OrderedDict(
                    [
                        ("report_name", "SAAFG original threat traceability check"),
                        ("version", "v0.2"),
                        ("generated_at_utc", now_utc()),
                        (
                            "verification_rule",
                            [
                                "Case-level source_knowledge_id must match the case registry entry.",
                                "OWASP threat items must keep the same source_knowledge_id as the original OWASP source item, and their source_evidence must point back to that source summary/requirement.",
                                "ATLAS threat items must keep a source_knowledge_id of the form <atlas_case_id>#<technique_id>, where the technique_id exists in the original procedure list and source_evidence maps back to that original procedure/technique.",
                                "A case is counted as retaining its original threat if it contains at least one threat item that satisfies the relevant rule above.",
                            ],
                        ),
                    ]
                ),
            ),
            (
                "summary",
                OrderedDict(
                    [
                        ("case_count", len(case_results)),
                        ("pass_case_count", pass_case_count),
                        ("expected_exception_case_count", 0),
                        ("fail_case_count", fail_case_count),
                        ("retains_original_threat_case_count", sum(1 for item in case_results if item["retains_original_threat"])),
                        ("missing_original_threat_case_ids", missing_original_threat_case_ids),
                        ("threat_item_count", total_threat_items),
                        ("traceable_threat_item_count", traceable_threat_items),
                        ("untraceable_threat_item_count", total_threat_items - traceable_threat_items),
                        ("expected_exception_case_ids", []),
                        ("fail_case_ids", fail_case_ids),
                    ]
                ),
            ),
            ("case_results", case_results),
        ]
    )


def build_markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# SAAFG Original Threat Traceability Check v0.2",
        "",
        "## Conclusion",
        "",
        f"- Total cases checked: {summary['case_count']}",
        f"- Cases that retain at least one original/source-grounded threat: {summary['retains_original_threat_case_count']}",
        f"- Clean pass cases: {summary['pass_case_count']}",
        f"- Expected exception cases: {summary['expected_exception_case_count']}",
        f"- Fail cases: {summary['fail_case_count']}",
        f"- Threat items checked: {summary['threat_item_count']}",
        f"- Threat items traceable to original source threat/attack evidence: {summary['traceable_threat_item_count']}",
        f"- Threat items not traceable: {summary['untraceable_threat_item_count']}",
        "",
        "## Verification Rule",
        "",
    ]

    for rule in report["meta"]["verification_rule"]:
        lines.append(f"- {rule}")

    lines.extend(
        [
            "",
            "## Exceptions",
            "",
        ]
    )

    if summary["expected_exception_case_ids"]:
        for use_case_id in summary["expected_exception_case_ids"]:
            case_result = next(item for item in report["case_results"] if item["use_case_id"] == use_case_id)
            lines.append(
                f"- {use_case_id}: source `{case_result['source_knowledge_id']}`, threat_count={case_result['threat_count']}, status={case_result['status']}"
            )
            for note in case_result["notes"]:
                lines.append(f"  {note}")
    else:
        lines.append("- None.")

    if summary["fail_case_ids"]:
        lines.extend(
            [
                "",
                "## Fail Cases",
                "",
            ]
        )
        for use_case_id in summary["fail_case_ids"]:
            case_result = next(item for item in report["case_results"] if item["use_case_id"] == use_case_id)
            lines.append(
                f"- {use_case_id}: source `{case_result['source_knowledge_id']}`, threat_count={case_result['threat_count']}"
            )
            for issue in case_result["case_issues"]:
                lines.append(f"  case issue: {issue}")
            for threat in case_result["threat_results"]:
                if threat["issues"]:
                    lines.append(
                        f"  {threat['threat_id']} ({threat['source_knowledge_id']}): {'; '.join(threat['issues'])}"
                    )
    else:
        lines.extend(
            [
                "",
                "## Fail Cases",
                "",
                "- None.",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    write_json(OUTPUT_JSON_PATH, report)
    write_text(OUTPUT_MD_PATH, build_markdown(report))
    summary = report["summary"]
    print("Cases:", summary["case_count"])
    print("Retain original threat:", summary["retains_original_threat_case_count"])
    print("Expected exceptions:", summary["expected_exception_case_count"])
    print("Fails:", summary["fail_case_count"])
    print("Threat items:", summary["threat_item_count"])
    print("Traceable threat items:", summary["traceable_threat_item_count"])


if __name__ == "__main__":
    main()
