#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from safr_bench_templates import (
    GOLD_SUBSET_IDS,
    OWASP_DEV_SOURCE_IDS,
    OWASP_TEST_SOURCE_IDS,
    OWASP_THREAT_TEMPLATES,
)
from build_functional_use_case_flows import (
    DEFAULT_OUTPUT_PATH as CLEAN_FLOW_INPUT_PATH,
    DEFAULT_SOURCE_PATH as RAW_RESULT_PATH,
    build_output as build_clean_flow_output,
    load_results as load_raw_results,
)
from safr_common import (
    atlas_candidate_allowed,
    atlas_family_for_text,
    atlas_hint_profile,
    choose_retry_target,
    clean_source_text,
    first_sentence,
    normalize_text,
    score_step_for_keywords,
    tokenize,
    unique_tokens,
)


ROOT = Path(__file__).resolve().parents[2]
FLOW_INPUT_PATH = CLEAN_FLOW_INPUT_PATH
OWASP_SOURCE_PATH = (
    ROOT
    / "0_Data"
    / "5_Knowledge_Base"
    / "source"
    / "owasp_knowledge.json"
)
ATLAS_SOURCE_PATH = (
    ROOT
    / "0_Data"
    / "5_Knowledge_Base"
    / "source"
    / "mitre_atlas_knowledge.json"
)
BENCHMARK_ROOT = ROOT / "0_Data" / "6_SAAFG"
PACKAGE_ROOT = BENCHMARK_ROOT / "7_Benchmark_Package_v0_2"
HUMAN_CHECK_ROOT = BENCHMARK_ROOT / "5_Gold_or_Human_Check"
THREAT_OVERRIDE_PATH = HUMAN_CHECK_ROOT / "saafg_manual_threat_record_overrides_v0_2.json"
SECURITY_OVERRIDE_PATH = HUMAN_CHECK_ROOT / "saafg_security_flow_overrides_v0_2.json"


TACTIC_IMPACT_MAP = {
    "AI Model Access": "Exposed model interfaces or capabilities can be abused to compromise downstream behavior.",
    "Collection": "Sensitive user, model, or operational data can be gathered.",
    "Command and Control": "The AI workflow can be redirected to follow attacker-controlled instructions or infrastructure.",
    "Credential Access": "Sensitive credentials or secret material can be exposed or abused.",
    "Defense Evasion": "Malicious behavior can bypass or degrade AI-enabled detection and safeguards.",
    "Discovery": "Additional system details can be learned and used to widen compromise.",
    "Execution": "Malicious instructions or content can be executed by the AI system or connected tools.",
    "Exfiltration": "Sensitive data can be transmitted outside the intended trust boundary.",
    "Impact": "Service integrity, availability, or safety can be directly harmed.",
    "Initial Access": "Malicious foothold established through an exposed AI entry point.",
    "Persistence": "Long-lived malicious behavior can be retained in the AI workflow.",
    "Privilege Escalation": "Attackers can gain broader permissions or tool access than intended.",
}


OWASP_ANCHOR_PROFILES = {
    "LLM01_PromptInjection": {
        "preferred_verbs": ["receive", "ingest", "retrieve", "parse", "respond", "generate"],
        "bias": "early",
    },
    "LLM02_SensitiveInformationDisclosure": {
        "preferred_verbs": ["provide", "include", "send", "display", "share", "store"],
        "bias": "late",
    },
    "LLM03_SupplyChain": {
        "preferred_verbs": ["load", "integrate", "update", "install", "use", "execute"],
        "bias": "late",
    },
    "LLM04_DataModelPoisoning": {
        "preferred_verbs": ["ingest", "collect", "train", "update", "index", "retrieve"],
        "bias": "early",
    },
    "LLM05_ImproperOutputHandling": {
        "preferred_verbs": ["generate", "render", "execute", "forward", "call", "store"],
        "bias": "late",
    },
    "LLM06_ExcessiveAgency": {
        "preferred_verbs": ["invoke", "execute", "write", "delete", "modify", "access"],
        "bias": "late",
    },
    "LLM07_SystemPromptLeakage": {
        "preferred_verbs": ["share", "display", "provide", "send", "access", "read"],
        "bias": "late",
    },
    "LLM08_VectorAndEmbeddingWeaknesses": {
        "preferred_verbs": ["retrieve", "search", "index", "embed", "compose"],
        "bias": "early",
    },
    "LLM09_Misinformation": {
        "preferred_verbs": ["generate", "recommend", "classify", "publish", "display"],
        "bias": "late",
    },
    "LLM10_UnboundedConsumption": {
        "preferred_verbs": ["accept", "receive", "submit", "process", "compute", "analyze"],
        "bias": "early",
    },
}


FAMILY_PRIORITY = {
    "prompt": 8,
    "exec": 7,
    "secret": 6,
    "exfil": 5,
    "evasion": 4,
    "access": 3,
    "context": 2,
    "poison": 1,
    "destructive": 1,
    "generic": 0,
}

ATLAS_LOW_SIGNAL_TOKENS = {
    "agent",
    "application",
    "computer",
    "external",
    "local",
    "model",
    "platform",
    "researcher",
    "service",
    "system",
    "tool",
    "user",
    "victim",
    "workflow",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_if_exists(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    return read_json(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def normalize_title(source: Dict[str, Any]) -> str:
    return (
        source.get("source_title")
        or source.get("source_name")
        or source.get("title")
        or source.get("id")
        or ""
    )


def ensure_clean_input_exists() -> None:
    if FLOW_INPUT_PATH.exists():
        return
    results = load_raw_results(RAW_RESULT_PATH)
    payload = build_clean_flow_output(results, RAW_RESULT_PATH)
    FLOW_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FLOW_INPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_case_registry() -> Tuple[List[OrderedDict], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    ensure_clean_input_exists()
    flow_payload = read_json(FLOW_INPUT_PATH)
    owasp_cases = read_json(OWASP_SOURCE_PATH)
    atlas_cases = read_json(ATLAS_SOURCE_PATH)

    requirement_to_source: Dict[str, Dict[str, Any]] = {}
    for item in owasp_cases:
        requirement_to_source[normalize_text(item["requirement_text"])] = item
    for item in atlas_cases:
        requirement_to_source[normalize_text(item["requirement_text"])] = item

    registry: List[OrderedDict] = []
    for case in flow_payload["use_case_flows"]:
        req = normalize_text(case["source_requirement_text"])
        source = requirement_to_source[req]
        dataset = source["dataset"]
        source_id = source["id"]
        if dataset == "owasp":
            if source_id in OWASP_TEST_SOURCE_IDS:
                split = "test"
            elif source_id in OWASP_DEV_SOURCE_IDS:
                split = "dev"
            else:
                split = "train"
        else:
            atlas_order = int(case["use_case_id"][2:]) - 101
            if case["use_case_id"] in GOLD_SUBSET_IDS:
                split = "test"
            elif atlas_order % 7 == 0:
                split = "dev"
            elif atlas_order % 7 == 1:
                split = "test"
            else:
                split = "train"

        registry.append(
            OrderedDict(
                [
                    ("use_case_id", case["use_case_id"]),
                    ("dataset", dataset),
                    ("split", split),
                    ("source_knowledge_id", source_id),
                    ("source_title", normalize_title(source)),
                    ("source_result_index", case["source_result_index"]),
                    ("input_flow_version", case["input_flow_version"]),
                    ("basic_flow_step_count", len(case["basic_flow"])),
                    ("alternative_flow_count", len(case.get("alternative_flows", []))),
                    ("source_requirement_text", req),
                ]
            )
        )
    return registry, flow_payload["use_case_flows"], owasp_cases, atlas_cases


def build_splits(registry: Sequence[Dict[str, str]]) -> Dict[str, List[str]]:
    splits = {"train": [], "dev": [], "test": [], "gold_subset": list(GOLD_SUBSET_IDS)}
    for row in registry:
        splits[row["split"]].append(row["use_case_id"])
    return splits


def build_schema() -> OrderedDict:
    return OrderedDict(
        [
            ("benchmark_name", "SAAFG-Bench"),
            ("version", "v0.2"),
            ("anchor_semantics", "anchor_steps contains exactly one primary defense-actionable BF step."),
            (
                "layers",
                OrderedDict(
                    [
                        ("requirement_text", ["use_case_id", "dataset", "source_knowledge_id", "source_requirement_text"]),
                        ("functional_flow", ["use_case_id", "basic_flow", "alternative_flows"]),
                        ("threat_records", ["use_case_id", "threat_records"]),
                        ("security_augmented_flow", ["use_case_id", "security_augmented_flow"]),
                    ]
                ),
            ),
            ("basic_flow_step_fields", ["step_id", "source_step_id", "step_sentence", "subject", "verb", "object", "flow_from"]),
            (
                "threat_record_fields",
                [
                    "threat_id",
                    "threat_name",
                    "anchor_steps",
                    "threat_mechanism",
                    "security_impact",
                    "source_knowledge_id",
                    "source_evidence",
                ],
            ),
            ("security_basic_flow_fields", ["step_id", "anchor_after", "step_sentence"]),
            ("security_alternative_flow_fields", ["saf_id", "mitigates", "entry_condition", "source_evidence", "steps"]),
        ]
    )


def build_silver_functional_flows(registry: Sequence[Dict[str, str]], flows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    split_map = {item["use_case_id"]: item["split"] for item in registry}
    output = []
    for case in flows:
        item = OrderedDict(case)
        item["split"] = split_map[case["use_case_id"]]
        output.append(item)
    return {
        "meta": {
            "dataset_name": "SAAFG-Bench functional flows silver",
            "version": "v0.2",
            "generated_at_utc": now_utc(),
            "case_count": len(output),
            "notes": [
                "All flow steps are currently treated as BF steps.",
                "Alternative flows are intentionally left empty in this version.",
                "This file is derived from parser-produced system_flow outputs after conservative object and sentence cleanup.",
            ],
        },
        "use_case_flows": output,
    }


def build_owasp_source_evidence(source: Dict[str, Any]) -> List[OrderedDict]:
    summary = clean_source_text(first_sentence(source.get("source_summary") or source.get("requirement_text") or ""))
    return [
        OrderedDict(
            [
                ("evidence_type", "owasp_category"),
                ("source_knowledge_id", source["id"]),
                ("source_title", normalize_title(source)),
                ("evidence_text", summary),
            ]
        )
    ]


def owasp_position_bias(index: int, total: int, bias: str) -> int:
    if bias == "early":
        return total - index
    if bias == "late":
        return index + 1
    midpoint = total // 2
    return total - abs(index - midpoint)


def pick_primary_owasp_anchor_step(flow_steps: Sequence[Dict[str, str]], source_id: str) -> str:
    template = OWASP_THREAT_TEMPLATES[source_id]
    profile = OWASP_ANCHOR_PROFILES[source_id]
    keywords = list(template["anchor_groups"][0]) + list(template["anchor_groups"][1])
    best_step_id = flow_steps[0]["step_id"]
    best_key = (-1, -1)
    for index, step in enumerate(flow_steps):
        score = score_step_for_keywords(step, keywords, profile["preferred_verbs"])
        key = (score, owasp_position_bias(index, len(flow_steps), profile["bias"]))
        if key > best_key:
            best_key = key
            best_step_id = step["step_id"]
    return best_step_id


def build_owasp_threat_record(case: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    template = OWASP_THREAT_TEMPLATES[source["id"]]
    anchor = pick_primary_owasp_anchor_step(case["basic_flow"], source["id"])
    source_evidence = build_owasp_source_evidence(source)
    return {
        "use_case_id": case["use_case_id"],
        "dataset": "owasp",
        "source_knowledge_id": source["id"],
        "threat_records": [
            OrderedDict(
                [
                    ("threat_id", "T1"),
                    ("threat_name", template["name"]),
                    ("anchor_steps", [anchor]),
                    ("threat_mechanism", template["mechanism"]),
                    ("security_impact", template["impact"]),
                    ("source_knowledge_id", source["id"]),
                    ("source_evidence", source_evidence),
                ]
            )
        ],
    }


def atlas_position_bias(index: int, total: int, tactic_name: str, family: str) -> int:
    if tactic_name in {"Execution", "Exfiltration", "Impact"} or family in {"exec", "exfil", "destructive"}:
        return index + 1
    if tactic_name in {"Initial Access", "AI Model Access"} or family in {"prompt", "access"}:
        return total - index
    midpoint = total // 2
    return total - abs(index - midpoint)


def pick_primary_atlas_anchor(
    flow_steps: Sequence[Dict[str, str]],
    technique_name: str,
    technique_description: str,
    procedure_description: str,
    tactic_name: str,
    family: str,
) -> Tuple[Optional[str], int]:
    lexical_keywords = unique_tokens(
        [
            technique_name,
            first_sentence(technique_description),
            first_sentence(procedure_description),
        ],
        min_len=4,
    )
    lexical_keywords = [token for token in lexical_keywords if token not in ATLAS_LOW_SIGNAL_TOKENS]
    hint_profile = atlas_hint_profile(family)
    family_keywords = hint_profile["keywords"]
    preferred_verbs = hint_profile["preferred_verbs"]

    best_step_id: Optional[str] = None
    best_key = (-1, -1)
    for index, step in enumerate(flow_steps):
        lexical_score = score_step_for_keywords(step, lexical_keywords, [])
        family_score = score_step_for_keywords(step, family_keywords, preferred_verbs)
        total_score = lexical_score * 2 + family_score
        step_verb = (step.get("verb") or "").lower()
        step_text = " ".join(
            [
                step.get("step_sentence") or "",
                step.get("object") or "",
                step.get("subject") or "",
            ]
        ).lower()

        if step_verb in {"enable", "support", "facilitate", "ensure", "maintain"}:
            total_score -= 2
        if family == "prompt" and any(keyword in step_text for keyword in ("instruction", "prompt", "text", "message", "website", "content")):
            total_score += 3
        if family == "exec" and any(keyword in step_text for keyword in ("execute", "command", "script", "terminal", "clipboard", "invoke", "tool", "bash")):
            total_score += 3
        if family == "secret" and any(keyword in step_text for keyword in ("config", "credential", "token", "secret", "key", "file", "mcp", "ssh")):
            total_score += 3
        if family == "exfil" and any(keyword in step_text for keyword in ("transmit", "send", "share", "upload", "deliver", "distribute", "outbound", "context")):
            total_score += 3
        if family == "evasion" and any(keyword in step_text for keyword in ("validate", "verify", "classif", "authent", "approval", "match", "evaluate", "compatib")):
            total_score += 3
        if family == "poison" and any(keyword in step_text for keyword in ("upload", "submit", "repository", "publish", "registry", "artifact", "knowledge", "tuning", "modify")):
            total_score += 3
        key = (
            total_score,
            atlas_position_bias(index, len(flow_steps), tactic_name, family),
        )
        if key > best_key:
            best_key = key
            best_step_id = step["step_id"]

    if best_step_id is None:
        return (None, 0)
    return (best_step_id, best_key[0])


def build_atlas_candidate(case: Dict[str, Any], source: Dict[str, Any], proc_index: int, proc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    technique = proc.get("technique", {}) or {}
    tactic = proc.get("tactic", {}) or {}
    technique_id = technique.get("id")
    technique_name = normalize_text(technique.get("name") or technique_id or "")
    technique_description = clean_source_text(technique.get("description") or "")
    tactic_name = normalize_text(tactic.get("name") or "Unknown")
    tactic_id = normalize_text(tactic.get("id") or "")
    procedure_description = clean_source_text(proc.get("description") or "")
    if not technique_id or not technique_name:
        return None
    if not atlas_candidate_allowed(technique_id, technique_name, tactic_name):
        return None

    family = atlas_family_for_text(
        technique_name,
        procedure_description,
        tactic_name=tactic_name,
    )
    if family == "generic":
        family = atlas_family_for_text(
            technique_name,
            technique_description,
            procedure_description,
            tactic_name=tactic_name,
        )
    anchor_step_id, score = pick_primary_atlas_anchor(
        case["basic_flow"],
        technique_name,
        technique_description,
        procedure_description,
        tactic_name,
        family,
    )
    if not anchor_step_id or score <= 0:
        return None

    mechanism = first_sentence(procedure_description) or first_sentence(technique_description)
    impact = TACTIC_IMPACT_MAP.get(tactic_name) or str(atlas_hint_profile(family)["impact"])
    return {
        "technique_id": technique_id,
        "technique_name": technique_name,
        "technique_description": technique_description,
        "tactic_id": tactic_id,
        "tactic_name": tactic_name,
        "procedure_description": procedure_description,
        "anchor_step_id": anchor_step_id,
        "family": family,
        "score": score,
        "mechanism": mechanism,
        "impact": impact,
        "source_knowledge_id": f"{source['id']}#{technique_id}",
        "source_evidence": [
            OrderedDict(
                [
                    ("evidence_type", "atlas_technique"),
                    ("source_knowledge_id", f"{source['id']}#{technique_id}"),
                    ("atlas_case_id", source["id"]),
                    ("tactic_id", tactic_id),
                    ("tactic_name", tactic_name),
                    ("technique_id", technique_id),
                    ("technique_name", technique_name),
                    ("procedure_index", proc_index),
                    ("evidence_text", mechanism or procedure_description or technique_description),
                ]
            )
        ],
    }


def reduce_atlas_candidates(flow_steps: Sequence[Dict[str, str]], candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_technique: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        existing = best_by_technique.get(candidate["technique_id"])
        if not existing or (candidate["score"], FAMILY_PRIORITY[candidate["family"]]) > (
            existing["score"],
            FAMILY_PRIORITY[existing["family"]],
        ):
            best_by_technique[candidate["technique_id"]] = candidate

    best_by_anchor_family: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for candidate in best_by_technique.values():
        key = (candidate["anchor_step_id"], candidate["family"])
        existing = best_by_anchor_family.get(key)
        if not existing or (candidate["score"], len(tokenize(candidate["mechanism"]))) > (
            existing["score"],
            len(tokenize(existing["mechanism"])),
        ):
            best_by_anchor_family[key] = candidate

    ordered = sorted(
        best_by_anchor_family.values(),
        key=lambda item: (
            item["score"],
            FAMILY_PRIORITY[item["family"]],
            -int(item["anchor_step_id"][2:]),
        ),
        reverse=True,
    )

    limit = min(4, max(1, len(flow_steps)))
    per_anchor_counts: Dict[str, int] = defaultdict(int)
    selected: List[Dict[str, Any]] = []
    for candidate in ordered:
        anchor = candidate["anchor_step_id"]
        anchor_count = per_anchor_counts[anchor]
        if anchor_count >= 2:
            continue
        if anchor_count >= 1 and candidate["family"] not in {"exec", "secret", "exfil", "context"}:
            continue
        if anchor_count >= 1 and candidate["score"] < 6:
            continue
        selected.append(candidate)
        per_anchor_counts[anchor] += 1
        if len(selected) >= limit:
            break

    if not selected and ordered:
        top = ordered[0]
        if top["score"] >= 4:
            selected = [top]

    selected.sort(key=lambda item: int(item["anchor_step_id"][2:]))
    return selected


def build_atlas_threat_record(case: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    raw_candidates = []
    for proc_index, proc in enumerate(source.get("procedure", []), start=1):
        candidate = build_atlas_candidate(case, source, proc_index, proc)
        if candidate:
            raw_candidates.append(candidate)

    reduced_candidates = reduce_atlas_candidates(case["basic_flow"], raw_candidates)
    threat_records = []
    for counter, candidate in enumerate(reduced_candidates, start=1):
        threat_records.append(
            OrderedDict(
                [
                    ("threat_id", f"T{counter}"),
                    ("threat_name", f"{candidate['technique_name']} ({candidate['technique_id']})"),
                    ("anchor_steps", [candidate["anchor_step_id"]]),
                    ("threat_mechanism", candidate["mechanism"]),
                    ("security_impact", candidate["impact"]),
                    ("source_knowledge_id", candidate["source_knowledge_id"]),
                    ("source_evidence", candidate["source_evidence"]),
                ]
            )
        )

    return {
        "use_case_id": case["use_case_id"],
        "dataset": "atlas",
        "source_knowledge_id": source["id"],
        "threat_records": threat_records,
    }


def build_silver_threat_records(
    registry: Sequence[Dict[str, str]],
    flows: Sequence[Dict[str, Any]],
    owasp_cases: Sequence[Dict[str, Any]],
    atlas_cases: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    flow_map = {case["use_case_id"]: case for case in flows}
    owasp_map = {normalize_text(item["requirement_text"]): item for item in owasp_cases}
    atlas_map = {normalize_text(item["requirement_text"]): item for item in atlas_cases}
    records = []
    empty_cases = []
    for row in registry:
        case = flow_map[row["use_case_id"]]
        req = row["source_requirement_text"]
        if row["dataset"] == "owasp":
            source = owasp_map[req]
            record_case = build_owasp_threat_record(case, source)
        else:
            source = atlas_map[req]
            record_case = build_atlas_threat_record(case, source)
        if not record_case["threat_records"]:
            empty_cases.append(row["use_case_id"])
        records.append(record_case)
    return {
        "meta": {
            "dataset_name": "SAAFG-Bench threat records silver",
            "version": "v0.2",
            "generated_at_utc": now_utc(),
            "case_count": len(records),
            "empty_threat_case_count": len(empty_cases),
            "empty_threat_case_ids": empty_cases,
            "notes": [
                "Threat records in this file are source-grounded silver annotations.",
                "ATLAS core threats in v0.2 exclude attacker-side-only preparation, staging, and reconnaissance techniques unless they map to a victim-side actionable step.",
                "anchor_steps now contains a single primary defense-actionable BF step.",
            ],
        },
        "threat_record_cases": records,
    }


def apply_reviewed_threat_record_overrides(silver_threats: Dict[str, Any]) -> Dict[str, Any]:
    override_pack = read_json_if_exists(THREAT_OVERRIDE_PATH) or {}
    anchor_override_map = {
        (item["use_case_id"], item["threat_id"]): item["revised_anchor"]
        for item in override_pack.get("anchor_overrides", [])
    }
    manual_case_override_map = {
        item["use_case_id"]: item["threat_records"]
        for item in override_pack.get("manual_threat_case_overrides", [])
    }
    if not anchor_override_map and not manual_case_override_map:
        return silver_threats

    empty_cases = []
    for case in silver_threats["threat_record_cases"]:
        case_id = case["use_case_id"]
        if case_id in manual_case_override_map:
            case["threat_records"] = manual_case_override_map[case_id]
        for threat in case.get("threat_records", []):
            revised_anchor = anchor_override_map.get((case_id, threat["threat_id"]))
            if revised_anchor:
                threat["anchor_steps"] = [revised_anchor]
        if not case.get("threat_records"):
            empty_cases.append(case_id)

    silver_threats["meta"]["generated_at_utc"] = now_utc()
    silver_threats["meta"]["empty_threat_case_count"] = len(empty_cases)
    silver_threats["meta"]["empty_threat_case_ids"] = empty_cases
    notes = list(silver_threats["meta"].get("notes", []))
    if "Reviewed threat-record overrides were applied from 5_Gold_or_Human_Check when available." not in notes:
        notes.append("Reviewed threat-record overrides were applied from 5_Gold_or_Human_Check when available.")
    silver_threats["meta"]["notes"] = notes
    return silver_threats


def step(step_id: str, sentence: str) -> OrderedDict:
    return OrderedDict([("step_id", step_id), ("step_sentence", sentence)])


def sbf_step(step_id: str, anchor_after: str, sentence: str) -> OrderedDict:
    return OrderedDict(
        [
            ("step_id", step_id),
            ("anchor_after", anchor_after),
            ("step_sentence", sentence),
        ]
    )


def infer_owasp_silver_defense(source_knowledge_id: str) -> Tuple[str, str, str]:
    template_map = {
        "LLM01_PromptInjection": (
            "The system validates incoming instructions before the protected step proceeds.",
            "The system flags unsafe instruction content.",
            "The user revises the instruction and removes the injected content.",
        ),
        "LLM02_SensitiveInformationDisclosure": (
            "The system classifies sensitive data required by the protected step.",
            "The system flags sensitive information exposure risk.",
            "The system redacts or minimizes the sensitive data before retry.",
        ),
        "LLM03_SupplyChain": (
            "The system verifies the integrity and provenance of the external component.",
            "The system flags an untrusted external dependency.",
            "The system isolates the dependency and requires a trusted replacement.",
        ),
        "LLM04_DataModelPoisoning": (
            "The system validates the integrity of data entering the protected step.",
            "The system flags suspicious data contamination.",
            "The system quarantines the suspicious data and reloads trusted input.",
        ),
        "LLM05_ImproperOutputHandling": (
            "The system validates the safety of generated output before downstream use.",
            "The system flags unsafe output handling.",
            "The system sanitizes the output and requires a safe retry.",
        ),
        "LLM06_ExcessiveAgency": (
            "The system checks whether the protected action exceeds the approved privilege scope.",
            "The system flags an over-privileged agent action.",
            "The system narrows the requested action scope or requires approval.",
        ),
        "LLM07_SystemPromptLeakage": (
            "The system inspects whether hidden configuration is exposed by the protected step.",
            "The system flags internal prompt or configuration leakage.",
            "The system removes the sensitive configuration detail before retry.",
        ),
        "LLM08_VectorAndEmbeddingWeaknesses": (
            "The system validates the trustworthiness of retrieved context before use.",
            "The system flags untrusted retrieved context.",
            "The system filters the retrieved context and reloads trusted evidence.",
        ),
        "LLM09_Misinformation": (
            "The system verifies that the generated content is grounded before release.",
            "The system flags unsupported generated content.",
            "The system regenerates the content using grounded evidence only.",
        ),
        "LLM10_UnboundedConsumption": (
            "The system enforces resource and scope limits before processing continues.",
            "The system flags excessive resource demand.",
            "The system reduces the request scope before retry.",
        ),
    }
    return template_map[source_knowledge_id]


def infer_atlas_silver_defense(threat: Dict[str, Any]) -> Tuple[str, str, str]:
    family = atlas_family_for_text(
        threat.get("threat_name", ""),
        threat.get("threat_mechanism", ""),
        threat.get("security_impact", ""),
    )
    if family == "prompt":
        return (
            "The system validates retrieved or user-provided instructions before the protected step proceeds.",
            "The system flags untrusted instruction content.",
            "The system removes the injected instruction and requests a safe retry.",
        )
    if family == "exec":
        return (
            "The system validates privileged code or tool execution requests at the protected step.",
            "The system flags an unsafe execution request.",
            "The system blocks the execution path until it is manually approved or rewritten safely.",
        )
    if family == "secret":
        return (
            "The system checks whether the protected step touches secrets or sensitive configuration.",
            "The system flags secret exposure risk.",
            "The system masks the sensitive material and narrows the data scope before retry.",
        )
    if family == "exfil":
        return (
            "The system inspects outbound data scope before the protected step releases content.",
            "The system flags unauthorized outbound data transfer.",
            "The system redacts or blocks the outbound content before retry.",
        )
    if family == "evasion":
        return (
            "The system adds an adversarial validation check at the protected decision point.",
            "The system flags a suspicious bypass pattern.",
            "The system routes the input for stricter validation before retry.",
        )
    if family == "access":
        return (
            "The system verifies access conditions and provenance before the protected step proceeds.",
            "The system flags an untrusted access path.",
            "The system requires a trusted access path or additional verification before retry.",
        )
    if family == "context":
        return (
            "The system validates retrieved context provenance before it is used downstream.",
            "The system flags poisoned or untrusted contextual evidence.",
            "The system reloads trusted context and discards the untrusted entry.",
        )
    if family == "poison":
        return (
            "The system validates the integrity of artifacts or data entering the protected step.",
            "The system flags compromised artifacts or poisoned data.",
            "The system isolates the compromised artifact and reloads a trusted version.",
        )
    if family == "destructive":
        return (
            "The system checks whether the protected step triggers a destructive operation.",
            "The system flags a destructive action request.",
            "The system blocks the destructive path and requires manual approval.",
        )
    return (
        "The system performs a high-risk action review before the protected step proceeds.",
        "The system flags unsafe agent behavior.",
        "The system requests manual review and safe retry.",
    )


def build_silver_security_augmented_flows(
    registry: Sequence[Dict[str, str]],
    flows: Sequence[Dict[str, Any]],
    threat_cases: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    flow_map = {case["use_case_id"]: case for case in flows}
    threat_map = {case["use_case_id"]: case for case in threat_cases}
    results = []

    for row in registry:
        use_case_id = row["use_case_id"]
        flow_case = flow_map[use_case_id]
        flow_steps = flow_case["basic_flow"]
        threat_case = threat_map[use_case_id]
        suffix_counter: Dict[int, int] = defaultdict(int)
        security_basic_flow = []
        security_alternative_flows = []

        for threat in threat_case["threat_records"]:
            anchor_step_id = (threat.get("anchor_steps") or [flow_steps[0]["step_id"]])[0]
            anchor_num = int(anchor_step_id[2:]) if anchor_step_id.startswith("BF") else 1
            suffix_counter[anchor_num] += 1
            suffix = "" if suffix_counter[anchor_num] == 1 else chr(ord("a") + suffix_counter[anchor_num] - 2)
            sbf_id = f"SBF{anchor_num}{suffix}"
            saf_id = f"SAF{anchor_num}{suffix}"
            closure_mode, closure_target = choose_retry_target(flow_steps, anchor_step_id)

            if row["dataset"] == "owasp":
                detect_sentence, alert_sentence, remediate_sentence = infer_owasp_silver_defense(row["source_knowledge_id"])
            else:
                detect_sentence, alert_sentence, remediate_sentence = infer_atlas_silver_defense(threat)

            if closure_mode == "return" and closure_target:
                closure_sentence = f"The system returns to {closure_target}."
            else:
                closure_sentence = "The system terminates the unsafe flow and requires manual follow-up."

            security_basic_flow.append(sbf_step(sbf_id, anchor_step_id, detect_sentence))
            security_alternative_flows.append(
                OrderedDict(
                    [
                        ("saf_id", saf_id),
                        ("mitigates", [threat["threat_id"]]),
                        ("entry_condition", f"{sbf_id} flags a threat condition."),
                        ("source_evidence", threat.get("source_evidence", [])),
                        (
                            "steps",
                            [
                                step(f"{saf_id}.1", alert_sentence),
                                step(f"{saf_id}.2", remediate_sentence),
                                step(f"{saf_id}.3", closure_sentence),
                            ],
                        ),
                    ]
                )
            )

        results.append(
            OrderedDict(
                [
                    ("use_case_id", use_case_id),
                    ("dataset", row["dataset"]),
                    ("split", row["split"]),
                    ("source_knowledge_id", row["source_knowledge_id"]),
                    (
                        "security_augmented_flow",
                        OrderedDict(
                            [
                                ("security_basic_flow", security_basic_flow),
                                ("security_alternative_flows", security_alternative_flows),
                            ]
                        ),
                    ),
                ]
            )
        )

    return {
        "meta": {
            "dataset_name": "SAAFG-Bench security-augmented flows silver",
            "version": "v0.2",
            "generated_at_utc": now_utc(),
            "case_count": len(results),
            "notes": [
                "Security-augmented flows in this file are silver annotations.",
                "Each SAF carries source_evidence copied from the underlying threat record.",
                "SAF closure returns to the primary protected step when retry is meaningful; terminal anchors fall back to an earlier retryable step or explicit termination.",
            ],
        },
        "security_augmented_flow_cases": results,
    }


def apply_reviewed_security_flow_overrides(silver_sa_flows: Dict[str, Any]) -> Dict[str, Any]:
    override_pack = read_json_if_exists(SECURITY_OVERRIDE_PATH) or {}
    branch_override_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in override_pack.get("branch_overrides", []):
        branch_override_map[(item["use_case_id"], item["threat_id"])] = item
    if not branch_override_map:
        return silver_sa_flows

    for case in silver_sa_flows["security_augmented_flow_cases"]:
        case_id = case["use_case_id"]
        sf = case["security_augmented_flow"]
        sbf_map = {sbf["step_id"]: sbf for sbf in sf.get("security_basic_flow", [])}
        for saf in sf.get("security_alternative_flows", []):
            mitigates = saf.get("mitigates") or []
            if not mitigates:
                continue
            threat_id = mitigates[0]
            override = branch_override_map.get((case_id, threat_id))
            if not override:
                continue
            sbf_id = saf["entry_condition"].split()[0]
            step_overrides = override.get("step_overrides", {})
            if sbf_id in sbf_map and sbf_id in step_overrides:
                sbf_map[sbf_id]["step_sentence"] = step_overrides[sbf_id]
            for step_item in saf.get("steps", [])[:2]:
                if step_item["step_id"] in step_overrides:
                    step_item["step_sentence"] = step_overrides[step_item["step_id"]]

    silver_sa_flows["meta"]["generated_at_utc"] = now_utc()
    notes = list(silver_sa_flows["meta"].get("notes", []))
    if "Reviewed security-flow overrides were applied from 5_Gold_or_Human_Check when available." not in notes:
        notes.append("Reviewed security-flow overrides were applied from 5_Gold_or_Human_Check when available.")
    silver_sa_flows["meta"]["notes"] = notes
    return silver_sa_flows


def build_annotation_guideline() -> str:
    return """# SAAFG Annotation Guideline v0.2

## Purpose

This guideline defines how to interpret and annotate the four benchmark layers:

1. requirement_text
2. functional_use_case_flow
3. threat_records
4. security_augmented_flow

## Functional Flow Rules

1. Each BF step should describe one act-object pair.
2. v0.2 may repair nominalized parser objects using step-sentence and requirement-text context, but should remain conservative.
3. Do not inject security controls into the functional flow layer unless they already exist in the source specification.
4. If the source flow is linear, do not fabricate business AF branches.

## Threat Record Rules

1. anchor_steps must contain exactly one BF step in v0.2.
2. That step is the primary defense-actionable point: the place where the system can most plausibly detect, block, or constrain the threat.
3. For ATLAS, attacker-side-only preparation, staging, and reconnaissance techniques do not enter the core silver set unless they map to a victim-side actionable BF step.
4. Threat Mechanism should explain how the threat acts on the flow.
5. Security Impact should explain why the threat matters.
6. Source Knowledge ID must point to OWASP or ATLAS evidence, and source_evidence should preserve the supporting trace, preferably at ATLAS technique level.

## Security-Augmented Flow Rules

1. A defense must be expressed as SBF/SAF artifacts, not only as a flat mitigation sentence.
2. Each SAF must reference at least one Threat ID using mitigates.
3. Each SAF must include an entry_condition.
4. Each SAF must include source_evidence.
5. Each SAF must contain a retry target or an explicit termination action.
6. The original BF order must remain valid after insertion.

## Review Priorities

When reviewing a case, check in this order:

1. threat validity
2. anchor step correctness
3. threat-defense traceability
4. branch closure
5. flow consistency

## Labeling Caveat

Unless a subset has been confirmed by a human reviewer, it must be labeled as AI-reviewed or author-verified seed data, not expert-annotated gold.
"""


def build_evaluation_protocol() -> str:
    return """# SAAFG Evaluation Protocol v0.2

## Benchmark Tasks

### Task A: Threat Anchoring

Input:
- functional_use_case_flow

Output:
- threat_records

Metrics:
- threat validity
- primary anchor correctness
- source grounding completeness

### Task B: Defense Branch Generation

Input:
- functional_use_case_flow
- threat_records

Output:
- security_augmented_flow

Metrics:
- defense insertion correctness
- threat coverage
- branch closure
- flow consistency
- threat-defense traceability

### Task C: End-to-End SAAFG

Input:
- functional_use_case_flow

Output:
- security_augmented_flow

Metrics:
- end-to-end threat validity
- end-to-end defense coverage
- branch closure
- artifact usability

## Dataset Notes

- Silver set:
  large-scale, source-grounded, heuristic or model-assisted
- The current v0.2 freeze contains no empty core-threat cases after reviewed author overrides.
- The legacy AI-reviewed gold subset remains in the repository but was not automatically re-reviewed under the v0.2 anchor semantics.
"""


def build_critic_report_schema() -> OrderedDict:
    return OrderedDict(
        [
            ("dataset_name", "SAAFG Critic Report Schema"),
            ("version", "v0.2"),
            (
                "fields",
                [
                    "use_case_id",
                    "overall_decision",
                    "threat_validity",
                    "threat_coverage",
                    "traceability",
                    "branch_closure",
                    "flow_consistency",
                    "notes",
                ],
            ),
            ("decision_values", ["accept", "revise", "reject"]),
        ]
    )


def build_readme(metadata: Dict[str, Any], splits: Dict[str, List[str]]) -> str:
    return """# SAAFG-Bench v0.2

## Summary

This package provides the current v0.2 silver freeze for Security-Augmented Alternative Flow Generation (SAAFG).

It contains:

- a case registry
- split files
- cleaned functional flow inputs
- source-grounded silver threat records
- source-grounded silver security-augmented flows
- schema and protocol documents
- author-verified human-check artifacts under `0_Data/6_SAAFG/5_Gold_or_Human_Check`

## Counts

- total cases: {total}
- train cases: {train}
- dev cases: {dev}
- test cases: {test}
- empty-core-threat cases: {empty}
- author-verified gold subset cases: {author_verified}
- legacy AI-reviewed gold subset seed cases: {legacy_gold}

## Human-Check Files

- `author_verified_subset.json`: current round revised-case subset with reviewed flows, threat records, and security-augmented flows.
- `author_verified_subset_notes.md`: scope and labeling notes for the author-verified subset.
- `optional_anchor_adjudication.json`: ambiguity sidecar for acceptable alternate anchors that do not change the canonical benchmark anchor.
- `saafg_ai_reviewed_gold_subset_seed_v0_1.json`: legacy AI-reviewed seed subset retained for historical comparison.

## Important Note

The v0.2 author-verified subset is suitable for internal evaluation, ablation, and focused human-check workflows.
It should be described as author-verified rather than independent third-party expert gold unless additional external human confirmation is added later.
""".format(
        total=metadata["case_count"],
        train=len(splits["train"]),
        dev=len(splits["dev"]),
        test=len(splits["test"]),
        empty=metadata["empty_threat_case_count"],
        author_verified=metadata.get("human_check_breakdown", {}).get("author_verified_gold_subset_v0_2", 0),
        legacy_gold=metadata.get("human_check_breakdown", {}).get("legacy_ai_reviewed_gold_subset_v0_1", len(splits["gold_subset"])),
    )


def build_public_release_manifest() -> OrderedDict:
    return OrderedDict(
        [
            ("benchmark_name", "SAAFG-Bench"),
            ("version", "v0.2"),
            (
                "public_files",
                [
                    "0_Data/6_SAAFG/7_Benchmark_Package_v0_2/safr_bench_metadata.json",
                    "0_Data/6_SAAFG/7_Benchmark_Package_v0_2/safr_bench_schema.json",
                    "0_Data/6_SAAFG/7_Benchmark_Package_v0_2/case_registry_test_1.json",
                    "0_Data/6_SAAFG/7_Benchmark_Package_v0_2/case_registry_test_1.csv",
                    "0_Data/6_SAAFG/7_Benchmark_Package_v0_2/splits_test_1.json",
                    "0_Data/6_SAAFG/7_Benchmark_Package_v0_2/annotation_guidelines.md",
                    "0_Data/6_SAAFG/7_Benchmark_Package_v0_2/evaluation_protocol.md",
                    "0_Data/6_SAAFG/7_Benchmark_Package_v0_2/critic_report_schema.json",
                    "0_Data/6_SAAFG/7_Benchmark_Package_v0_2/README.md",
                    "0_Data/6_SAAFG/1_Input_Functional_Flows/input_functional_use_case_flows.json",
                    "0_Data/6_SAAFG/1_Input_Functional_Flows/functional_use_case_flows.json",
                    "0_Data/6_SAAFG/2_RedTeam_Threat_Records/threat_records.json",
                    "0_Data/6_SAAFG/3_BlueTeam_SA_Flows/security_augmented_use_case_flows.json",
                    "0_Data/6_SAAFG/5_Gold_or_Human_Check/author_verified_subset.json",
                    "0_Data/6_SAAFG/5_Gold_or_Human_Check/author_verified_subset_notes.md",
                ],
            ),
            ("non_public_internal_reference_files", ["0_Data/3_Experiment_Result/*"]),
            (
                "release_note",
                "The v0.2 package includes an author-verified gold subset derived from directly revised cases in the current review round; the legacy AI-reviewed subset remains as historical seed data.",
            ),
        ]
    )


def write_registry_csv(path: Path, registry: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(registry[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in registry:
            writer.writerow(row)


def main() -> None:
    for required_path in (THREAT_OVERRIDE_PATH, SECURITY_OVERRIDE_PATH):
        if not required_path.exists():
            raise FileNotFoundError(
                "Missing required reviewed override sidecar for the v0.2 freeze: {}".format(required_path)
            )
    registry, flows, owasp_cases, atlas_cases = build_case_registry()
    splits = build_splits(registry)
    schema = build_schema()
    silver_flows = build_silver_functional_flows(registry, flows)
    silver_threats = build_silver_threat_records(registry, flows, owasp_cases, atlas_cases)
    silver_threats = apply_reviewed_threat_record_overrides(silver_threats)
    silver_sa_flows = build_silver_security_augmented_flows(
        registry,
        flows,
        silver_threats["threat_record_cases"],
    )
    silver_sa_flows = apply_reviewed_security_flow_overrides(silver_sa_flows)

    dataset_breakdown = OrderedDict(
        [
            ("owasp", sum(1 for row in registry if row["dataset"] == "owasp")),
            ("atlas", sum(1 for row in registry if row["dataset"] == "atlas")),
        ]
    )
    threat_case_map = {item["use_case_id"]: item for item in silver_threats["threat_record_cases"]}
    threat_count_breakdown = OrderedDict(
        [
            (
                "owasp",
                sum(len(threat_case_map[row["use_case_id"]]["threat_records"]) for row in registry if row["dataset"] == "owasp"),
            ),
            (
                "atlas",
                sum(len(threat_case_map[row["use_case_id"]]["threat_records"]) for row in registry if row["dataset"] == "atlas"),
            ),
        ]
    )
    metadata = OrderedDict(
        [
            ("benchmark_name", "SAAFG-Bench"),
            ("version", "v0.2"),
            ("generated_at_utc", now_utc()),
            ("case_count", len(registry)),
            ("dataset_breakdown", dataset_breakdown),
            (
                "split_breakdown",
                OrderedDict(
                    [
                        ("train", len(splits["train"])),
                        ("dev", len(splits["dev"])),
                        ("test", len(splits["test"])),
                        ("gold_subset_legacy_v0_1", len(splits["gold_subset"])),
                    ]
                ),
            ),
            ("threat_count_breakdown", threat_count_breakdown),
            ("empty_threat_case_count", silver_threats["meta"]["empty_threat_case_count"]),
            ("empty_threat_case_ids", silver_threats["meta"]["empty_threat_case_ids"]),
            (
                "notes",
                [
                    "This benchmark package is built on local OWASP/ATLAS-derived assets already present in the repository.",
                    "The functional flow layer is derived from parser-produced system_flow outputs after conservative cleanup.",
                    "ATLAS core silver excludes attacker-side-only preparation, staging, and reconnaissance techniques unless they map to a victim-side actionable step.",
                    "anchor_steps in v0.2 contains exactly one primary defense-actionable BF step.",
                    "Reviewed threat-record and security-flow overrides are applied automatically when the override sidecars are present.",
                    "The legacy AI-reviewed gold subset remains available for historical reference.",
                ],
            ),
        ]
    )
    metadata["human_check_breakdown"] = OrderedDict(
        [
            ("legacy_ai_reviewed_gold_subset_v0_1", len(splits["gold_subset"])),
            ("author_verified_gold_subset_v0_2", 0),
            ("optional_anchor_sidecar_v0_2_case_count", 0),
        ]
    )

    write_json(BENCHMARK_ROOT / "1_Input_Functional_Flows" / FLOW_INPUT_PATH.name, read_json(FLOW_INPUT_PATH))
    write_json(BENCHMARK_ROOT / "1_Input_Functional_Flows" / "functional_use_case_flows.json", silver_flows)
    write_json(BENCHMARK_ROOT / "2_RedTeam_Threat_Records" / "threat_records.json", silver_threats)
    write_json(BENCHMARK_ROOT / "3_BlueTeam_SA_Flows" / "security_augmented_use_case_flows.json", silver_sa_flows)

    write_json(PACKAGE_ROOT / "safr_bench_metadata.json", metadata)
    write_json(PACKAGE_ROOT / "safr_bench_schema.json", schema)
    write_json(PACKAGE_ROOT / "case_registry_test_1.json", {"cases": registry})
    write_registry_csv(PACKAGE_ROOT / "case_registry_test_1.csv", registry)
    write_json(PACKAGE_ROOT / "splits_test_1.json", splits)
    write_text(PACKAGE_ROOT / "annotation_guidelines.md", build_annotation_guideline())
    write_text(PACKAGE_ROOT / "evaluation_protocol.md", build_evaluation_protocol())
    write_json(PACKAGE_ROOT / "critic_report_schema.json", build_critic_report_schema())
    write_text(PACKAGE_ROOT / "README.md", build_readme(metadata, splits))
    write_json(PACKAGE_ROOT / "release_manifest.json", build_public_release_manifest())

    from build_review_overrides import main as build_review_override_main
    from build_author_verified_subset import main as build_author_verified_subset_main

    build_review_override_main()
    build_author_verified_subset_main()

    print("Built SAAFG-Bench v0.2")
    print("Cases:", len(registry))
    print("OWASP threats:", threat_count_breakdown["owasp"])
    print("ATLAS threats:", threat_count_breakdown["atlas"])
    print("Empty threat cases:", silver_threats["meta"]["empty_threat_case_count"])


if __name__ == "__main__":
    main()
