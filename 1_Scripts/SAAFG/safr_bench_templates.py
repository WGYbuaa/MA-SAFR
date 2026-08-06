#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
import re
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FLOW_INPUT_PATH = (
    ROOT
    / "0_Data"
    / "6_SAAFG"
    / "1_Input_Functional_Flows"
    / "saafg_input_functional_use_case_flows_v1_from_baseline_ma_norag_qwen35plus.json"
)
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


OWASP_THREAT_TEMPLATES = {
    "LLM01_PromptInjection": {
        "name": "Prompt Injection -> Behavior Manipulation",
        "mechanism": "Untrusted instructions can override intended model behavior and steer downstream reasoning or actions.",
        "impact": "Unsafe model behavior, policy bypass, or downstream data exposure.",
        "anchor_groups": [
            ["provide", "enter", "submit", "receive", "accept", "ingest", "retrieve", "parse", "input", "prompt"],
            ["analyze", "generate", "adapt", "modify", "respond", "execute", "plan"],
        ],
    },
    "LLM02_SensitiveInformationDisclosure": {
        "name": "Sensitive Information Disclosure",
        "mechanism": "Sensitive user or system information can be retained, combined, or surfaced during model-supported interactions.",
        "impact": "Exposure of personal data, confidential information, or protected context.",
        "anchor_groups": [
            ["profile", "attribute", "context", "data", "record", "history", "information"],
            ["generate", "deliver", "respond", "send", "display", "include", "output"],
        ],
    },
    "LLM03_SupplyChain": {
        "name": "Supply Chain Compromise",
        "mechanism": "External components, datasets, models, tools, or dependencies can inject unsafe behavior into the AI workflow.",
        "impact": "Compromised integrity of the AI pipeline, unsafe model behavior, or malicious downstream actions.",
        "anchor_groups": [
            ["external", "tool", "dependency", "package", "model", "dataset", "retrieve", "integrate", "load", "connect"],
            ["execute", "apply", "use", "deploy", "update", "configure"],
        ],
    },
    "LLM04_DataModelPoisoning": {
        "name": "Data / Model Poisoning",
        "mechanism": "Malicious or corrupted data can influence model state, updates, retrieval assets, or downstream outputs.",
        "impact": "Integrity degradation, persistent unsafe behavior, or poisoned outputs.",
        "anchor_groups": [
            ["ingest", "collect", "store", "retrieve", "index", "train", "learn", "update", "fine-tune"],
            ["analyze", "generate", "classify", "respond"],
        ],
    },
    "LLM05_ImproperOutputHandling": {
        "name": "Improper Output Handling",
        "mechanism": "LLM outputs can be executed, rendered, stored, or forwarded without deterministic validation.",
        "impact": "Unsafe execution, unintended side effects, or security control bypass.",
        "anchor_groups": [
            ["generate", "respond", "produce", "evaluate", "output", "summary"],
            ["execute", "apply", "send", "store", "render", "forward", "call", "configure"],
        ],
    },
    "LLM06_ExcessiveAgency": {
        "name": "Excessive Agency / Over-Privileged Actions",
        "mechanism": "An LLM-enabled agent can be granted permissions or tools beyond what is necessary for the intended task.",
        "impact": "Unauthorized state changes, unsafe tool usage, or destructive operations.",
        "anchor_groups": [
            ["agent", "control", "access", "execute", "perform", "invoke", "write", "delete", "modify", "tool"],
            ["repository", "system", "environment", "configuration", "action"],
        ],
    },
    "LLM07_SystemPromptLeakage": {
        "name": "System Prompt / Configuration Leakage",
        "mechanism": "Operational instructions, hidden prompts, configuration details, or credentials can be exposed to users or external tools.",
        "impact": "Leakage of internal behavior constraints, architecture details, or authentication parameters.",
        "anchor_groups": [
            ["instruction", "configuration", "architecture", "connection", "authentication", "credential", "visibility", "access"],
            ["provide", "expose", "share", "optimize", "maintain"],
        ],
    },
    "LLM08_VectorAndEmbeddingWeaknesses": {
        "name": "Vector / Embedding Retrieval Weakness",
        "mechanism": "Embedding or retrieval components can surface untrusted or adversarially prepared context into generation.",
        "impact": "Context poisoning, retrieval manipulation, or unsafe grounded outputs.",
        "anchor_groups": [
            ["retrieve", "index", "embed", "search", "knowledge", "context", "database", "rag"],
            ["generate", "respond", "compose", "include"],
        ],
    },
    "LLM09_Misinformation": {
        "name": "Misinformation and Fabrication",
        "mechanism": "The model can generate or amplify inaccurate content that appears plausible within the workflow context.",
        "impact": "Misleading outputs, incorrect decisions, or unsafe downstream actions based on fabricated content.",
        "anchor_groups": [
            ["generate", "summarize", "classify", "recommend", "respond", "answer"],
            ["deliver", "send", "publish", "display"],
        ],
    },
    "LLM10_UnboundedConsumption": {
        "name": "Unbounded Resource Consumption",
        "mechanism": "Large, repeated, or computationally expensive inputs can exhaust system resources or degrade service quality.",
        "impact": "Availability degradation, latency increase, or denial of service.",
        "anchor_groups": [
            ["ingest", "receive", "accept", "input", "submit", "content"],
            ["analyze", "generate", "distribute", "maintain", "process", "compute"],
        ],
    },
}


TACTIC_IMPACT_MAP = {
    "Reconnaissance": "Exposure of system information that enables later compromise.",
    "Resource Development": "Preparation of malicious resources that support later attacks.",
    "AI Attack Staging": "Preparation of inputs or models that undermine downstream AI behavior.",
    "Initial Access": "Malicious foothold established through an exposed AI entry point.",
    "Persistence": "Long-lived malicious behavior can be retained in the AI workflow.",
    "Execution": "Malicious instructions or content can be executed by the AI system or connected tools.",
    "Privilege Escalation": "Attackers can gain broader permissions or tool access than intended.",
    "Defense Evasion": "Malicious behavior can bypass or degrade AI-enabled detection and safeguards.",
    "Credential Access": "Sensitive credentials or secret material can be exposed or abused.",
    "Discovery": "Additional system details can be learned and used to widen compromise.",
    "Collection": "Sensitive user, model, or operational data can be gathered.",
    "Exfiltration": "Sensitive data can be transmitted outside the intended trust boundary.",
    "Impact": "Service integrity, availability, or safety can be directly harmed.",
}


OWASP_TEST_SOURCE_IDS = {
    "LLM01_PromptInjection",
    "LLM02_SensitiveInformationDisclosure",
    "LLM06_ExcessiveAgency",
    "LLM07_SystemPromptLeakage",
    "LLM10_UnboundedConsumption",
}
OWASP_DEV_SOURCE_IDS = {
    "LLM05_ImproperOutputHandling",
}

GOLD_SUBSET_IDS = [
    "UC0001",
    "UC0012",
    "UC0050",
    "UC0061",
    "UC0068",
    "UC0088",
    "UC0117",
    "UC0125",
    "UC0138",
    "UC0140",
    "UC0147",
    "UC0155",
]


def now_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def clean_text(text):
    return " ".join((text or "").split())


def first_sentence(text):
    text = clean_text(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    return parts[0].strip()


def normalize_title(source):
    return source.get("source_title") or source.get("source_name") or source.get("id") or ""


def tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def score_step(step, keywords):
    score = 0
    haystack = " ".join(
        [
            step.get("step_sentence") or "",
            step.get("subject") or "",
            step.get("verb") or "",
            step.get("object") or "",
        ]
    ).lower()
    for keyword in keywords:
        if keyword and keyword.lower() in haystack:
            score += 1
    return score


def pick_top_steps(flow_steps, keywords, top_n=2, reverse_position_bias=False):
    scored = []
    for idx, step in enumerate(flow_steps):
        score = score_step(step, keywords)
        position_bonus = idx if reverse_position_bias else (len(flow_steps) - idx)
        scored.append((score, position_bonus, step["step_id"]))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    picked = [item[2] for item in scored if item[0] > 0][:top_n]
    if picked:
        return picked
    if reverse_position_bias:
        return [flow_steps[-1]["step_id"]]
    return [flow_steps[0]["step_id"]]


def pick_owasp_anchor_steps(flow_steps, source_id):
    template = OWASP_THREAT_TEMPLATES[source_id]
    groups = template["anchor_groups"]
    anchors = []
    if groups:
        anchors.extend(pick_top_steps(flow_steps, groups[0], top_n=1, reverse_position_bias=False))
    if len(groups) > 1:
        anchors.extend(pick_top_steps(flow_steps, groups[1], top_n=1, reverse_position_bias=True))
    seen = []
    for anchor in anchors:
        if anchor not in seen:
            seen.append(anchor)
    return seen[:2]


def atlas_position_default(flow_steps, tactic_name):
    mid_index = len(flow_steps) // 2
    if tactic_name in ("Reconnaissance", "Resource Development", "Initial Access", "Persistence"):
        return [flow_steps[0]["step_id"]]
    if tactic_name in ("Execution", "Privilege Escalation", "Collection", "Exfiltration", "Impact"):
        return [flow_steps[-1]["step_id"]]
    if tactic_name in ("AI Attack Staging", "Defense Evasion", "Discovery"):
        return [flow_steps[mid_index]["step_id"]]
    return [flow_steps[mid_index]["step_id"]]


def pick_atlas_anchor_steps(flow_steps, technique_name, technique_desc, tactic_name):
    name_keywords = [token for token in tokenize(technique_name) if len(token) >= 4]
    desc_keywords = [token for token in tokenize(first_sentence(technique_desc)) if len(token) >= 5]
    keywords = name_keywords[:6] + desc_keywords[:8]
    if keywords:
        picked = pick_top_steps(
            flow_steps,
            keywords,
            top_n=2,
            reverse_position_bias=tactic_name in ("Execution", "Privilege Escalation", "Collection", "Exfiltration", "Impact"),
        )
        if picked:
            return picked
    return atlas_position_default(flow_steps, tactic_name)


def build_case_registry():
    flow_payload = read_json(FLOW_INPUT_PATH)
    owasp_cases = read_json(OWASP_SOURCE_PATH)
    atlas_cases = read_json(ATLAS_SOURCE_PATH)

    requirement_to_source = {}
    for item in owasp_cases:
        requirement_to_source[clean_text(item["requirement_text"])] = item
    for item in atlas_cases:
        requirement_to_source[clean_text(item["requirement_text"])] = item

    registry = []
    for case in flow_payload["use_case_flows"]:
        req = clean_text(case["source_requirement_text"])
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


def build_splits(registry):
    splits = {"train": [], "dev": [], "test": [], "gold_subset": list(GOLD_SUBSET_IDS)}
    for row in registry:
        splits[row["split"]].append(row["use_case_id"])
    return splits


def build_schema():
    return OrderedDict(
        [
            ("benchmark_name", "SAAFG-Bench"),
            ("version", "v0.1"),
            ("layers", OrderedDict(
                [
                    ("requirement_text", ["use_case_id", "dataset", "source_knowledge_id", "source_requirement_text"]),
                    ("functional_flow", ["use_case_id", "basic_flow", "alternative_flows"]),
                    ("threat_records", ["use_case_id", "threat_records"]),
                    ("security_augmented_flow", ["use_case_id", "security_augmented_flow"]),
                ]
            )),
            ("basic_flow_step_fields", ["step_id", "source_step_id", "step_sentence", "subject", "verb", "object", "flow_from"]),
            ("threat_record_fields", ["threat_id", "threat_name", "anchor_steps", "threat_mechanism", "security_impact", "source_knowledge_id"]),
            ("security_basic_flow_fields", ["step_id", "anchor_after", "step_sentence"]),
            ("security_alternative_flow_fields", ["saf_id", "mitigates", "entry_condition", "steps"]),
        ]
    )


def build_silver_functional_flows(registry, flows):
    split_map = {item["use_case_id"]: item["split"] for item in registry}
    output = []
    for case in flows:
        item = OrderedDict(case)
        item["split"] = split_map[case["use_case_id"]]
        output.append(item)
    return {
        "meta": {
            "dataset_name": "SAAFG-Bench functional flows silver",
            "version": "v0.1",
            "generated_at_utc": now_utc(),
            "case_count": len(output),
            "notes": [
                "All flow steps are currently treated as BF steps.",
                "Alternative flows are intentionally left empty in this version.",
                "This file is derived from the parser-produced system_flow outputs of the legacy MA-NoRAG baseline.",
            ],
        },
        "use_case_flows": output,
    }


def build_owasp_threat_record(case, source):
    template = OWASP_THREAT_TEMPLATES[source["id"]]
    anchors = pick_owasp_anchor_steps(case["basic_flow"], source["id"])
    return {
        "use_case_id": case["use_case_id"],
        "dataset": "owasp",
        "source_knowledge_id": source["id"],
        "threat_records": [
            OrderedDict(
                [
                    ("threat_id", "T1"),
                    ("threat_name", template["name"]),
                    ("anchor_steps", anchors),
                    ("threat_mechanism", template["mechanism"]),
                    ("security_impact", template["impact"]),
                    ("source_knowledge_id", source["id"]),
                ]
            )
        ],
    }


def build_atlas_threat_record(case, source):
    unique = OrderedDict()
    for proc in source.get("procedure", []):
        technique = proc.get("technique", {}) or {}
        tactic = proc.get("tactic", {}) or {}
        tid = technique.get("id")
        if not tid:
            continue
        if tid not in unique:
            unique[tid] = {
                "technique_name": technique.get("name") or tid,
                "technique_description": technique.get("description") or "",
                "tactic_name": tactic.get("name") or "Unknown",
                "procedure_description": proc.get("description") or "",
            }
    threat_records = []
    counter = 1
    for technique_id, info in unique.items():
        mechanism_source = first_sentence(info["procedure_description"]) or first_sentence(info["technique_description"])
        impact = TACTIC_IMPACT_MAP.get(info["tactic_name"], "The AI-enabled workflow can be compromised or misused.")
        anchors = pick_atlas_anchor_steps(
            case["basic_flow"],
            info["technique_name"],
            info["technique_description"],
            info["tactic_name"],
        )
        threat_records.append(
            OrderedDict(
                [
                    ("threat_id", "T{0}".format(counter)),
                    ("threat_name", "{0} ({1})".format(info["technique_name"], technique_id)),
                    ("anchor_steps", anchors),
                    ("threat_mechanism", mechanism_source),
                    ("security_impact", impact),
                    ("source_knowledge_id", "{0}#{1}".format(source["id"], technique_id)),
                ]
            )
        )
        counter += 1
    return {
        "use_case_id": case["use_case_id"],
        "dataset": "atlas",
        "source_knowledge_id": source["id"],
        "threat_records": threat_records,
    }


def build_silver_threat_records(registry, flows, owasp_cases, atlas_cases):
    flow_map = {case["use_case_id"]: case for case in flows}
    owasp_map = {clean_text(item["requirement_text"]): item for item in owasp_cases}
    atlas_map = {clean_text(item["requirement_text"]): item for item in atlas_cases}
    records = []
    for row in registry:
        case = flow_map[row["use_case_id"]]
        req = row["source_requirement_text"]
        if row["dataset"] == "owasp":
            source = owasp_map[req]
            records.append(build_owasp_threat_record(case, source))
        else:
            source = atlas_map[req]
            records.append(build_atlas_threat_record(case, source))
    return {
        "meta": {
            "dataset_name": "SAAFG-Bench threat records silver",
            "version": "v0.1",
            "generated_at_utc": now_utc(),
            "case_count": len(records),
            "notes": [
                "Threat records in this file are source-grounded silver annotations.",
                "They are derived from OWASP/ATLAS source knowledge plus deterministic anchor-step heuristics.",
                "They are not expert-labeled gold annotations.",
            ],
        },
        "threat_record_cases": records,
    }


def build_gold_subset_case_ids():
    return list(GOLD_SUBSET_IDS)


def get_flow(flow_map, use_case_id):
    return flow_map[use_case_id]["basic_flow"]


def step(step_id, sentence):
    return OrderedDict([("step_id", step_id), ("step_sentence", sentence)])


def step_number(step_id):
    match = re.search(r"(\d+)", step_id or "")
    return int(match.group(1)) if match else 0


def next_bf_step_id(flow_steps, anchor_step_id):
    for idx, flow_step in enumerate(flow_steps):
        if flow_step["step_id"] == anchor_step_id:
            if idx + 1 < len(flow_steps):
                return flow_steps[idx + 1]["step_id"]
            return anchor_step_id
    return flow_steps[0]["step_id"]


def sbf_step(step_id, anchor_after, sentence):
    return OrderedDict(
        [
            ("step_id", step_id),
            ("anchor_after", anchor_after),
            ("step_sentence", sentence),
        ]
    )


def infer_owasp_silver_defense(source_knowledge_id):
    template_map = {
        "LLM01_PromptInjection": (
            "The system detects high-risk instruction patterns.",
            "The system flags unsafe instruction content.",
            "The user revises the instruction.",
        ),
        "LLM02_SensitiveInformationDisclosure": (
            "The system classifies sensitive context data.",
            "The system flags sensitive information exposure.",
            "The system redacts sensitive context data.",
        ),
        "LLM03_SupplyChain": (
            "The system verifies external component integrity.",
            "The system flags an untrusted external component.",
            "The system isolates the external component.",
        ),
        "LLM04_DataModelPoisoning": (
            "The system validates ingested data integrity.",
            "The system flags suspicious data contamination.",
            "The system quarantines the suspicious data.",
        ),
        "LLM05_ImproperOutputHandling": (
            "The system validates generated output safety.",
            "The system flags unsafe output handling.",
            "The system sanitizes the generated output.",
        ),
        "LLM06_ExcessiveAgency": (
            "The system verifies requested tool privileges.",
            "The system flags an over-privileged agent action.",
            "The user narrows the requested action scope.",
        ),
        "LLM07_SystemPromptLeakage": (
            "The system inspects internal prompt exposure.",
            "The system flags internal configuration leakage.",
            "The system removes the sensitive configuration detail.",
        ),
        "LLM08_VectorAndEmbeddingWeaknesses": (
            "The system validates retrieved context trust.",
            "The system flags untrusted retrieved context.",
            "The system filters the retrieved context.",
        ),
        "LLM09_Misinformation": (
            "The system verifies generated claim support.",
            "The system flags unsupported generated content.",
            "The system regenerates the grounded content.",
        ),
        "LLM10_UnboundedConsumption": (
            "The system enforces request resource limits.",
            "The system flags excessive resource demand.",
            "The user reduces the request scope.",
        ),
    }
    return template_map.get(
        source_knowledge_id,
        (
            "The system performs a security validation check.",
            "The system flags a security policy violation.",
            "The system revises the unsafe operation.",
        ),
    )


def infer_atlas_silver_defense(threat_name, threat_mechanism, security_impact):
    haystack = " ".join([threat_name or "", threat_mechanism or "", security_impact or ""]).lower()
    if "prompt injection" in haystack or "injection" in haystack:
        return (
            "The system detects untrusted instruction content.",
            "The system flags unsafe injected context.",
            "The system sanitizes the injected context.",
        )
    if any(keyword in haystack for keyword in ["code execution", "shell", "command", "execute", "tool"]):
        return (
            "The system validates privileged execution requests.",
            "The system flags an unsafe execution request.",
            "The system requires manual execution approval.",
        )
    if any(keyword in haystack for keyword in ["credential", "secret", "token", "password"]):
        return (
            "The system detects secret-bearing data paths.",
            "The system flags secret exposure risk.",
            "The system masks the secret-bearing data.",
        )
    if any(keyword in haystack for keyword in ["exfiltration", "data", "collection", "packag", "outbound"]):
        return (
            "The system inspects outbound data scope.",
            "The system flags unauthorized data transfer.",
            "The system redacts the outbound data.",
        )
    if any(keyword in haystack for keyword in ["poison", "rag", "retriev", "context"]):
        return (
            "The system validates retrieved context provenance.",
            "The system flags untrusted contextual evidence.",
            "The system reloads trusted contextual evidence.",
        )
    if any(keyword in haystack for keyword in ["destruct", "delete", "impact"]):
        return (
            "The system detects destructive action requests.",
            "The system flags a destructive operation.",
            "The system requires destructive-action approval.",
        )
    return (
        "The system performs a high-risk action review.",
        "The system flags unsafe agent behavior.",
        "The system requests manual review.",
    )


def build_silver_security_augmented_flows(registry, flows, threat_cases):
    flow_map = {case["use_case_id"]: case for case in flows}
    threat_map = {case["use_case_id"]: case for case in threat_cases}
    results = []

    for row in registry:
        use_case_id = row["use_case_id"]
        flow_case = flow_map[use_case_id]
        flow_steps = flow_case["basic_flow"]
        threat_case = threat_map[use_case_id]
        suffix_counter = defaultdict(int)
        security_basic_flow = []
        security_alternative_flows = []

        for threat in threat_case["threat_records"]:
            anchor_step_id = (threat.get("anchor_steps") or [flow_steps[0]["step_id"]])[0]
            anchor_num = step_number(anchor_step_id) or 1
            suffix_counter[anchor_num] += 1
            suffix = "" if suffix_counter[anchor_num] == 1 else chr(ord("a") + suffix_counter[anchor_num] - 2)
            sbf_id = "SBF{0}{1}".format(anchor_num, suffix)
            saf_id = "SAF{0}{1}".format(anchor_num, suffix)
            return_target = next_bf_step_id(flow_steps, anchor_step_id)

            if row["dataset"] == "owasp":
                detect_sentence, alert_sentence, remediate_sentence = infer_owasp_silver_defense(row["source_knowledge_id"])
            else:
                detect_sentence, alert_sentence, remediate_sentence = infer_atlas_silver_defense(
                    threat["threat_name"],
                    threat["threat_mechanism"],
                    threat["security_impact"],
                )

            security_basic_flow.append(sbf_step(sbf_id, anchor_step_id, detect_sentence))
            security_alternative_flows.append(
                OrderedDict(
                    [
                        ("saf_id", saf_id),
                        ("mitigates", [threat["threat_id"]]),
                        ("entry_condition", "{0} flags a threat condition.".format(sbf_id)),
                        ("steps", [
                            step("{0}.1".format(saf_id), alert_sentence),
                            step("{0}.2".format(saf_id), remediate_sentence),
                            step("{0}.3".format(saf_id), "The system returns to {0}.".format(return_target)),
                        ]),
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
                    ("security_augmented_flow", OrderedDict(
                        [
                            ("security_basic_flow", security_basic_flow),
                            ("security_alternative_flows", security_alternative_flows),
                        ]
                    )),
                ]
            )
        )

    return {
        "meta": {
            "dataset_name": "SAAFG-Bench security-augmented flows silver",
            "version": "v0.1",
            "generated_at_utc": now_utc(),
            "case_count": len(results),
            "notes": [
                "Security-augmented flows in this file are silver annotations.",
                "They are derived from silver threat records through deterministic defense-branch templates.",
                "They are suitable for benchmark prototyping and baseline evaluation, not as expert-labeled gold artifacts.",
            ],
        },
        "security_augmented_flow_cases": results,
    }


def build_ai_reviewed_gold_subset(registry, flows):
    flow_map = {case["use_case_id"]: case for case in flows}
    registry_map = {row["use_case_id"]: row for row in registry}

    manual = OrderedDict()

    manual["UC0001"] = {
        "threat_records": [
            {
                "threat_id": "T1",
                "threat_name": "Prompt Injection -> Behavior Manipulation",
                "anchor_steps": ["BF1", "BF4"],
                "threat_mechanism": "Direct user instructions can override intended constraints and steer downstream behavior adaptation.",
                "security_impact": "Unsafe behavior changes, policy bypass, or unintended downstream actions.",
                "source_knowledge_id": "LLM01_PromptInjection",
            }
        ],
        "security_basic_flow": [
            step("SBF2", "The AI system detects high-risk instruction patterns."),
            step("SBF4", "The AI system validates behavior changes against safety policy."),
        ],
        "security_alternative_flows": [
            OrderedDict(
                [
                    ("saf_id", "SAF2"),
                    ("mitigates", ["T1"]),
                    ("entry_condition", "SBF2 flags a high-risk instruction pattern."),
                    ("steps", [
                        step("SAF2.1", "The AI system warns the user about unsafe instruction content."),
                        step("SAF2.2", "The user revises the instruction."),
                        step("SAF2.3", "The AI system returns to BF2."),
                    ]),
                ]
            ),
            OrderedDict(
                [
                    ("saf_id", "SAF4"),
                    ("mitigates", ["T1"]),
                    ("entry_condition", "SBF4 detects a behavior change that violates policy."),
                    ("steps", [
                        step("SAF4.1", "The AI system blocks the unsafe behavior update."),
                        step("SAF4.2", "The AI system requests manual review."),
                        step("SAF4.3", "The AI system returns to BF4."),
                    ]),
                ]
            ),
        ],
        "review_notes": "Reviewed as a prompt-injection case. The critical anchors are user instruction intake and behavior adaptation.",
    }

    manual["UC0012"] = {
        "threat_records": [
            {
                "threat_id": "T1",
                "threat_name": "Sensitive Attribute Disclosure",
                "anchor_steps": ["BF1", "BF3", "BF4"],
                "threat_mechanism": "Personal attributes can be overexposed when the system maintains context and injects profile information into downstream service outcomes.",
                "security_impact": "Exposure of personal data and privacy violations.",
                "source_knowledge_id": "LLM02_SensitiveInformationDisclosure",
            }
        ],
        "security_basic_flow": [
            step("SBF3", "The system classifies the sensitivity of profile information."),
            step("SBF4", "The system checks the minimum data needed for the service outcome."),
        ],
        "security_alternative_flows": [
            OrderedDict(
                [
                    ("saf_id", "SAF3"),
                    ("mitigates", ["T1"]),
                    ("entry_condition", "SBF3 identifies sensitive profile information."),
                    ("steps", [
                        step("SAF3.1", "The system masks unnecessary profile attributes."),
                        step("SAF3.2", "The system records a privacy handling event."),
                        step("SAF3.3", "The system returns to BF3."),
                    ]),
                ]
            ),
            OrderedDict(
                [
                    ("saf_id", "SAF4"),
                    ("mitigates", ["T1"]),
                    ("entry_condition", "SBF4 finds that the planned output contains excessive personal detail."),
                    ("steps", [
                        step("SAF4.1", "The system removes excess personal detail from the service outcome."),
                        step("SAF4.2", "The system returns to BF4."),
                    ]),
                ]
            ),
        ],
        "review_notes": "Reviewed as a privacy-preserving branch insertion case.",
    }

    manual["UC0050"] = {
        "threat_records": [
            {
                "threat_id": "T1",
                "threat_name": "Unsafe Execution of LLM-Generated Commands",
                "anchor_steps": ["BF2", "BF3"],
                "threat_mechanism": "The host environment may execute model-produced commands via eval or exec without deterministic control.",
                "security_impact": "Arbitrary code execution and unsafe system state changes.",
                "source_knowledge_id": "LLM05_ImproperOutputHandling",
            }
        ],
        "security_basic_flow": [
            step("SBF2", "The system validates commands against an allowlist."),
            step("SBF3", "The system executes approved commands inside a sandbox."),
        ],
        "security_alternative_flows": [
            OrderedDict(
                [
                    ("saf_id", "SAF2"),
                    ("mitigates", ["T1"]),
                    ("entry_condition", "SBF2 rejects the command or detects unsafe capability use."),
                    ("steps", [
                        step("SAF2.1", "The system blocks the command."),
                        step("SAF2.2", "The system requests human approval."),
                        step("SAF2.3", "The system returns to BF1."),
                    ]),
                ]
            )
        ],
        "review_notes": "Reviewed as an output-handling case centered on deterministic command validation.",
    }

    manual["UC0061"] = {
        "threat_records": [
            {
                "threat_id": "T1",
                "threat_name": "Over-Privileged Agent Repository Operations",
                "anchor_steps": ["BF2", "BF4", "BF5"],
                "threat_mechanism": "An LLM agent with broad repository control can perform destructive write or delete actions beyond the intended task scope.",
                "security_impact": "Unauthorized repository modification, destructive deletion, and loss of integrity.",
                "source_knowledge_id": "LLM06_ExcessiveAgency",
            }
        ],
        "security_basic_flow": [
            step("SBF2", "The system scopes repository permissions by operation type."),
            step("SBF5", "The system checks whether delete operations require approval."),
        ],
        "security_alternative_flows": [
            OrderedDict(
                [
                    ("saf_id", "SAF5"),
                    ("mitigates", ["T1"]),
                    ("entry_condition", "SBF5 determines that the delete operation exceeds the approved permission scope."),
                    ("steps", [
                        step("SAF5.1", "The system blocks the delete operation."),
                        step("SAF5.2", "The user submits an approval request."),
                        step("SAF5.3", "The system returns to BF2."),
                    ]),
                ]
            )
        ],
        "review_notes": "Reviewed as an excessive-agency case; delete privilege is treated as the highest-risk anchor.",
    }

    manual["UC0068"] = {
        "threat_records": [
            {
                "threat_id": "T1",
                "threat_name": "System Prompt and Configuration Leakage",
                "anchor_steps": ["BF1", "BF2"],
                "threat_mechanism": "Operational instructions, architecture details, and authentication parameters can be exposed through overly transparent debugging interfaces.",
                "security_impact": "Leakage of hidden prompts, credentials, and internal configuration context.",
                "source_knowledge_id": "LLM07_SystemPromptLeakage",
            }
        ],
        "security_basic_flow": [
            step("SBF2", "The platform classifies requested configuration items by sensitivity."),
        ],
        "security_alternative_flows": [
            OrderedDict(
                [
                    ("saf_id", "SAF2"),
                    ("mitigates", ["T1"]),
                    ("entry_condition", "SBF2 marks requested configuration items as sensitive."),
                    ("steps", [
                        step("SAF2.1", "The platform redacts sensitive configuration content."),
                        step("SAF2.2", "The platform records an access audit event."),
                        step("SAF2.3", "The platform returns to BF2."),
                    ]),
                ]
            )
        ],
        "review_notes": "Reviewed as a configuration leakage case; sensitive visibility is constrained before disclosure.",
    }

    manual["UC0088"] = {
        "threat_records": [
            {
                "threat_id": "T1",
                "threat_name": "Resource Exhaustion via Unbounded Ingestion",
                "anchor_steps": ["BF1", "BF2"],
                "threat_mechanism": "Large or repeated user submissions can consume disproportionate processing resources before stable throttling occurs.",
                "security_impact": "Latency spikes, degraded responsiveness, or denial of service.",
                "source_knowledge_id": "LLM10_UnboundedConsumption",
            }
        ],
        "security_basic_flow": [
            step("SBF1", "The system measures content size and request rate."),
        ],
        "security_alternative_flows": [
            OrderedDict(
                [
                    ("saf_id", "SAF1"),
                    ("mitigates", ["T1"]),
                    ("entry_condition", "SBF1 detects that content size or request rate exceeds the service budget."),
                    ("steps", [
                        step("SAF1.1", "The system throttles the request."),
                        step("SAF1.2", "The system queues the content for deferred processing."),
                        step("SAF1.3", "The system returns to BF1."),
                    ]),
                ]
            )
        ],
        "review_notes": "Reviewed as an availability-focused case anchored at the first ingestion step.",
    }

    manual["UC0117"] = {
        "threat_records": [
            {
                "threat_id": "T1",
                "threat_name": "Prompt-Triggered Code Execution with Credential Abuse",
                "anchor_steps": ["BF1", "BF4", "BF5", "BF6"],
                "threat_mechanism": "Natural language input can cause generated Python code to access environment variables or API credentials before execution.",
                "security_impact": "Unauthorized code execution, credential misuse, and host compromise.",
                "source_knowledge_id": "AML.CS0016#AML.T0051.001",
            }
        ],
        "security_basic_flow": [
            step("SBF2", "The platform classifies generated code by risk level."),
            step("SBF5", "The platform checks whether the generated code requests credential access."),
        ],
        "security_alternative_flows": [
            OrderedDict(
                [
                    ("saf_id", "SAF5"),
                    ("mitigates", ["T1"]),
                    ("entry_condition", "SBF5 detects credential or host-environment access in generated code."),
                    ("steps", [
                        step("SAF5.1", "The platform blocks credential access."),
                        step("SAF5.2", "The platform requests human approval for the code path."),
                        step("SAF5.3", "The platform returns to BF2."),
                    ]),
                ]
            )
        ],
        "review_notes": "Reviewed as a code-execution case where credential access is treated as the dominant risk anchor.",
    }

    manual["UC0125"] = {
        "threat_records": [
            {
                "threat_id": "T1",
                "threat_name": "Indirect Prompt Injection via Historical Correspondence",
                "anchor_steps": ["BF1", "BF3", "BF4"],
                "threat_mechanism": "Instructions embedded in prior communications can be ingested into the knowledge base and later propagated into new outputs.",
                "security_impact": "Knowledge-base contamination, unsafe response generation, and downstream data leakage.",
                "source_knowledge_id": "AML.CS0024#AML.T0051.001",
            }
        ],
        "security_basic_flow": [
            step("SBF1", "The assistant classifies ingested correspondence by trust level."),
            step("SBF3", "The assistant detects instruction-like content in retrieved history."),
        ],
        "security_alternative_flows": [
            OrderedDict(
                [
                    ("saf_id", "SAF3"),
                    ("mitigates", ["T1"]),
                    ("entry_condition", "SBF3 detects instruction-like content in historical correspondence."),
                    ("steps", [
                        step("SAF3.1", "The assistant quarantines the historical message."),
                        step("SAF3.2", "The assistant excludes the message from generation context."),
                        step("SAF3.3", "The assistant returns to BF3."),
                    ]),
                ]
            )
        ],
        "review_notes": "Reviewed as a RAG-style indirect prompt injection case.",
    }

    manual["UC0138"] = {
        "threat_records": [
            {
                "threat_id": "T1",
                "threat_name": "Data Exfiltration via Agent Context Packaging",
                "anchor_steps": ["BF4", "BF5", "BF6"],
                "threat_mechanism": "The agent can retrieve excessive customer context from connected stores and forward the full package via email.",
                "security_impact": "Exfiltration of customer data beyond the minimum necessary handoff scope.",
                "source_knowledge_id": "AML.CS0037#AML.T0086",
            }
        ],
        "security_basic_flow": [
            step("SBF4", "The system classifies retrieved customer context by sensitivity."),
            step("SBF5", "The system checks whether the case package exceeds the minimum disclosure policy."),
        ],
        "security_alternative_flows": [
            OrderedDict(
                [
                    ("saf_id", "SAF5"),
                    ("mitigates", ["T1"]),
                    ("entry_condition", "SBF5 finds that the case package contains excessive sensitive context."),
                    ("steps", [
                        step("SAF5.1", "The system removes unnecessary sensitive context from the package."),
                        step("SAF5.2", "The system records a disclosure review event."),
                        step("SAF5.3", "The system returns to BF5."),
                    ]),
                ]
            )
        ],
        "review_notes": "Reviewed as a data-minimization and exfiltration-control case.",
    }

    manual["UC0140"] = {
        "threat_records": [
            {
                "threat_id": "T1",
                "threat_name": "Indirect Prompt Injection Triggering Privileged Agent Actions",
                "anchor_steps": ["BF1", "BF4", "BF6"],
                "threat_mechanism": "A malicious support ticket can cause the internal AI agent to interpret untrusted text as an action plan and execute it with organizational privileges.",
                "security_impact": "Privilege misuse, unauthorized actions, and unsafe workflow execution.",
                "source_knowledge_id": "AML.CS0039#AML.T0053",
            }
        ],
        "security_basic_flow": [
            step("SBF2", "The platform marks external ticket content as untrusted."),
            step("SBF4", "The internal AI agent validates the requested action against policy."),
        ],
        "security_alternative_flows": [
            OrderedDict(
                [
                    ("saf_id", "SAF4"),
                    ("mitigates", ["T1"]),
                    ("entry_condition", "SBF4 determines that the requested action is derived from untrusted ticket content and requires elevated privileges."),
                    ("steps", [
                        step("SAF4.1", "The internal AI agent suspends autonomous action execution."),
                        step("SAF4.2", "The platform requests human approval."),
                        step("SAF4.3", "The platform returns to BF4."),
                    ]),
                ]
            )
        ],
        "review_notes": "Reviewed as an agentic workflow case with external-content-triggered action risk.",
    }

    manual["UC0147"] = {
        "threat_records": [
            {
                "threat_id": "T1",
                "threat_name": "Document-Borne Prompt Injection -> Destructive Shell Execution",
                "anchor_steps": ["BF1", "BF2", "BF4", "BF5"],
                "threat_mechanism": "Operational directives embedded in uploaded documents can be extracted and translated into destructive shell commands.",
                "security_impact": "Data destruction, unsafe shell execution, and operational disruption.",
                "source_knowledge_id": "AML.CS0046#AML.T0086",
            }
        ],
        "security_basic_flow": [
            step("SBF2", "The AI agent classifies extracted directives by execution risk."),
            step("SBF4", "The AI agent validates shell commands against a destructive-action policy."),
        ],
        "security_alternative_flows": [
            OrderedDict(
                [
                    ("saf_id", "SAF4"),
                    ("mitigates", ["T1"]),
                    ("entry_condition", "SBF4 identifies a destructive shell command pattern."),
                    ("steps", [
                        step("SAF4.1", "The AI agent blocks the shell command."),
                        step("SAF4.2", "The AI agent requests manual review."),
                        step("SAF4.3", "The AI agent returns to BF2."),
                    ]),
                ]
            )
        ],
        "review_notes": "Reviewed as a destructive command case caused by document-borne instructions.",
    }

    manual["UC0155"] = {
        "threat_records": [
            {
                "threat_id": "T1",
                "threat_name": "Poisoned MCP Tool -> Context Exfiltration",
                "anchor_steps": ["BF1", "BF2", "BF4"],
                "threat_mechanism": "Embedded tool metadata can direct the agent to access local configuration files and transmit their contents to a remote service.",
                "security_impact": "Exfiltration of local configuration context and secret-bearing parameters.",
                "source_knowledge_id": "AML.CS0054#AML.T0086",
            }
        ],
        "security_basic_flow": [
            step("SBF1", "The AI agent verifies remote tool manifest provenance."),
            step("SBF2", "The AI agent classifies local configuration data by sensitivity."),
        ],
        "security_alternative_flows": [
            OrderedDict(
                [
                    ("saf_id", "SAF2"),
                    ("mitigates", ["T1"]),
                    ("entry_condition", "SBF2 determines that the tool request requires sensitive local configuration data."),
                    ("steps", [
                        step("SAF2.1", "The AI agent denies access to the sensitive configuration data."),
                        step("SAF2.2", "The AI agent logs the remote tool request for review."),
                        step("SAF2.3", "The AI agent returns to BF1."),
                    ]),
                ]
            )
        ],
        "review_notes": "Reviewed as a remote-tool exfiltration case focused on provenance and local data minimization.",
    }

    gold_cases = []
    for use_case_id in GOLD_SUBSET_IDS:
        base_case = flow_map[use_case_id]
        reg = registry_map[use_case_id]
        reviewed_security_basic_flow = []
        for item in manual[use_case_id]["security_basic_flow"]:
            reviewed_security_basic_flow.append(
                sbf_step(
                    item["step_id"],
                    "BF{0}".format(step_number(item["step_id"])),
                    item["step_sentence"],
                )
            )
        gold_cases.append(
            OrderedDict(
                [
                    ("use_case_id", use_case_id),
                    ("dataset", reg["dataset"]),
                    ("split", reg["split"]),
                    ("source_knowledge_id", reg["source_knowledge_id"]),
                    ("source_title", reg["source_title"]),
                    ("review_status", "ai_reviewed_seed_v0.1"),
                    ("review_disclaimer", "This subset was carefully reviewed by Codex/AI, not by an external human security expert."),
                    ("functional_flow", base_case),
                    ("reviewed_threat_records", manual[use_case_id]["threat_records"]),
                    ("reviewed_security_augmented_flow", OrderedDict(
                        [
                            ("security_basic_flow", reviewed_security_basic_flow),
                            ("security_alternative_flows", manual[use_case_id]["security_alternative_flows"]),
                        ]
                    )),
                    ("review_notes", manual[use_case_id]["review_notes"]),
                ]
            )
        )
    return {
        "meta": {
            "dataset_name": "SAAFG-Bench AI-reviewed gold subset seed",
            "version": "v0.1",
            "generated_at_utc": now_utc(),
            "case_count": len(gold_cases),
            "notes": [
                "This file is a carefully reviewed AI-generated seed subset.",
                "It is suitable as an internal gold-seed package or as an author-confirmed subset after human spot checking.",
                "It should not be described as expert-annotated gold without additional human confirmation.",
            ],
        },
        "gold_subset_cases": gold_cases,
    }


def build_annotation_guideline():
    return """# SAAFG Annotation Guideline v0.1

## Purpose

This guideline defines how to interpret and annotate the four benchmark layers:

1. requirement_text
2. functional_use_case_flow
3. threat_records
4. security_augmented_flow

## Functional Flow Rules

1. Each BF step should describe one act-object pair.
2. Keep step sentences concise and action oriented.
3. Do not inject security controls into the functional flow layer unless they already exist in the source specification.
4. If the source flow is linear, do not fabricate business AF branches.

## Threat Record Rules

1. A threat must be anchored to one or more BF steps.
2. Threat Name should describe the threat category in concise form.
3. Threat Mechanism should explain how the threat acts on the flow.
4. Security Impact should explain why the threat matters.
5. Source Knowledge ID must point to OWASP or ATLAS evidence.

## Security-Augmented Flow Rules

1. A defense must be expressed as SBF/SAF artifacts, not only as a flat mitigation sentence.
2. Each SAF must reference at least one Threat ID using Mitigates.
3. Each SAF must include an Entry Condition.
4. Each SAF must contain a return target or an explicit termination action.
5. The original BF order must remain valid after insertion.

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


def build_evaluation_protocol():
    return """# SAAFG Evaluation Protocol v0.1

## Benchmark Tasks

### Task A: Threat Anchoring

Input:
- functional_use_case_flow

Output:
- threat_records

Metrics:
- threat validity
- anchor step correctness
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

## Dataset Layers

- Silver set:
  large-scale, source-grounded, heuristic or model-assisted
- AI-reviewed gold subset seed:
  small-scale, carefully reviewed by AI/Codex, suitable for author confirmation

## Use of Critic

The Critic may be used for proxy evaluation on the silver set.
The AI-reviewed gold subset seed should be used for stricter qualitative checks and later human confirmation.
"""


def build_readme(metadata, registry, splits):
    return """# SAAFG-Bench v0.1

## Summary

This package provides the first benchmark-oriented release for Security-Augmented Alternative Flow Generation (SAAFG).

It contains:

- a case registry
- split files
- functional flow inputs
- source-grounded silver threat records
- source-grounded silver security-augmented flows
- an AI-reviewed gold subset seed
- schema and protocol documents

## Counts

- total cases: {total}
- train cases: {train}
- dev cases: {dev}
- test cases: {test}
- AI-reviewed gold subset seed cases: {gold}

## Important note

The gold subset in this release is AI-reviewed by Codex/AI.
It is not expert-annotated gold.
If the authors later confirm or revise it, it can be upgraded to an author-verified subset.
""".format(
        total=metadata["case_count"],
        train=len(splits["train"]),
        dev=len(splits["dev"]),
        test=len(splits["test"]),
        gold=len(splits["gold_subset"]),
    )


def build_public_release_manifest():
    return OrderedDict(
        [
            ("benchmark_name", "SAAFG-Bench"),
            ("version", "v0.1"),
            ("public_files", [
                "0_Data/6_SAAFG/saafg_benchmark_metadata_v0_1.json",
                "0_Data/6_SAAFG/saafg_schema_v0_1.json",
                "0_Data/6_SAAFG/saafg_case_registry_v0_1.json",
                "0_Data/6_SAAFG/saafg_case_registry_v0_1.csv",
                "0_Data/6_SAAFG/saafg_splits_v0_1.json",
                "0_Data/6_SAAFG/saafg_annotation_guideline_v0_1.md",
                "0_Data/6_SAAFG/saafg_evaluation_protocol_v0_1.md",
                "0_Data/6_SAAFG/README_benchmark_v0_1.md",
                "0_Data/6_SAAFG/1_Input_Functional_Flows/saafg_functional_flows_silver_v0_1.json",
                "0_Data/6_SAAFG/2_RedTeam_Threat_Records/saafg_threat_records_silver_v0_1.json",
                "0_Data/6_SAAFG/5_Gold_or_Human_Check/saafg_ai_reviewed_gold_subset_seed_v0_1.json",
            ]),
            ("non_public_internal_reference_files", [
                "0_Data/3_Experiment_Result/*",
                "ReadMe_旧版_需求文本安全分析基线.md",
            ]),
            ("release_note", "The gold subset file is publishable only if it is clearly labeled as AI-reviewed seed data unless manually confirmed by the authors."),
        ]
    )


def write_registry_csv(path, registry):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(registry[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in registry:
            writer.writerow(row)


def build_public_release_manifest_v2():
    return OrderedDict(
        [
            ("benchmark_name", "SAAFG-Bench"),
            ("version", "v0.1"),
            ("public_files", [
                "0_Data/6_SAAFG/saafg_benchmark_metadata_v0_1.json",
                "0_Data/6_SAAFG/saafg_schema_v0_1.json",
                "0_Data/6_SAAFG/saafg_case_registry_v0_1.json",
                "0_Data/6_SAAFG/saafg_case_registry_v0_1.csv",
                "0_Data/6_SAAFG/saafg_splits_v0_1.json",
                "0_Data/6_SAAFG/saafg_annotation_guideline_v0_1.md",
                "0_Data/6_SAAFG/saafg_evaluation_protocol_v0_1.md",
                "0_Data/6_SAAFG/README_benchmark_v0_1.md",
                "0_Data/6_SAAFG/1_Input_Functional_Flows/saafg_functional_flows_silver_v0_1.json",
                "0_Data/6_SAAFG/2_RedTeam_Threat_Records/saafg_threat_records_silver_v0_1.json",
                "0_Data/6_SAAFG/3_BlueTeam_SA_Flows/saafg_security_augmented_flows_silver_v0_1.json",
                "0_Data/6_SAAFG/4_Critic_Reports/saafg_critic_report_schema_v0_1.json",
                "0_Data/6_SAAFG/5_Gold_or_Human_Check/saafg_ai_reviewed_gold_subset_seed_v0_1.json",
                "0_Data/6_SAAFG/5_Gold_or_Human_Check/saafg_ai_reviewed_gold_subset_review_notes_v0_1.md",
            ]),
            ("non_public_internal_reference_files", [
                "0_Data/3_Experiment_Result/*",
                "ReadMe_旧版_需求文本安全分析基线.md",
            ]),
            ("release_note", "The gold subset file is publishable only if it is clearly labeled as AI-reviewed seed data unless manually confirmed by the authors."),
        ]
    )


def build_critic_report_schema():
    return OrderedDict(
        [
            ("dataset_name", "SAAFG Critic Report Schema"),
            ("version", "v0.1"),
            ("fields", [
                "use_case_id",
                "overall_decision",
                "threat_validity",
                "threat_coverage",
                "traceability",
                "branch_closure",
                "flow_consistency",
                "notes",
            ]),
            ("decision_values", ["accept", "revise", "reject"]),
        ]
    )


def main():
    registry, flows, owasp_cases, atlas_cases = build_case_registry()
    splits = build_splits(registry)
    schema = build_schema()
    silver_flows = build_silver_functional_flows(registry, flows)
    silver_threats = build_silver_threat_records(registry, flows, owasp_cases, atlas_cases)
    silver_security_augmented_flows = build_silver_security_augmented_flows(
        registry,
        flows,
        silver_threats["threat_record_cases"],
    )
    gold_subset = build_ai_reviewed_gold_subset(registry, flows)
    metadata = OrderedDict(
        [
            ("benchmark_name", "SAAFG-Bench"),
            ("version", "v0.1"),
            ("generated_at_utc", now_utc()),
            ("case_count", len(registry)),
            ("dataset_breakdown", OrderedDict(
                [
                    ("owasp", sum(1 for row in registry if row["dataset"] == "owasp")),
                    ("atlas", sum(1 for row in registry if row["dataset"] == "atlas")),
                ]
            )),
            ("split_breakdown", OrderedDict(
                [
                    ("train", len(splits["train"])),
                    ("dev", len(splits["dev"])),
                    ("test", len(splits["test"])),
                    ("gold_subset", len(splits["gold_subset"])),
                ]
            )),
            ("notes", [
                "This benchmark package is built on local OWASP/ATLAS-derived assets already present in the repository.",
                "The functional flow layer is derived from the parser-produced system_flow outputs of the legacy MA-NoRAG baseline.",
                "Silver threat records are source-grounded and heuristic-assisted.",
                "The gold subset in this release is an AI-reviewed seed, not expert-annotated gold.",
            ]),
        ]
    )

    write_json(BENCHMARK_ROOT / "saafg_benchmark_metadata_v0_1.json", metadata)
    write_json(BENCHMARK_ROOT / "saafg_schema_v0_1.json", schema)
    write_json(BENCHMARK_ROOT / "saafg_case_registry_v0_1.json", {"cases": registry})
    write_registry_csv(BENCHMARK_ROOT / "saafg_case_registry_v0_1.csv", registry)
    write_json(BENCHMARK_ROOT / "saafg_splits_v0_1.json", splits)
    write_text(BENCHMARK_ROOT / "saafg_annotation_guideline_v0_1.md", build_annotation_guideline())
    write_text(BENCHMARK_ROOT / "saafg_evaluation_protocol_v0_1.md", build_evaluation_protocol())
    write_text(BENCHMARK_ROOT / "README_benchmark_v0_1.md", build_readme(metadata, registry, splits))
    write_json(BENCHMARK_ROOT / "1_Input_Functional_Flows" / "saafg_functional_flows_silver_v0_1.json", silver_flows)
    write_json(BENCHMARK_ROOT / "2_RedTeam_Threat_Records" / "saafg_threat_records_silver_v0_1.json", silver_threats)
    write_json(
        BENCHMARK_ROOT / "3_BlueTeam_SA_Flows" / "saafg_security_augmented_flows_silver_v0_1.json",
        silver_security_augmented_flows,
    )
    write_json(
        BENCHMARK_ROOT / "4_Critic_Reports" / "saafg_critic_report_schema_v0_1.json",
        build_critic_report_schema(),
    )
    write_json(BENCHMARK_ROOT / "5_Gold_or_Human_Check" / "saafg_ai_reviewed_gold_subset_seed_v0_1.json", gold_subset)
    write_text(
        BENCHMARK_ROOT / "5_Gold_or_Human_Check" / "saafg_ai_reviewed_gold_subset_review_notes_v0_1.md",
        "# SAAFG AI-Reviewed Gold Subset Seed Review Notes v0.1\n\n"
        "This file accompanies `saafg_ai_reviewed_gold_subset_seed_v0_1.json`.\n\n"
        "The subset was reviewed case by case by Codex/AI based on:\n"
        "- the functional flow input\n"
        "- the OWASP/ATLAS source case\n"
        "- the SAAFG task definition and schema\n\n"
        "It is suitable as:\n"
        "- an internal gold-seed package\n"
        "- an author-confirmation package\n"
        "- a case-study benchmark slice\n\n"
        "It should not be described as expert-annotated gold without additional human confirmation.\n"
    )
    write_json(BENCHMARK_ROOT / "public_release_manifest_v0_1.json", build_public_release_manifest_v2())

    print("Built SAAFG-Bench v0.1")
    print("Cases:", len(registry))
    print("Gold subset seed cases:", len(GOLD_SUBSET_IDS))


if __name__ == "__main__":
    main()
