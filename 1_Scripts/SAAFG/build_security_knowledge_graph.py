#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build the static knowledge graph for Rec-EvoGraph-RAG v0.2.

The graph is intentionally built from source security knowledge only
(OWASP/ATLAS), not from SAAFG silver outputs, so that later experiments can
avoid benchmark leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[2]
KB_DIR = BASE_DIR / "0_Data" / "5_Knowledge_Base"
KB_SOURCE_DIR = KB_DIR / "source"

DEFAULT_OWASP_SOURCE = KB_SOURCE_DIR / "owasp_knowledge.json"
DEFAULT_ATLAS_SOURCE = (
    KB_SOURCE_DIR / "mitre_atlas_knowledge.json"
)
DEFAULT_OUTPUT_DIR = KB_DIR / "recevograph_rag"

VERSION = "v0.2"
GRAPH_NAME = "Rec-EvoGraph-RAG static security knowledge graph"
CREATED_BY = "static_ingestion"
DEFAULT_EDGE_WEIGHT = 1.0

NODE_TYPES = {
    "Risk",
    "AttackPattern",
    "Technique",
    "Mitigation",
    "SourceEvidence",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_key(value: Any, max_len: int = 96) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"['`]", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "empty"
    if len(text) > max_len:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        text = f"{text[: max_len - 11].rstrip('_')}_{digest}"
    return text


def stable_hash(*parts: Any, length: int = 12) -> str:
    joined = "\u241f".join(str(part or "") for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]


def compact_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def truncate(text: Any, max_len: int = 500) -> str:
    value = compact_space(text)
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip() + "..."


def strip_ordinal(title: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", title or "").strip()


def humanize_anchor(anchor: str) -> str:
    return compact_space(str(anchor or "").replace("_", " ")).title()


def source_label(dataset: str, source_id: Any, step_index: Optional[int] = None) -> str:
    label = f"{dataset}:{source_id}"
    if step_index is not None:
        label = f"{label}#step{step_index}"
    return label


def global_source_label(dataset: str, source_id: Any, global_id: Any) -> str:
    label = f"{dataset}:{source_id}"
    if global_id not in (None, ""):
        label = f"{label}#global{global_id}"
    return label


def clean_source_title(title: str, fallback: str) -> str:
    text = compact_space(title)
    if text:
        return text
    return compact_space(fallback)


def merge_unique(existing: Sequence[Any], incoming: Iterable[Any]) -> List[Any]:
    seen = {str(item) for item in existing}
    merged = list(existing)
    for item in incoming:
        if item is None:
            continue
        key = str(item)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.stats: Dict[str, Any] = {
            "owasp_items": 0,
            "owasp_items_with_mitigation_sections": 0,
            "owasp_mitigation_section_total": 0,
            "atlas_items": 0,
            "atlas_procedure_steps_total": 0,
            "atlas_procedure_steps_with_mitigations": 0,
            "atlas_procedure_steps_without_mitigations": 0,
            "atlas_mitigation_reference_total": 0,
            "warnings": [],
        }

    def add_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        description: str = "",
        dataset: str = "",
        source_ids: Optional[Iterable[Any]] = None,
        source_files: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unknown node_type={node_type!r} for {node_id}")
        incoming_source_ids = list(source_ids or [])
        incoming_source_files = list(source_files or [])
        incoming_metadata = dict(metadata or {})

        existing = self.nodes.get(node_id)
        if existing is None:
            self.nodes[node_id] = {
                "node_id": node_id,
                "node_type": node_type,
                "name": compact_space(name),
                "description": compact_space(description),
                "dataset": dataset,
                "source_ids": merge_unique([], incoming_source_ids),
                "source_files": merge_unique([], incoming_source_files),
                "metadata": incoming_metadata,
            }
            return node_id

        if existing["node_type"] != node_type:
            raise ValueError(
                f"Node type conflict for {node_id}: {existing['node_type']} vs {node_type}"
            )
        if not existing.get("description") and description:
            existing["description"] = compact_space(description)
        existing["source_ids"] = merge_unique(existing.get("source_ids", []), incoming_source_ids)
        existing["source_files"] = merge_unique(existing.get("source_files", []), incoming_source_files)
        if not existing.get("dataset") and dataset:
            existing["dataset"] = dataset
        for key, value in incoming_metadata.items():
            if key not in existing["metadata"] or existing["metadata"].get(key) in (None, "", []):
                existing["metadata"][key] = value
        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        source_dataset: str,
        source_ids: Optional[Iterable[Any]] = None,
        source_files: Optional[Iterable[str]] = None,
        source_evidence_ids: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        weight: float = DEFAULT_EDGE_WEIGHT,
    ) -> str:
        key = (source, relation, target)
        incoming_source_ids = list(source_ids or [])
        incoming_source_files = list(source_files or [])
        incoming_evidence_ids = list(source_evidence_ids or [])
        incoming_metadata = dict(metadata or {})

        existing = self.edges.get(key)
        if existing is None:
            edge_id = f"edge::{normalize_key(relation, 40)}::{stable_hash(source, relation, target)}"
            self.edges[key] = {
                "edge_id": edge_id,
                "source": source,
                "target": target,
                "relation": relation,
                "weight": float(weight),
                "base_weight": float(weight),
                "evidence_count": len(set(map(str, incoming_evidence_ids))) or 1,
                "source_dataset": source_dataset,
                "source_ids": merge_unique([], incoming_source_ids),
                "source_files": merge_unique([], incoming_source_files),
                "source_evidence_ids": merge_unique([], incoming_evidence_ids),
                "created_by": CREATED_BY,
                "metadata": incoming_metadata,
            }
            return edge_id

        existing["source_ids"] = merge_unique(existing.get("source_ids", []), incoming_source_ids)
        existing["source_files"] = merge_unique(existing.get("source_files", []), incoming_source_files)
        before = len(existing.get("source_evidence_ids", []))
        existing["source_evidence_ids"] = merge_unique(
            existing.get("source_evidence_ids", []), incoming_evidence_ids
        )
        after = len(existing.get("source_evidence_ids", []))
        existing["evidence_count"] = max(int(existing.get("evidence_count") or 1), after, before)
        if source_dataset and source_dataset not in str(existing.get("source_dataset", "")):
            existing["source_dataset"] = "+".join(
                sorted(set(str(existing.get("source_dataset", "")).split("+")) | {source_dataset})
            ).strip("+")
        for key_name, value in incoming_metadata.items():
            if key_name not in existing["metadata"] or existing["metadata"].get(key_name) in (
                None,
                "",
                [],
            ):
                existing["metadata"][key_name] = value
        return str(existing["edge_id"])


def build_owasp(builder: GraphBuilder, items: Sequence[Dict[str, Any]], source_path: Path) -> None:
    source_file = source_path.name
    builder.stats["owasp_items"] = len(items)

    for item in items:
        source_id = str(item.get("id") or "")
        global_id = item.get("global_id", "")
        item_index = item.get("item_index", "")
        source_title = clean_source_title(item.get("source_title", ""), source_id)
        evidence_source_id = global_source_label("owasp", source_id, global_id)

        evidence_id = f"evidence::owasp::{normalize_key(global_id if global_id != '' else item_index)}"
        risk_id = f"risk::owasp::{normalize_key(source_id or source_title)}"

        evidence_text = "\n".join(
            part
            for part in [
                f"source_title: {source_title}",
                f"original_text: {compact_space(item.get('original_text'))}",
                f"requirement_text: {compact_space(item.get('requirement_text'))}",
                f"business_value: {compact_space(item.get('business_value'))}",
                "implicit_risk_hints: " + ", ".join(item.get("implicit_risk_hints") or []),
            ]
            if part and not part.endswith(": ")
        )

        builder.add_node(
            evidence_id,
            "SourceEvidence",
            name=f"OWASP evidence {source_id} #{global_id}",
            description=truncate(evidence_text, 1200),
            dataset="owasp",
            source_ids=[evidence_source_id],
            source_files=[source_file],
            metadata={
                "global_id": global_id,
                "item_index": item_index,
                "source_title": source_title,
                "source_relative_path": item.get("source_relative_path"),
            },
        )

        builder.add_node(
            risk_id,
            "Risk",
            name=source_title,
            description=compact_space(item.get("original_text") or item.get("requirement_text")),
            dataset="owasp",
            source_ids=[source_id],
            source_files=[source_file],
            metadata={
                "source_title": source_title,
                "source_relative_path": item.get("source_relative_path"),
            },
        )
        builder.add_edge(
            risk_id,
            evidence_id,
            "supported_by",
            "owasp",
            source_ids=[source_id],
            source_files=[source_file],
            source_evidence_ids=[evidence_id],
        )

        hints = [compact_space(hint) for hint in (item.get("implicit_risk_hints") or []) if compact_space(hint)]
        if not hints:
            hints = ["Primary risk pattern"]

        attack_ids: List[str] = []
        for hint in hints:
            attack_id = (
                f"attack_pattern::owasp::{normalize_key(source_id or source_title, 48)}::"
                f"{normalize_key(hint, 48)}::{normalize_key(global_id if global_id != '' else item_index, 24)}"
            )
            attack_ids.append(attack_id)
            builder.add_node(
                attack_id,
                "AttackPattern",
                name=f"{source_title}: {hint}",
                description=compact_space(item.get("original_text") or item.get("requirement_text")),
                dataset="owasp",
                source_ids=[evidence_source_id],
                source_files=[source_file],
                metadata={
                    "risk_hint": hint,
                    "source_title": source_title,
                    "source_knowledge_id": source_id,
                    "global_id": global_id,
                },
            )
            builder.add_edge(
                attack_id,
                risk_id,
                "exploits",
                "owasp",
                source_ids=[source_id],
                source_files=[source_file],
                source_evidence_ids=[evidence_id],
                metadata={"risk_hint": hint},
            )
            builder.add_edge(
                attack_id,
                evidence_id,
                "supported_by",
                "owasp",
                source_ids=[source_id],
                source_files=[source_file],
                source_evidence_ids=[evidence_id],
            )

        mitigation_sections = item.get("prevention_and_mitigation_strategies_subsections") or {}
        if mitigation_sections:
            builder.stats["owasp_items_with_mitigation_sections"] += 1

        for title, text in mitigation_sections.items():
            clean_title = strip_ordinal(str(title))
            mitigation_id = f"mitigation::{normalize_key(clean_title or text, 80)}"
            builder.stats["owasp_mitigation_section_total"] += 1
            builder.add_node(
                mitigation_id,
                "Mitigation",
                name=clean_title or truncate(text, 80),
                description=compact_space(text),
                dataset="owasp",
                source_ids=[source_id],
                source_files=[source_file],
                metadata={
                    "mitigation_title": clean_title,
                    "source_title": source_title,
                    "source_knowledge_id": source_id,
                },
            )
            builder.add_edge(
                mitigation_id,
                evidence_id,
                "supported_by",
                "owasp",
                source_ids=[source_id],
                source_files=[source_file],
                source_evidence_ids=[evidence_id],
            )
            builder.add_edge(
                risk_id,
                mitigation_id,
                "mitigated_by",
                "owasp",
                source_ids=[source_id],
                source_files=[source_file],
                source_evidence_ids=[evidence_id],
                metadata={"mitigation_title": clean_title},
            )
            for attack_id in attack_ids:
                builder.add_edge(
                    attack_id,
                    mitigation_id,
                    "mitigated_by",
                    "owasp",
                    source_ids=[source_id],
                    source_files=[source_file],
                    source_evidence_ids=[evidence_id],
                    metadata={"mitigation_title": clean_title},
                )


def extract_atlas_tactic(step: Dict[str, Any]) -> Dict[str, Any]:
    tactic = step.get("tactic") or {}
    return tactic if isinstance(tactic, dict) else {}


def extract_atlas_technique(step: Dict[str, Any]) -> Dict[str, Any]:
    technique = step.get("technique") or {}
    return technique if isinstance(technique, dict) else {}


def iter_mitigation_items(mitigations: Any) -> Iterable[Tuple[str, str]]:
    if not isinstance(mitigations, list):
        return
    for item in mitigations:
        if not isinstance(item, dict):
            continue
        for anchor, text in item.items():
            anchor_text = compact_space(anchor)
            mitigation_text = compact_space(text)
            if anchor_text or mitigation_text:
                yield anchor_text, mitigation_text


def build_atlas(builder: GraphBuilder, items: Sequence[Dict[str, Any]], source_path: Path) -> None:
    source_file = source_path.name
    builder.stats["atlas_items"] = len(items)

    for item in items:
        case_id = str(item.get("id") or "")
        global_id = item.get("global_id", "")
        source_name = clean_source_title(item.get("source_name", ""), case_id)
        procedure = item.get("procedure") or []
        if not isinstance(procedure, list):
            builder.stats["warnings"].append(f"ATLAS case {case_id} has non-list procedure")
            continue

        for step_index, step in enumerate(procedure, start=1):
            if not isinstance(step, dict):
                builder.stats["warnings"].append(f"ATLAS case {case_id} step {step_index} is not an object")
                continue

            builder.stats["atlas_procedure_steps_total"] += 1
            tactic = extract_atlas_tactic(step)
            technique = extract_atlas_technique(step)
            tactic_id_raw = tactic.get("id") or tactic.get("anchor") or "unknown_tactic"
            technique_id_raw = technique.get("id") or technique.get("anchor") or f"unknown_technique_{step_index}"
            tactic_name = clean_source_title(tactic.get("name", ""), str(tactic_id_raw))
            technique_name = clean_source_title(technique.get("name", ""), str(technique_id_raw))
            step_description = compact_space(step.get("description"))

            evidence_id = (
                f"evidence::atlas::{normalize_key(case_id or global_id, 48)}::step_{step_index:03d}"
            )
            risk_id = f"risk::atlas_tactic::{normalize_key(tactic_id_raw)}"
            technique_id = f"technique::atlas::{normalize_key(technique_id_raw)}"
            attack_id = (
                f"attack_pattern::atlas::{normalize_key(case_id or global_id, 48)}::"
                f"step_{step_index:03d}::{normalize_key(technique_id_raw, 36)}"
            )

            step_source_id = source_label("atlas", case_id, step_index)
            evidence_text = "\n".join(
                part
                for part in [
                    f"source_name: {source_name}",
                    f"tactic: {tactic_name}",
                    f"technique: {technique_name}",
                    f"procedure_step: {step_description}",
                    f"case_summary: {compact_space(item.get('source_summary'))}",
                ]
                if part and not part.endswith(": ")
            )

            builder.add_node(
                evidence_id,
                "SourceEvidence",
                name=f"ATLAS evidence {case_id} step {step_index}",
                description=truncate(evidence_text, 1200),
                dataset="atlas",
                source_ids=[step_source_id],
                source_files=[source_file],
                metadata={
                    "global_id": global_id,
                    "case_id": case_id,
                    "source_name": source_name,
                    "step_index": step_index,
                    "tactic_id": tactic.get("id"),
                    "tactic_name": tactic_name,
                    "technique_id": technique.get("id"),
                    "technique_name": technique_name,
                },
            )

            builder.add_node(
                risk_id,
                "Risk",
                name=f"ATLAS tactic: {tactic_name}",
                description=compact_space(tactic.get("description")),
                dataset="atlas",
                source_ids=[tactic.get("id") or tactic.get("anchor")],
                source_files=[source_file],
                metadata={
                    "risk_family": "atlas_tactic",
                    "tactic_id": tactic.get("id"),
                    "tactic_anchor": tactic.get("anchor"),
                    "tactic_name": tactic_name,
                },
            )
            builder.add_node(
                technique_id,
                "Technique",
                name=technique_name,
                description=compact_space(technique.get("description")),
                dataset="atlas",
                source_ids=[technique.get("id") or technique.get("anchor")],
                source_files=[source_file],
                metadata={
                    "technique_id": technique.get("id"),
                    "technique_anchor": technique.get("anchor"),
                    "subtechnique_of": technique.get("subtechnique_of"),
                    "tactic_id": tactic.get("id"),
                    "tactic_name": tactic_name,
                },
            )
            builder.add_node(
                attack_id,
                "AttackPattern",
                name=f"{technique_name} in {source_name}",
                description=step_description or compact_space(technique.get("description")),
                dataset="atlas",
                source_ids=[step_source_id],
                source_files=[source_file],
                metadata={
                    "case_id": case_id,
                    "step_index": step_index,
                    "source_name": source_name,
                    "tactic_id": tactic.get("id"),
                    "tactic_name": tactic_name,
                    "technique_id": technique.get("id"),
                    "technique_name": technique_name,
                },
            )

            builder.add_edge(
                risk_id,
                evidence_id,
                "supported_by",
                "atlas",
                source_ids=[case_id],
                source_files=[source_file],
                source_evidence_ids=[evidence_id],
            )
            builder.add_edge(
                technique_id,
                evidence_id,
                "supported_by",
                "atlas",
                source_ids=[case_id],
                source_files=[source_file],
                source_evidence_ids=[evidence_id],
            )
            builder.add_edge(
                attack_id,
                evidence_id,
                "supported_by",
                "atlas",
                source_ids=[case_id],
                source_files=[source_file],
                source_evidence_ids=[evidence_id],
            )
            builder.add_edge(
                technique_id,
                risk_id,
                "belongs_to",
                "atlas",
                source_ids=[case_id],
                source_files=[source_file],
                source_evidence_ids=[evidence_id],
                metadata={"tactic_name": tactic_name},
            )
            builder.add_edge(
                technique_id,
                attack_id,
                "implements_or_examples",
                "atlas",
                source_ids=[case_id],
                source_files=[source_file],
                source_evidence_ids=[evidence_id],
                metadata={"step_index": step_index},
            )
            builder.add_edge(
                attack_id,
                risk_id,
                "exploits",
                "atlas",
                source_ids=[case_id],
                source_files=[source_file],
                source_evidence_ids=[evidence_id],
                metadata={"step_index": step_index, "technique_name": technique_name},
            )

            mitigation_items = list(iter_mitigation_items(step.get("mitigations")))
            if mitigation_items:
                builder.stats["atlas_procedure_steps_with_mitigations"] += 1
            else:
                builder.stats["atlas_procedure_steps_without_mitigations"] += 1

            for mitigation_anchor, mitigation_text in mitigation_items:
                builder.stats["atlas_mitigation_reference_total"] += 1
                mitigation_name = humanize_anchor(mitigation_anchor) or truncate(mitigation_text, 80)
                mitigation_id = f"mitigation::{normalize_key(mitigation_anchor or mitigation_text, 80)}"
                builder.add_node(
                    mitigation_id,
                    "Mitigation",
                    name=mitigation_name,
                    description=mitigation_text,
                    dataset="atlas",
                    source_ids=[mitigation_anchor],
                    source_files=[source_file],
                    metadata={
                        "mitigation_anchor": mitigation_anchor,
                        "case_id": case_id,
                        "source_name": source_name,
                    },
                )
                builder.add_edge(
                    mitigation_id,
                    evidence_id,
                    "supported_by",
                    "atlas",
                    source_ids=[case_id],
                    source_files=[source_file],
                    source_evidence_ids=[evidence_id],
                    metadata={"mitigation_anchor": mitigation_anchor},
                )
                for source_node in (risk_id, technique_id, attack_id):
                    builder.add_edge(
                        source_node,
                        mitigation_id,
                        "mitigated_by",
                        "atlas",
                        source_ids=[case_id],
                        source_files=[source_file],
                        source_evidence_ids=[evidence_id],
                        metadata={
                            "mitigation_anchor": mitigation_anchor,
                            "step_index": step_index,
                        },
                    )


def build_networkx_graph(nodes: Dict[str, Dict[str, Any]], edges: Dict[Tuple[str, str, str], Dict[str, Any]]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(name=GRAPH_NAME, version=VERSION)
    for node_id, node in nodes.items():
        attrs = {key: value for key, value in node.items() if key != "node_id"}
        graph.add_node(node_id, **attrs)
    for edge in edges.values():
        attrs = {key: value for key, value in edge.items() if key not in {"source", "target"}}
        graph.add_edge(edge["source"], edge["target"], key=edge["edge_id"], **attrs)
    return graph


def validate_graph(nodes: Dict[str, Dict[str, Any]], edges: Dict[Tuple[str, str, str], Dict[str, Any]]) -> Dict[str, Any]:
    missing_endpoints: List[Dict[str, str]] = []
    for edge in edges.values():
        if edge["source"] not in nodes or edge["target"] not in nodes:
            missing_endpoints.append(
                {
                    "edge_id": edge["edge_id"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "relation": edge["relation"],
                }
            )

    graph = build_networkx_graph(nodes, edges)
    isolated_nodes = sorted(nx.isolates(graph))

    outgoing_mitigated_by: Counter[str] = Counter()
    for edge in edges.values():
        if edge["relation"] == "mitigated_by":
            outgoing_mitigated_by[edge["source"]] += 1

    attack_patterns_without_mitigation = sorted(
        node_id
        for node_id, node in nodes.items()
        if node["node_type"] == "AttackPattern" and outgoing_mitigated_by[node_id] == 0
    )
    risks_without_mitigation = sorted(
        node_id
        for node_id, node in nodes.items()
        if node["node_type"] == "Risk" and outgoing_mitigated_by[node_id] == 0
    )

    return {
        "missing_edge_endpoint_count": len(missing_endpoints),
        "missing_edge_endpoints": missing_endpoints[:50],
        "isolated_node_count": len(isolated_nodes),
        "isolated_node_sample": isolated_nodes[:50],
        "attack_patterns_without_mitigation_count": len(attack_patterns_without_mitigation),
        "attack_patterns_without_mitigation_sample": attack_patterns_without_mitigation[:50],
        "risks_without_mitigation_count": len(risks_without_mitigation),
        "risks_without_mitigation_sample": risks_without_mitigation[:50],
    }


def summarize(
    builder: GraphBuilder,
    validation: Dict[str, Any],
    owasp_source: Path,
    atlas_source: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    node_type_counts = Counter(node["node_type"] for node in builder.nodes.values())
    edge_relation_counts = Counter(edge["relation"] for edge in builder.edges.values())
    edge_dataset_counts = Counter(edge["source_dataset"] for edge in builder.edges.values())
    atlas_steps = int(builder.stats["atlas_procedure_steps_total"] or 0)
    atlas_without = int(builder.stats["atlas_procedure_steps_without_mitigations"] or 0)

    return {
        "graph_name": GRAPH_NAME,
        "version": VERSION,
        "generated_at_utc": now_utc(),
        "created_by": CREATED_BY,
        "source_files": {
            "owasp": str(owasp_source),
            "atlas": str(atlas_source),
        },
        "output_dir": str(output_dir),
        "node_count": len(builder.nodes),
        "edge_count": len(builder.edges),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "edge_relation_counts": dict(sorted(edge_relation_counts.items())),
        "edge_source_dataset_counts": dict(sorted(edge_dataset_counts.items())),
        "ingestion_stats": {
            **builder.stats,
            "atlas_empty_mitigation_step_rate": round(atlas_without / atlas_steps, 6)
            if atlas_steps
            else None,
        },
        "validation": validation,
        "leakage_policy": (
            "Built only from OWASP/ATLAS source knowledge files. SAAFG silver threats/defenses are "
            "not ingested into this static graph."
        ),
    }


def sample_paths(nodes: Dict[str, Dict[str, Any]], edges: Dict[Tuple[str, str, str], Dict[str, Any]]) -> Dict[str, Any]:
    graph = build_networkx_graph(nodes, edges)
    risk_candidates = [
        node_id
        for node_id, node in nodes.items()
        if node["node_type"] == "Risk" and "prompt injection" in node.get("name", "").lower()
    ]
    prompt_paths: List[Dict[str, Any]] = []
    for risk_id in risk_candidates[:2]:
        for _, mitigation_id, _, edge_data in graph.out_edges(risk_id, keys=True, data=True):
            if edge_data.get("relation") != "mitigated_by":
                continue
            prompt_paths.append(
                {
                    "risk_id": risk_id,
                    "risk_name": nodes[risk_id]["name"],
                    "mitigation_id": mitigation_id,
                    "mitigation_name": nodes[mitigation_id]["name"],
                    "edge_weight": edge_data.get("weight"),
                    "source_ids": edge_data.get("source_ids", []),
                }
            )
            if len(prompt_paths) >= 8:
                break
        if len(prompt_paths) >= 8:
            break
    return {"prompt_injection_mitigation_paths": prompt_paths}


def build_report(metadata: Dict[str, Any], samples: Dict[str, Any]) -> str:
    stats = metadata["ingestion_stats"]
    validation = metadata["validation"]
    node_counts = metadata["node_type_counts"]
    edge_counts = metadata["edge_relation_counts"]
    dataset_counts = metadata["edge_source_dataset_counts"]

    lines = [
        "# Rec-EvoGraph-RAG Static KG Build Report",
        "",
        f"- graph_name: {metadata['graph_name']}",
        f"- version: {metadata['version']}",
        f"- generated_at_utc: {metadata['generated_at_utc']}",
        f"- output_dir: `{metadata['output_dir']}`",
        "",
        "## Leakage Policy",
        "",
        metadata["leakage_policy"],
        "",
        "## Source Files",
        "",
        f"- OWASP: `{metadata['source_files']['owasp']}`",
        f"- ATLAS: `{metadata['source_files']['atlas']}`",
        "",
        "## Graph Size",
        "",
        f"- nodes: {metadata['node_count']}",
        f"- edges: {metadata['edge_count']}",
        "",
        "## Node Type Counts",
        "",
    ]
    for name, count in node_counts.items():
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Edge Relation Counts", ""])
    for name, count in edge_counts.items():
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Edge Source Dataset Counts", ""])
    for name, count in dataset_counts.items():
        lines.append(f"- {name}: {count}")

    lines.extend(
        [
            "",
            "## Ingestion Stats",
            "",
            f"- owasp_items: {stats['owasp_items']}",
            f"- owasp_items_with_mitigation_sections: {stats['owasp_items_with_mitigation_sections']}",
            f"- owasp_mitigation_section_total: {stats['owasp_mitigation_section_total']}",
            f"- atlas_items: {stats['atlas_items']}",
            f"- atlas_procedure_steps_total: {stats['atlas_procedure_steps_total']}",
            f"- atlas_procedure_steps_with_mitigations: {stats['atlas_procedure_steps_with_mitigations']}",
            f"- atlas_procedure_steps_without_mitigations: {stats['atlas_procedure_steps_without_mitigations']}",
            f"- atlas_empty_mitigation_step_rate: {stats['atlas_empty_mitigation_step_rate']}",
            f"- atlas_mitigation_reference_total: {stats['atlas_mitigation_reference_total']}",
            "",
            "## Validation",
            "",
            f"- missing_edge_endpoint_count: {validation['missing_edge_endpoint_count']}",
            f"- isolated_node_count: {validation['isolated_node_count']}",
            "- attack_patterns_without_mitigation_count: "
            f"{validation['attack_patterns_without_mitigation_count']}",
            f"- risks_without_mitigation_count: {validation['risks_without_mitigation_count']}",
            "",
            "## Sample Prompt Injection Paths",
            "",
        ]
    )
    prompt_paths = samples.get("prompt_injection_mitigation_paths") or []
    if not prompt_paths:
        lines.append("- No prompt injection mitigation path sample found.")
    else:
        for path in prompt_paths:
            lines.append(
                "- "
                f"{path['risk_name']} -> {path['mitigation_name']} "
                f"(weight={path['edge_weight']}, sources={', '.join(map(str, path['source_ids']))})"
            )

    warnings = stats.get("warnings") or []
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for warning in warnings[:100]:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Rec-EvoGraph-RAG static KG v0.2.")
    parser.add_argument("--owasp-source", type=Path, default=DEFAULT_OWASP_SOURCE)
    parser.add_argument("--atlas-source", type=Path, default=DEFAULT_ATLAS_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    owasp_source = args.owasp_source.resolve()
    atlas_source = args.atlas_source.resolve()
    output_dir = args.output_dir.resolve()

    if not owasp_source.exists():
        raise FileNotFoundError(f"OWASP source not found: {owasp_source}")
    if not atlas_source.exists():
        raise FileNotFoundError(f"ATLAS source not found: {atlas_source}")

    builder = GraphBuilder()
    build_owasp(builder, read_json_list(owasp_source), owasp_source)
    build_atlas(builder, read_json_list(atlas_source), atlas_source)
    validation = validate_graph(builder.nodes, builder.edges)
    metadata = summarize(builder, validation, owasp_source, atlas_source, output_dir)
    samples = sample_paths(builder.nodes, builder.edges)

    nodes_sorted = sorted(builder.nodes.values(), key=lambda node: node["node_id"])
    edges_sorted = sorted(builder.edges.values(), key=lambda edge: edge["edge_id"])
    graph = build_networkx_graph(builder.nodes, builder.edges)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "graph_nodes.json", {"meta": metadata, "nodes": nodes_sorted})
    write_json(output_dir / "graph_edges.json", {"meta": metadata, "edges": edges_sorted})
    write_json(output_dir / "graph_metadata.json", metadata)
    (output_dir / "build_report.md").write_text(
        build_report(metadata, samples),
        encoding="utf-8",
    )
    with (output_dir / "networkx_graph.pkl").open("wb") as f:
        pickle.dump(graph, f)

    print(f"[Done] Built {metadata['node_count']} nodes and {metadata['edge_count']} edges.")
    print(f"[Done] Output directory: {output_dir}")
    print(f"[Done] Build report: {output_dir / 'build_report.md'}")


if __name__ == "__main__":
    main()
