#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Static Rec-EvoGraph-RAG retrieval utilities for SAAFG v0.2.

This module does not call an LLM and does not modify existing baseline scripts.
It loads the static graph artifacts produced by
build_security_knowledge_graph.py, retrieves role-specific graph evidence,
and applies a deterministic reranking function.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[2]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"
DEFAULT_KG_DIR = BASE_DIR / "0_Data" / "5_Knowledge_Base" / "recevograph_rag"
DEFAULT_FLOW_PATH = (
    SAAFG_ROOT / "1_Input_Functional_Flows" / "functional_use_case_flows.json"
)
DEFAULT_THREAT_PATH = (
    SAAFG_ROOT / "2_RedTeam_Threat_Records" / "threat_records.json"
)

VERSION = "v0.2"
MAX_DESCRIPTION_CHARS = 700
MAX_EVIDENCE_CHARS = 650

SEARCH_NODE_TYPES = {"Risk", "AttackPattern", "Technique", "Mitigation"}
RED_SEARCH_NODE_TYPES = {"Risk", "AttackPattern", "Technique"}
BLUE_SOURCE_NODE_TYPES = {"Risk", "AttackPattern", "Technique"}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "shall",
    "the",
    "their",
    "this",
    "to",
    "using",
    "via",
    "when",
    "with",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def truncate(text: Any, max_len: int) -> str:
    value = compact_space(text)
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip() + "..."


def tokenize(text: Any) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def token_overlap_score(query: str, candidate: str) -> float:
    query_tokens = set(tokenize(query))
    candidate_tokens = set(tokenize(candidate))
    if not query_tokens or not candidate_tokens:
        return 0.0
    intersection = len(query_tokens & candidate_tokens)
    if intersection == 0:
        return 0.0
    return intersection / math.sqrt(len(query_tokens) * len(candidate_tokens))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def edge_weight(edge: Dict[str, Any]) -> float:
    return safe_float(edge.get("weight"), 1.0)


def normalize_source_id(value: Any) -> str:
    return compact_space(value).lower()


def flatten_metadata(metadata: Any) -> str:
    if metadata is None:
        return ""
    if isinstance(metadata, dict):
        parts: List[str] = []
        for key, value in sorted(metadata.items()):
            if isinstance(value, (dict, list)):
                parts.append(flatten_metadata(value))
            else:
                parts.append(f"{key}: {value}")
        return " ".join(part for part in parts if part)
    if isinstance(metadata, list):
        return " ".join(flatten_metadata(item) for item in metadata)
    return compact_space(metadata)


def load_functional_cases(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = read_json(path)
    cases = payload.get("use_case_flows")
    if not isinstance(cases, list):
        raise ValueError(f"Expected use_case_flows list in {path}")
    return {case["use_case_id"]: case for case in cases}


def load_threat_cases(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = read_json(path)
    cases = payload.get("threat_record_cases")
    if not isinstance(cases, list):
        raise ValueError(f"Expected threat_record_cases list in {path}")
    return {case["use_case_id"]: case for case in cases}


def select_threat(threat_case: Dict[str, Any], threat_id: Optional[str]) -> Dict[str, Any]:
    threats = threat_case.get("threat_records") or []
    if not isinstance(threats, list) or not threats:
        raise ValueError(f"No threat_records available for {threat_case.get('use_case_id')}")
    if threat_id is None:
        return threats[0]
    for threat in threats:
        if str(threat.get("threat_id")) == threat_id:
            return threat
    raise ValueError(f"Unknown threat_id={threat_id!r} for {threat_case.get('use_case_id')}")


def get_basic_flow_steps(flow_case: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = flow_case.get("basic_flow") or []
    return [step for step in steps if isinstance(step, dict)]


def get_anchor_step_text(flow_case: Dict[str, Any], threat: Dict[str, Any]) -> str:
    anchors = threat.get("anchor_steps") or []
    anchor_set = {str(anchor) for anchor in anchors}
    matched = [
        step
        for step in get_basic_flow_steps(flow_case)
        if str(step.get("step_id")) in anchor_set
    ]
    if not matched:
        return ""
    return " ".join(
        f"{step.get('step_id')}: {step.get('step_sentence')} "
        f"{step.get('subject', '')} {step.get('verb', '')} {step.get('object', '')}"
        for step in matched
    )


def build_flow_query(flow_case: Dict[str, Any]) -> str:
    step_text = " ".join(
        f"{step.get('step_id')}: {step.get('step_sentence')} "
        f"{step.get('subject', '')} {step.get('verb', '')} {step.get('object', '')}"
        for step in get_basic_flow_steps(flow_case)
    )
    return compact_space(
        " ".join(
            [
                flow_case.get("source_requirement_text", ""),
                step_text,
            ]
        )
    )


def build_threat_query(flow_case: Dict[str, Any], threat: Dict[str, Any]) -> str:
    return compact_space(
        " ".join(
            [
                flow_case.get("source_requirement_text", ""),
                get_anchor_step_text(flow_case, threat),
                threat.get("threat_name", ""),
                threat.get("threat_mechanism", ""),
                threat.get("security_impact", ""),
                threat.get("source_knowledge_id", ""),
            ]
        )
    )


class RecEvoGraphRetriever:
    """Deterministic graph retriever and reranker for static Rec-EvoGraph-RAG."""

    def __init__(self, kg_dir: Path = DEFAULT_KG_DIR) -> None:
        self.kg_dir = Path(kg_dir)
        self.nodes_payload = read_json(self.kg_dir / "graph_nodes.json")
        self.edges_payload = read_json(self.kg_dir / "graph_edges.json")
        self.metadata = read_json(self.kg_dir / "graph_metadata.json")

        self.nodes: Dict[str, Dict[str, Any]] = {
            node["node_id"]: node for node in self.nodes_payload.get("nodes", [])
        }
        self.edges: List[Dict[str, Any]] = list(self.edges_payload.get("edges", []))
        self.out_edges: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.in_edges: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for edge in self.edges:
            self.out_edges[edge["source"]][edge["relation"]].append(edge)
            self.in_edges[edge["target"]][edge["relation"]].append(edge)

        self.search_node_ids = [
            node_id
            for node_id, node in self.nodes.items()
            if node.get("node_type") in SEARCH_NODE_TYPES
        ]
        self.search_documents = [self.node_search_text(self.nodes[node_id]) for node_id in self.search_node_ids]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
        self.search_matrix = self.vectorizer.fit_transform(self.search_documents)

    def node_search_text(self, node: Dict[str, Any]) -> str:
        return compact_space(
            " ".join(
                [
                    node.get("node_type", ""),
                    node.get("name", ""),
                    node.get("description", ""),
                    node.get("dataset", ""),
                    " ".join(str(item) for item in node.get("source_ids", []) or []),
                    flatten_metadata(node.get("metadata") or {}),
                ]
            )
        )

    def vector_search(
        self,
        query: str,
        node_types: Optional[Iterable[str]] = None,
        top_n: int = 30,
    ) -> List[Dict[str, Any]]:
        allowed = set(node_types or SEARCH_NODE_TYPES)
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.search_matrix).ravel()
        hits: List[Dict[str, Any]] = []
        for idx, score in enumerate(scores):
            node_id = self.search_node_ids[idx]
            node = self.nodes[node_id]
            if node.get("node_type") not in allowed:
                continue
            if score <= 0:
                continue
            hits.append({"node_id": node_id, "similarity": float(score), "node_type": node.get("node_type")})
        hits.sort(key=lambda item: (-item["similarity"], item["node_type"], item["node_id"]))
        return hits[:top_n]

    def node_summary(self, node_id: str, max_description_chars: int = MAX_DESCRIPTION_CHARS) -> Dict[str, Any]:
        node = self.nodes[node_id]
        metadata = node.get("metadata") or {}
        keep_metadata = {
            key: metadata.get(key)
            for key in [
                "source_title",
                "source_name",
                "source_knowledge_id",
                "case_id",
                "step_index",
                "tactic_id",
                "tactic_name",
                "technique_id",
                "technique_name",
                "mitigation_anchor",
                "mitigation_title",
            ]
            if key in metadata and metadata.get(key) not in (None, "", [])
        }
        return {
            "node_id": node_id,
            "node_type": node.get("node_type"),
            "name": node.get("name"),
            "description": truncate(node.get("description"), max_description_chars),
            "dataset": node.get("dataset"),
            "source_ids": node.get("source_ids") or [],
            "metadata": keep_metadata,
        }

    def edges_from(self, node_id: str, relation: str) -> List[Dict[str, Any]]:
        return list(self.out_edges.get(node_id, {}).get(relation, []))

    def edges_to(self, node_id: str, relation: str) -> List[Dict[str, Any]]:
        return list(self.in_edges.get(node_id, {}).get(relation, []))

    def relation_targets(self, node_id: str, relation: str) -> List[str]:
        return [edge["target"] for edge in self.edges_from(node_id, relation)]

    def relation_sources(self, node_id: str, relation: str) -> List[str]:
        return [edge["source"] for edge in self.edges_to(node_id, relation)]

    def collect_evidence(self, node_ids: Sequence[str], limit: int = 4) -> List[Dict[str, Any]]:
        evidence_ids: List[str] = []
        seen: set[str] = set()
        for node_id in node_ids:
            for edge in self.edges_from(node_id, "supported_by"):
                evidence_id = edge.get("target")
                if evidence_id and evidence_id not in seen:
                    seen.add(evidence_id)
                    evidence_ids.append(evidence_id)
            for relation_edges in self.out_edges.get(node_id, {}).values():
                for edge in relation_edges:
                    for evidence_id in edge.get("source_evidence_ids") or []:
                        if evidence_id in self.nodes and evidence_id not in seen:
                            seen.add(evidence_id)
                            evidence_ids.append(evidence_id)
        return [
            self.node_summary(evidence_id, max_description_chars=MAX_EVIDENCE_CHARS)
            for evidence_id in evidence_ids[:limit]
            if evidence_id in self.nodes
        ]

    def related_risks(self, node_id: str, limit: int = 3) -> List[str]:
        node_type = self.nodes[node_id].get("node_type")
        risks: List[str] = []
        if node_type == "Risk":
            risks.append(node_id)
        if node_type == "AttackPattern":
            risks.extend(self.relation_targets(node_id, "exploits"))
        if node_type == "Technique":
            risks.extend(self.relation_targets(node_id, "belongs_to"))
            for attack_id in self.relation_targets(node_id, "implements_or_examples"):
                risks.extend(self.relation_targets(attack_id, "exploits"))
        return self.unique_existing(risks)[:limit]

    def related_attack_patterns(self, node_id: str, limit: int = 4) -> List[str]:
        node_type = self.nodes[node_id].get("node_type")
        attacks: List[str] = []
        if node_type == "AttackPattern":
            attacks.append(node_id)
        if node_type == "Risk":
            attacks.extend(self.relation_sources(node_id, "exploits"))
        if node_type == "Technique":
            attacks.extend(self.relation_targets(node_id, "implements_or_examples"))
        return self.unique_existing(attacks)[:limit]

    def related_techniques(self, node_id: str, limit: int = 4) -> List[str]:
        node_type = self.nodes[node_id].get("node_type")
        techniques: List[str] = []
        if node_type == "Technique":
            techniques.append(node_id)
        if node_type == "AttackPattern":
            techniques.extend(self.relation_sources(node_id, "implements_or_examples"))
        if node_type == "Risk":
            techniques.extend(self.relation_sources(node_id, "belongs_to"))
        return self.unique_existing(techniques)[:limit]

    def related_mitigations(self, node_id: str, limit: int = 5) -> List[str]:
        mitigation_ids: List[str] = []
        mitigation_ids.extend(self.relation_targets(node_id, "mitigated_by"))
        for risk_id in self.related_risks(node_id, limit=4):
            mitigation_ids.extend(self.relation_targets(risk_id, "mitigated_by"))
        for attack_id in self.related_attack_patterns(node_id, limit=4):
            mitigation_ids.extend(self.relation_targets(attack_id, "mitigated_by"))
        return self.unique_existing(mitigation_ids)[:limit]

    def unique_existing(self, node_ids: Iterable[str]) -> List[str]:
        result: List[str] = []
        seen: set[str] = set()
        for node_id in node_ids:
            if node_id in self.nodes and node_id not in seen:
                seen.add(node_id)
                result.append(node_id)
        return result

    def edge_path_summary(self, path_edges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "source": edge.get("source"),
                "source_name": self.nodes.get(edge.get("source"), {}).get("name"),
                "relation": edge.get("relation"),
                "target": edge.get("target"),
                "target_name": self.nodes.get(edge.get("target"), {}).get("name"),
                "weight": edge_weight(edge),
                "source_ids": edge.get("source_ids") or [],
            }
            for edge in path_edges
        ]

    def source_priority(self, node_id: str) -> float:
        dataset = str(self.nodes[node_id].get("dataset") or "").lower()
        if dataset == "owasp":
            return 0.08
        if dataset == "atlas":
            return 0.06
        return 0.0

    def graph_support_score(self, node_id: str) -> float:
        evidence_count = len(self.collect_evidence([node_id], limit=20))
        mitigation_count = len(self.related_mitigations(node_id, limit=20))
        relation_count = sum(len(edges) for edges in self.out_edges.get(node_id, {}).values())
        raw = 0.4 * min(evidence_count, 5) + 0.4 * min(mitigation_count, 5) + 0.2 * min(relation_count, 5)
        return min(raw / 5.0, 1.0)

    def expand_red_candidates(self, hits: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        candidates: Dict[str, float] = {}
        for hit in hits:
            node_id = hit["node_id"]
            similarity = hit["similarity"]
            candidates[node_id] = max(candidates.get(node_id, 0.0), similarity)
            for related_id in (
                self.related_risks(node_id, limit=4)
                + self.related_attack_patterns(node_id, limit=4)
                + self.related_techniques(node_id, limit=4)
            ):
                related_type = self.nodes[related_id].get("node_type")
                if related_type in RED_SEARCH_NODE_TYPES:
                    candidates[related_id] = max(candidates.get(related_id, 0.0), similarity * 0.86)
        return candidates

    def retrieve_red(
        self,
        flow_case: Dict[str, Any],
        top_k: int = 6,
        candidate_top_n: int = 40,
        max_per_risk: int = 2,
    ) -> Dict[str, Any]:
        query = build_flow_query(flow_case)
        hits = self.vector_search(query, RED_SEARCH_NODE_TYPES, top_n=candidate_top_n)
        candidates = self.expand_red_candidates(hits)

        scored: List[Dict[str, Any]] = []
        for node_id, similarity in candidates.items():
            node = self.nodes[node_id]
            node_text = self.node_search_text(node)
            context_fit = token_overlap_score(query, node_text)
            graph_support = self.graph_support_score(node_id)
            type_bonus = {"AttackPattern": 0.08, "Risk": 0.06, "Technique": 0.04}.get(
                node.get("node_type"), 0.0
            )
            score = (
                0.58 * similarity
                + 0.18 * context_fit
                + 0.12 * graph_support
                + type_bonus
                + self.source_priority(node_id)
            )
            scored.append(
                {
                    "node_id": node_id,
                    "score": score,
                    "score_breakdown": {
                        "vector_similarity": round(similarity, 6),
                        "context_fit": round(context_fit, 6),
                        "graph_support": round(graph_support, 6),
                        "type_bonus": round(type_bonus, 6),
                        "source_priority": round(self.source_priority(node_id), 6),
                    },
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["node_id"]))
        selected: List[Dict[str, Any]] = []
        per_risk_count: Dict[str, int] = defaultdict(int)
        for item in scored:
            risk_ids = self.related_risks(item["node_id"], limit=1)
            risk_key = risk_ids[0] if risk_ids else item["node_id"]
            if per_risk_count[risk_key] >= max_per_risk:
                continue
            per_risk_count[risk_key] += 1
            selected.append(item)
            if len(selected) >= top_k:
                break

        return {
            "meta": {
                "retrieval_role": "red_team",
                "retriever_version": VERSION,
                "generated_at_utc": now_utc(),
                "kg_dir": str(self.kg_dir),
                "use_case_id": flow_case.get("use_case_id"),
                "top_k": top_k,
                "candidate_top_n": candidate_top_n,
                "rerank_policy": (
                    "TF-IDF candidate retrieval plus graph expansion, then deterministic score over "
                    "vector similarity, context fit, graph support, node type, and source priority."
                ),
            },
            "query": truncate(query, 1600),
            "items": [self.build_red_item(item, rank + 1) for rank, item in enumerate(selected)],
        }

    def build_red_item(self, item: Dict[str, Any], rank: int) -> Dict[str, Any]:
        node_id = item["node_id"]
        related_node_ids = self.unique_existing(
            [node_id]
            + self.related_risks(node_id, limit=3)
            + self.related_attack_patterns(node_id, limit=4)
            + self.related_techniques(node_id, limit=4)
            + self.related_mitigations(node_id, limit=4)
        )
        return {
            "rank": rank,
            "score": round(item["score"], 6),
            "score_breakdown": item["score_breakdown"],
            "focus": self.node_summary(node_id),
            "related_risks": [
                self.node_summary(risk_id, max_description_chars=420)
                for risk_id in self.related_risks(node_id, limit=3)
            ],
            "related_attack_patterns": [
                self.node_summary(attack_id, max_description_chars=420)
                for attack_id in self.related_attack_patterns(node_id, limit=4)
            ],
            "related_techniques": [
                self.node_summary(technique_id, max_description_chars=420)
                for technique_id in self.related_techniques(node_id, limit=4)
            ],
            "candidate_mitigations": [
                self.node_summary(mitigation_id, max_description_chars=420)
                for mitigation_id in self.related_mitigations(node_id, limit=4)
            ],
            "source_evidence": self.collect_evidence(related_node_ids, limit=4),
            "why_relevant": self.explain_red_relevance(node_id),
        }

    def explain_red_relevance(self, node_id: str) -> str:
        node = self.nodes[node_id]
        node_type = node.get("node_type")
        if node_type == "Risk":
            return "The flow query matched a risk node with linked attack patterns and mitigations."
        if node_type == "AttackPattern":
            return "The flow query matched an attack pattern that exploits a risk in the static graph."
        if node_type == "Technique":
            return "The flow query matched an ATLAS technique with procedure evidence and linked risk context."
        return "The flow query matched a graph node with source evidence."

    def source_id_matches(self, node_id: str, source_knowledge_id: str) -> bool:
        if not source_knowledge_id:
            return False
        needle = normalize_source_id(source_knowledge_id)
        node = self.nodes[node_id]
        source_ids = " ".join(str(item) for item in node.get("source_ids") or [])
        metadata = flatten_metadata(node.get("metadata") or {})
        haystack = normalize_source_id(f"{source_ids} {metadata} {node.get('name', '')}")
        return needle in haystack

    def blue_source_candidates(
        self,
        flow_case: Dict[str, Any],
        threat: Dict[str, Any],
        candidate_top_n: int,
    ) -> Dict[str, float]:
        query = build_threat_query(flow_case, threat)
        hits = self.vector_search(query, BLUE_SOURCE_NODE_TYPES, top_n=candidate_top_n)
        candidates: Dict[str, float] = {}
        for hit in hits:
            node_id = hit["node_id"]
            candidates[node_id] = max(candidates.get(node_id, 0.0), hit["similarity"])
            for related_id in (
                self.related_risks(node_id, limit=4)
                + self.related_attack_patterns(node_id, limit=4)
                + self.related_techniques(node_id, limit=4)
            ):
                if self.nodes[related_id].get("node_type") in BLUE_SOURCE_NODE_TYPES:
                    candidates[related_id] = max(candidates.get(related_id, 0.0), hit["similarity"] * 0.84)

        source_knowledge_id = compact_space(threat.get("source_knowledge_id"))
        if source_knowledge_id:
            for node_id, node in self.nodes.items():
                if node.get("node_type") in BLUE_SOURCE_NODE_TYPES and self.source_id_matches(
                    node_id, source_knowledge_id
                ):
                    candidates[node_id] = max(candidates.get(node_id, 0.0), 0.92)
        return candidates

    def mitigation_paths_from_source(self, source_id: str) -> List[Dict[str, Any]]:
        paths: List[Dict[str, Any]] = []

        for edge in self.edges_from(source_id, "mitigated_by"):
            if self.nodes.get(edge["target"], {}).get("node_type") == "Mitigation":
                paths.append({"mitigation_id": edge["target"], "path_edges": [edge], "path_score": 1.0})

        for edge1 in self.edges_from(source_id, "exploits") + self.edges_from(source_id, "belongs_to"):
            for edge2 in self.edges_from(edge1["target"], "mitigated_by"):
                if self.nodes.get(edge2["target"], {}).get("node_type") == "Mitigation":
                    paths.append(
                        {
                            "mitigation_id": edge2["target"],
                            "path_edges": [edge1, edge2],
                            "path_score": 0.82,
                        }
                    )

        for edge1 in self.edges_from(source_id, "implements_or_examples"):
            attack_id = edge1["target"]
            for edge2 in self.edges_from(attack_id, "mitigated_by"):
                if self.nodes.get(edge2["target"], {}).get("node_type") == "Mitigation":
                    paths.append(
                        {
                            "mitigation_id": edge2["target"],
                            "path_edges": [edge1, edge2],
                            "path_score": 0.86,
                        }
                    )
            for edge2 in self.edges_from(attack_id, "exploits"):
                for edge3 in self.edges_from(edge2["target"], "mitigated_by"):
                    if self.nodes.get(edge3["target"], {}).get("node_type") == "Mitigation":
                        paths.append(
                            {
                                "mitigation_id": edge3["target"],
                                "path_edges": [edge1, edge2, edge3],
                                "path_score": 0.72,
                            }
                        )
        return paths

    def retrieve_blue(
        self,
        flow_case: Dict[str, Any],
        threat: Dict[str, Any],
        top_k: int = 6,
        candidate_top_n: int = 40,
        fallback_top_n: int = 12,
    ) -> Dict[str, Any]:
        query = build_threat_query(flow_case, threat)
        source_candidates = self.blue_source_candidates(flow_case, threat, candidate_top_n)

        mitigation_candidates: Dict[str, Dict[str, Any]] = {}
        for source_id, source_similarity in source_candidates.items():
            for path in self.mitigation_paths_from_source(source_id):
                mitigation_id = path["mitigation_id"]
                record = mitigation_candidates.setdefault(
                    mitigation_id,
                    {
                        "mitigation_id": mitigation_id,
                        "source_similarity": 0.0,
                        "path_score": 0.0,
                        "max_edge_weight": 1.0,
                        "matched_source_ids": [],
                        "paths": [],
                    },
                )
                record["source_similarity"] = max(record["source_similarity"], source_similarity)
                record["path_score"] = max(record["path_score"], path["path_score"])
                record["max_edge_weight"] = max(
                    record["max_edge_weight"],
                    max(edge_weight(edge) for edge in path["path_edges"]),
                )
                if source_id not in record["matched_source_ids"]:
                    record["matched_source_ids"].append(source_id)
                if len(record["paths"]) < 5:
                    record["paths"].append(path)

        fallback_hits = self.vector_search(query, {"Mitigation"}, top_n=fallback_top_n)
        for hit in fallback_hits:
            mitigation_id = hit["node_id"]
            record = mitigation_candidates.setdefault(
                mitigation_id,
                {
                    "mitigation_id": mitigation_id,
                    "source_similarity": 0.0,
                    "path_score": 0.0,
                    "max_edge_weight": 1.0,
                    "matched_source_ids": [],
                    "paths": [],
                },
            )
            record["fallback_similarity"] = max(record.get("fallback_similarity", 0.0), hit["similarity"])

        scored: List[Dict[str, Any]] = []
        for mitigation_id, record in mitigation_candidates.items():
            mitigation_text = self.node_search_text(self.nodes[mitigation_id])
            mitigation_similarity = max(
                safe_float(record.get("fallback_similarity")),
                token_overlap_score(query, mitigation_text),
            )
            context_fit = token_overlap_score(
                " ".join([get_anchor_step_text(flow_case, threat), threat.get("threat_mechanism", "")]),
                mitigation_text,
            )
            graph_path_score = safe_float(record.get("path_score"))
            edge_weight_score = min(safe_float(record.get("max_edge_weight"), 1.0), 3.0) / 3.0
            score = (
                0.32 * safe_float(record.get("source_similarity"))
                + 0.22 * mitigation_similarity
                + 0.24 * graph_path_score
                + 0.10 * context_fit
                + 0.07 * edge_weight_score
                + self.source_priority(mitigation_id)
            )
            scored.append(
                {
                    **record,
                    "score": score,
                    "score_breakdown": {
                        "source_similarity": round(safe_float(record.get("source_similarity")), 6),
                        "mitigation_similarity": round(mitigation_similarity, 6),
                        "graph_path_score": round(graph_path_score, 6),
                        "context_fit": round(context_fit, 6),
                        "edge_weight_score": round(edge_weight_score, 6),
                        "source_priority": round(self.source_priority(mitigation_id), 6),
                    },
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["mitigation_id"]))
        selected = scored[:top_k]

        return {
            "meta": {
                "retrieval_role": "blue_team",
                "retriever_version": VERSION,
                "generated_at_utc": now_utc(),
                "kg_dir": str(self.kg_dir),
                "use_case_id": flow_case.get("use_case_id"),
                "threat_id": threat.get("threat_id"),
                "top_k": top_k,
                "candidate_top_n": candidate_top_n,
                "fallback_top_n": fallback_top_n,
                "rerank_policy": (
                    "Threat/source node retrieval plus graph-path mitigation expansion, then deterministic "
                    "score over source similarity, mitigation fit, graph path strength, context fit, edge "
                    "weight, and source priority."
                ),
            },
            "query": truncate(query, 1800),
            "threat": {
                "threat_id": threat.get("threat_id"),
                "threat_name": threat.get("threat_name"),
                "anchor_steps": threat.get("anchor_steps") or [],
                "threat_mechanism": threat.get("threat_mechanism"),
                "security_impact": threat.get("security_impact"),
                "source_knowledge_id": threat.get("source_knowledge_id"),
            },
            "items": [self.build_blue_item(item, rank + 1) for rank, item in enumerate(selected)],
        }

    def build_blue_item(self, item: Dict[str, Any], rank: int) -> Dict[str, Any]:
        mitigation_id = item["mitigation_id"]
        matched_sources = item.get("matched_source_ids") or []
        related_node_ids = self.unique_existing([mitigation_id] + matched_sources)
        paths = item.get("paths") or []
        return {
            "rank": rank,
            "score": round(item["score"], 6),
            "score_breakdown": item["score_breakdown"],
            "mitigation": self.node_summary(mitigation_id),
            "matched_sources": [
                self.node_summary(source_id, max_description_chars=420)
                for source_id in matched_sources[:5]
                if source_id in self.nodes
            ],
            "graph_paths": [
                {
                    "path_score": path["path_score"],
                    "edges": self.edge_path_summary(path["path_edges"]),
                }
                for path in paths[:5]
            ],
            "source_evidence": self.collect_evidence(related_node_ids, limit=5),
            "why_relevant": self.explain_blue_relevance(item),
        }

    def explain_blue_relevance(self, item: Dict[str, Any]) -> str:
        if item.get("paths"):
            best_path = max(item["paths"], key=lambda path: path["path_score"])
            if len(best_path["path_edges"]) == 1:
                return "The mitigation is directly connected to a matched risk, attack pattern, or technique by a mitigated_by edge."
            return "The mitigation is connected through a multi-hop graph path from the matched threat context."
        return "The mitigation is included as a semantic fallback because no stronger graph path ranked it out."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rec-EvoGraph-RAG static retrieval v0.2.")
    parser.add_argument("--mode", choices=["red", "blue"], required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--threat-id", default=None, help="For blue mode; defaults to the first threat.")
    parser.add_argument("--kg-dir", type=Path, default=DEFAULT_KG_DIR)
    parser.add_argument("--flow-path", type=Path, default=DEFAULT_FLOW_PATH)
    parser.add_argument("--threat-path", type=Path, default=DEFAULT_THREAT_PATH)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidate-top-n", type=int, default=40)
    parser.add_argument("--fallback-top-n", type=int, default=12)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flows = load_functional_cases(args.flow_path)
    if args.case_id not in flows:
        raise ValueError(f"Unknown case_id={args.case_id!r} in {args.flow_path}")
    flow_case = flows[args.case_id]
    retriever = RecEvoGraphRetriever(args.kg_dir)

    if args.mode == "red":
        result = retriever.retrieve_red(
            flow_case,
            top_k=args.top_k,
            candidate_top_n=args.candidate_top_n,
        )
    else:
        threat_cases = load_threat_cases(args.threat_path)
        if args.case_id not in threat_cases:
            raise ValueError(f"Unknown case_id={args.case_id!r} in {args.threat_path}")
        threat = select_threat(threat_cases[args.case_id], args.threat_id)
        result = retriever.retrieve_blue(
            flow_case,
            threat,
            top_k=args.top_k,
            candidate_top_n=args.candidate_top_n,
            fallback_top_n=args.fallback_top_n,
        )

    if args.output_path:
        write_json(args.output_path, result)
        print(f"[Done] Wrote retrieval output to {args.output_path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
