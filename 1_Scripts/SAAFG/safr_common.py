#!/usr/bin/env python
# -*- coding: utf-8 -*-

import html
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MOJIBAKE_REPLACEMENTS = {
    "鈥檚": "'s",
    "鈥淧": '"P',
    "鈥淲": '"W',
    "鈥淗": '"H',
    "鈥淚": '"I',
    "鈥�": '"',
    "鈥": '"',
    "檚": "s",
    "â€™": "'",
    "â€œ": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "Â ": " ",
    "''": "'",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
}


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
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "via",
    "when",
    "with",
}


KNOWN_PLACEHOLDER_OBJECTS = {
    "Adaptation",
    "Authentication",
    "Availability",
    "Automation",
    "Backend Operations",
    "Compatibility",
    "Completion",
    "Configuration",
    "Consistency",
    "Continuous Learning and Adaptation",
    "Continuity",
    "Connectivity",
    "Data Retrieval",
    "Decision-Making and Transaction Resolution",
    "Dynamic Code Execution",
    "Enablement",
    "Environment Reset and Storage Reclamation",
    "Execution",
    "Link Insertion",
    "LLM interpretation",
    "LLM output",
    "Maintenance",
    "Operational Behavior Adjustment",
    "Operational Responsiveness",
    "Performance",
    "Performance Analytics",
    "Persistent Background Operation",
    "Resolution",
    "Seamless Process Management",
    "Service Availability",
    "System Integration",
    "Task Automation",
    "Termination",
    "Workflow Continuity",
    "Workflow Execution",
    "Workflow Optimization",
}


GENERIC_OBJECT_VERBS = {
    "adapt",
    "adjust",
    "complete",
    "conduct",
    "enable",
    "ensure",
    "execute",
    "facilitate",
    "guide",
    "include",
    "integrate",
    "maintain",
    "perform",
    "pull",
    "receive",
    "resolve",
    "streamline",
    "sustain",
    "terminate",
    "trigger",
}


TRAILING_ADVERBS = {
    "autonomously",
    "automatically",
    "broadly",
    "continuously",
    "directly",
    "dynamically",
    "efficiently",
    "effectively",
    "immediately",
    "independently",
    "reliably",
    "safely",
    "seamlessly",
}


ATLAS_ATTACKER_SIDE_TACTICS = {
    "AI Attack Staging",
    "Reconnaissance",
    "Resource Development",
}


ATLAS_EXCLUDED_TECHNIQUES = {
    "AI Development Workspaces",
    "AI Intellectual Property Theft",
    "Adversarial AI Attack Implementations",
    "Develop Capabilities",
    "Establish Accounts",
    "External Harms",
    "Financial Harm",
    "Generative AI",
    "Pre-Print Repositories",
    "Search Open Technical Databases",
    "Societal Harm",
    "Stage Capabilities",
    "User Harm",
    "Valid Accounts",
}


ATLAS_EXCLUDED_TECHNIQUE_PREFIXES = ("AML.T0048",)


ATLAS_FAMILY_HINTS: Dict[str, Dict[str, object]] = {
    "prompt": {
        "keywords": [
            "prompt",
            "instruction",
            "message",
            "query",
            "input",
            "retrieve",
            "ingest",
            "webpage",
            "website",
            "content",
            "render",
            "response",
        ],
        "preferred_verbs": ["receive", "retrieve", "render", "respond", "generate", "summarize", "ingest"],
        "impact": "Manipulated instructions or context can steer the AI workflow away from its intended behavior.",
    },
    "exec": {
        "keywords": [
            "execute",
            "execution",
            "command",
            "script",
            "tool",
            "terminal",
            "code",
            "shell",
            "plugin",
            "function",
            "invoke",
            "bash",
            "clipboard",
        ],
        "preferred_verbs": ["execute", "invoke", "run", "open", "call", "perform"],
        "impact": "Unsafe code or tool execution can compromise the host environment or connected systems.",
    },
    "secret": {
        "keywords": [
            "credential",
            "secret",
            "token",
            "key",
            "password",
            "configuration",
            "system prompt",
            "environment",
            "ssh",
            "mcp",
        ],
        "preferred_verbs": ["read", "display", "access", "share", "expose", "authenticate"],
        "impact": "Secrets or protected configuration can be exposed to an unauthorized party.",
    },
    "exfil": {
        "keywords": [
            "exfiltration",
            "exfiltrate",
            "outbound",
            "transfer",
            "transmit",
            "upload",
            "email",
            "send",
            "share",
            "distribute",
            "conversation",
            "bcc",
            "data",
        ],
        "preferred_verbs": ["send", "share", "upload", "deliver", "distribute", "transfer"],
        "impact": "Sensitive data can cross the trust boundary and leave the intended workflow.",
    },
    "evasion": {
        "keywords": [
            "evade",
            "evasion",
            "bypass",
            "approval",
            "authenticate",
            "validation",
            "verify",
            "classification",
            "matching",
            "compatibility",
        ],
        "preferred_verbs": ["validate", "verify", "evaluate", "authenticate", "classify", "match", "preserve"],
        "impact": "Malicious inputs can evade screening logic and be treated as legitimate business input.",
    },
    "access": {
        "keywords": [
            "access",
            "visit",
            "browse",
            "open",
            "query",
            "control",
            "interface",
            "document",
            "api",
            "identity",
        ],
        "preferred_verbs": ["visit", "open", "access", "query", "submit", "authenticate"],
        "impact": "Attackers can gain or misuse access through an exposed AI-facing interaction point.",
    },
    "context": {
        "keywords": [
            "context",
            "rag",
            "retrieve",
            "search",
            "knowledge",
            "document",
            "reference",
            "email",
            "citation",
        ],
        "preferred_verbs": ["retrieve", "search", "reference", "index", "store", "load"],
        "impact": "Untrusted retrieved content can poison the context the model relies on.",
    },
    "poison": {
        "keywords": [
            "poison",
            "training",
            "fine-tune",
            "model",
            "dataset",
            "artifact",
            "index",
            "update",
            "learn",
            "upload",
            "submission",
            "repository",
            "publish",
            "registry",
            "redistribution",
            "modify",
            "modification",
            "knowledge",
            "tuning",
        ],
        "preferred_verbs": ["train", "learn", "update", "index", "retrieve", "store", "upload", "submit", "publish", "modify"],
        "impact": "Compromised artifacts or data can degrade model integrity and downstream outputs.",
    },
    "destructive": {
        "keywords": [
            "delete",
            "truncate",
            "destroy",
            "wipe",
            "reset",
            "terminate",
            "harm",
            "denial",
        ],
        "preferred_verbs": ["delete", "truncate", "reset", "terminate"],
        "impact": "The workflow can trigger destructive or availability-impacting behavior.",
    },
    "generic": {
        "keywords": ["system", "agent", "model", "service", "workflow"],
        "preferred_verbs": ["perform", "process", "support", "use"],
        "impact": "The AI-enabled workflow can be misused or compromised.",
    },
}


RETRYABLE_PREVIOUS_VERBS = {
    "access",
    "authenticate",
    "configure",
    "enter",
    "input",
    "load",
    "open",
    "provide",
    "query",
    "read",
    "receive",
    "request",
    "retrieve",
    "search",
    "select",
    "submit",
    "upload",
    "visit",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def first_sentence(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[\.\!\?])\s+", cleaned)
    return parts[0].strip()


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def strip_markdown_links(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[<sup>.*?</sup>\]\[[^\]]+\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[0-9]+\]:\s*https?://\S+(?:\s+\"[^\"]+\")?", " ", text)
    return text


def clean_source_text(value: Any) -> str:
    text = str(value or "")
    text = html.unescape(text)
    text = text.replace("\\&lt;", "<").replace("\\&gt;", ">")
    text = unicodedata.normalize("NFKC", text)
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = strip_markdown_links(text)
    text = re.sub(r"\{\{\s*create_internal_link\(([^)]+)\)\s*\}\}", r"\1", text)
    text = re.sub(r"</?(?:div|span|code|pre|sup|br)[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("```", " ")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("`", "")
    text = re.sub(r"'([A-Za-z_]+)skill\b", r"'\1' skill", text)
    text = re.sub(r"'([A-Za-z_]+)Skill\b", r"'\1' Skill", text)
    text = text.replace("**", " ")
    text = text.replace("__", " ")
    text = text.replace("[dot]", ".")
    text = re.sub(r"\s*[-+]\s+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def unique_tokens(texts: Sequence[str], min_len: int = 4) -> List[str]:
    tokens: List[str] = []
    for text in texts:
        for token in tokenize(text):
            if len(token) < min_len or token in STOPWORDS:
                continue
            tokens.append(token)
    return dedupe_preserve_order(tokens)


def verb_forms(verb: str) -> List[str]:
    root = normalize_text(verb).lower()
    if not root:
        return []
    forms = {root}
    if root.endswith("e"):
        forms.add(root + "d")
        forms.add(root[:-1] + "ing")
    else:
        forms.add(root + "ed")
        forms.add(root + "ing")
    if root.endswith(("s", "x", "z", "ch", "sh", "o")):
        forms.add(root + "es")
    elif root.endswith("y") and len(root) > 1 and root[-2] not in "aeiou":
        forms.add(root[:-1] + "ies")
    else:
        forms.add(root + "s")
    return sorted(forms, key=len, reverse=True)


def drop_trailing_adverbs(phrase: str) -> str:
    words = normalize_text(phrase).split()
    while words and words[-1].lower().rstrip(".,") in TRAILING_ADVERBS:
        words.pop()
    return " ".join(words)


def extract_object_from_sentence(sentence: str, subject: str, verb: str) -> str:
    cleaned_sentence = normalize_text(sentence).rstrip(".")
    cleaned_subject = normalize_text(subject)
    cleaned_verb = normalize_text(verb)
    if not cleaned_sentence or not cleaned_verb:
        return ""

    remainder = cleaned_sentence
    lower_sentence = cleaned_sentence.lower()
    lower_subject = cleaned_subject.lower()
    for prefix in (
        lower_subject,
        f"the {lower_subject}",
        f"a {lower_subject}",
        f"an {lower_subject}",
    ):
        if prefix and lower_sentence.startswith(prefix):
            remainder = cleaned_sentence[len(prefix):].lstrip(" ,")
            break

    forms = "|".join(re.escape(form) for form in verb_forms(cleaned_verb))
    passive_forms = "|".join(
        re.escape(form) for form in [form for form in verb_forms(cleaned_verb) if form.endswith("ed")]
    )
    patterns = [
        rf"^(?:must|shall|can|could|should|would|may|might|will)\s+({forms})\b\s*(.*)$",
        rf"^(?:must|shall|can|could|should|would|may|might|will)\s+(?:be|been|being)\s+({passive_forms})\b\s*(.*)$" if passive_forms else "",
        rf"^(?:is|are|was|were|be|been|being)\s+({passive_forms})\b\s*(.*)$" if passive_forms else "",
        rf"^({forms})\b\s*(.*)$",
    ]
    for pattern in patterns:
        if not pattern:
            continue
        match = re.match(pattern, remainder, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = drop_trailing_adverbs(match.group(2).strip(" .,:;"))
        return candidate
    return ""


def is_placeholder_object(object_text: str, verb: str = "") -> bool:
    cleaned = normalize_text(object_text)
    if not cleaned:
        return True
    if cleaned in KNOWN_PLACEHOLDER_OBJECTS:
        return True
    if normalize_text(verb).lower() in GENERIC_OBJECT_VERBS and cleaned[:1].isupper():
        lowered = cleaned.lower()
        if any(
            marker in lowered
            for marker in (
                "adjustment",
                "authentication",
                "automation",
                "availability",
                "compatibility",
                "completion",
                "configuration",
                "consistency",
                "continuity",
                "execution",
                "maintenance",
                "performance",
                "resolution",
                "responsiveness",
                "support",
                "termination",
                "workflow",
            )
        ):
            return True
    return False


def recover_object(original_object: str, sentence: str, subject: str, verb: str, requirement_text: str) -> str:
    original = normalize_text(original_object)
    recovered = extract_object_from_sentence(sentence, subject, verb)
    if (not recovered or recovered.lower() == original.lower()) and requirement_text:
        recovered = extract_object_from_sentence(requirement_text, subject, verb) or recovered
    recovered = normalize_text(recovered)
    if not recovered:
        return original
    if not original:
        return recovered
    if original.lower() == recovered.lower():
        return original
    if is_placeholder_object(original, verb):
        return recovered
    if len(tokenize(recovered)) > len(tokenize(original)) and any(token.islower() for token in recovered.split()):
        return recovered
    return original


def clean_flow_step(step: Dict[str, Any], requirement_text: str, bf_step_id: str, flow_from: Optional[str]) -> Dict[str, Optional[str]]:
    subject = normalize_text(step.get("subject"))
    verb = normalize_text(step.get("verb"))
    sentence = normalize_text(step.get("step_sentence"))
    cleaned_object = recover_object(step.get("object"), sentence, subject, verb, requirement_text)

    if verb.lower() == "enable" and re.search(r"\bis enabled\b", sentence.lower()) and cleaned_object:
        infinitive = cleaned_object[3:] if cleaned_object.lower().startswith("to ") else cleaned_object
        if len(tokenize(infinitive)) > 12:
            infinitive = "carry out the configured automated operations"
        sentence = f"The system enables the {subject} to {infinitive}."
        cleaned_object = f"{subject} to {infinitive}"
        subject = "system"

    return {
        "step_id": bf_step_id,
        "source_step_id": str(step.get("step_id")),
        "step_sentence": normalize_text(sentence),
        "subject": subject,
        "verb": verb,
        "object": normalize_text(cleaned_object),
        "flow_from": flow_from,
    }


def atlas_family_for_text(*texts: str, tactic_name: str = "") -> str:
    haystack = " ".join(texts + (tactic_name,)).lower()
    if any(keyword in haystack for keyword in ("exfiltration", "exfiltrate", "upload", "bcc", "conversation", "outbound", "email contents")):
        return "exfil"
    if any(keyword in haystack for keyword in ("credential", "secret", "token", "password", "key", "system prompt", "ssh", "mcp.json", "environment variable")):
        return "secret"
    if any(keyword in haystack for keyword in ("command", "script", "shell", "execute", "execution", "bash", "tool invocation", "plugin")):
        return "exec"
    if tactic_name == "Defense Evasion" or any(keyword in haystack for keyword in ("evade", "evasion", "bypass", "approval", "authenticity", "matching")):
        return "evasion"
    if tactic_name in {"Initial Access", "AI Model Access"} or any(keyword in haystack for keyword in ("access", "visit", "browse", "query", "authenticate", "control interface")):
        return "access"
    if any(keyword in haystack for keyword in ("poison", "dataset", "model editing", "fine-tune", "index", "training")):
        return "poison"
    if tactic_name == "Impact" or any(keyword in haystack for keyword in ("delete", "truncate", "destroy", "wipe", "denial of service", "harm")):
        return "destructive"
    if any(keyword in haystack for keyword in ("retrieve", "retrieved", "rag", "citation", "reference", "search", "emailmessage", "context")):
        return "context"
    if any(keyword in haystack for keyword in ("prompt", "jailbreak", "injection", "infiltration", "reasoning")):
        return "prompt"
    return "generic"


def atlas_candidate_allowed(technique_id: str, technique_name: str, tactic_name: str) -> bool:
    if tactic_name in ATLAS_ATTACKER_SIDE_TACTICS:
        return False
    if technique_name in ATLAS_EXCLUDED_TECHNIQUES:
        return False
    return not any(technique_id.startswith(prefix) for prefix in ATLAS_EXCLUDED_TECHNIQUE_PREFIXES)


def atlas_hint_profile(family: str) -> Dict[str, object]:
    return ATLAS_FAMILY_HINTS.get(family, ATLAS_FAMILY_HINTS["generic"])


def score_step_for_keywords(step: Dict[str, str], keywords: Sequence[str], preferred_verbs: Sequence[str]) -> int:
    sentence = (step.get("step_sentence") or "").lower()
    subject = (step.get("subject") or "").lower()
    verb = (step.get("verb") or "").lower()
    object_text = (step.get("object") or "").lower()
    sentence_tokens = set(tokenize(sentence))
    subject_tokens = set(tokenize(subject))
    verb_tokens = set(tokenize(verb))
    object_tokens = set(tokenize(object_text))
    score = 0
    for keyword in dedupe_preserve_order(keyword.lower() for keyword in keywords if keyword):
        if " " in keyword:
            if keyword in sentence:
                score += 2
            if keyword in object_text:
                score += 2
            if keyword in subject:
                score += 1
            if keyword == verb:
                score += 1
            continue
        if keyword in sentence_tokens:
            score += 2
        if keyword in object_tokens:
            score += 2
        if keyword in subject_tokens:
            score += 1
        if keyword in verb_tokens:
            score += 1
    if verb in {item.lower() for item in preferred_verbs}:
        score += 2
    return score


def choose_retry_target(flow_steps: Sequence[Dict[str, str]], anchor_step_id: str) -> Tuple[str, Optional[str]]:
    if not flow_steps:
        return ("terminate", None)
    anchor_index = next(
        (idx for idx, step in enumerate(flow_steps) if step.get("step_id") == anchor_step_id),
        -1,
    )
    if anchor_index < 0:
        return ("terminate", None)
    if anchor_index < len(flow_steps) - 1:
        return ("return", anchor_step_id)
    for previous_step in reversed(flow_steps[:anchor_index]):
        if (previous_step.get("verb") or "").lower() in RETRYABLE_PREVIOUS_VERBS:
            return ("return", previous_step["step_id"])
    return ("terminate", None)
