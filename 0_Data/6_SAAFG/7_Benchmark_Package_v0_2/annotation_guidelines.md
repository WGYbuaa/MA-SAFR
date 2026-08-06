# SAAFG Annotation Guideline v0.2

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
