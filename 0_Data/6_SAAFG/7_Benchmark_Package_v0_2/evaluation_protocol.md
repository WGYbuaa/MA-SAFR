# SAAFG Evaluation Protocol v0.2

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
