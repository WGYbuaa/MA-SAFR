# Rec-EvoGraph-RAG Static KG Build Report

- graph_name: Rec-EvoGraph-RAG static security knowledge graph
- version: v0.2
- generated_at_utc: 2026-05-22T11:52:59.905507Z
- output_dir: `0_Data\5_Knowledge_Base\recevograph_rag`

## Leakage Policy

Built only from OWASP/ATLAS source knowledge files. SAAFG silver threats/defenses are not ingested into this static graph.

## Source Files

- OWASP: `0_Data\5_Knowledge_Base\source\owasp_knowledge.json`
- ATLAS: `0_Data\5_Knowledge_Base\source\mitre_atlas_knowledge.json`

## Graph Size

- nodes: 1576
- edges: 8613

## Node Type Counts

- AttackPattern: 747
- Mitigation: 113
- Risk: 26
- SourceEvidence: 547
- Technique: 143

## Edge Relation Counts

- belongs_to: 152
- exploits: 747
- implements_or_examples: 447
- mitigated_by: 3852
- supported_by: 3415

## Edge Source Dataset Counts

- atlas: 4374
- owasp: 4239

## Ingestion Stats

- owasp_items: 100
- owasp_items_with_mitigation_sections: 100
- owasp_mitigation_section_total: 865
- atlas_items: 57
- atlas_procedure_steps_total: 447
- atlas_procedure_steps_with_mitigations: 225
- atlas_procedure_steps_without_mitigations: 222
- atlas_empty_mitigation_step_rate: 0.496644
- atlas_mitigation_reference_total: 813

## Validation

- missing_edge_endpoint_count: 0
- isolated_node_count: 0
- attack_patterns_without_mitigation_count: 222
- risks_without_mitigation_count: 1

## Sample Prompt Injection Paths

- LLM01:2025 Prompt Injection -> Constrain model behavior (weight=1.0, sources=LLM01_PromptInjection)
- LLM01:2025 Prompt Injection -> Define and validate expected output formats (weight=1.0, sources=LLM01_PromptInjection)
- LLM01:2025 Prompt Injection -> Implement input and output filtering (weight=1.0, sources=LLM01_PromptInjection)
- LLM01:2025 Prompt Injection -> Enforce privilege control and least privilege access (weight=1.0, sources=LLM01_PromptInjection)
- LLM01:2025 Prompt Injection -> Require human approval for high-risk actions (weight=1.0, sources=LLM01_PromptInjection)
- LLM01:2025 Prompt Injection -> Segregate and identify external content (weight=1.0, sources=LLM01_PromptInjection)
- LLM01:2025 Prompt Injection -> Conduct adversarial testing and attack simulations (weight=1.0, sources=LLM01_PromptInjection)

## Warnings

- None
