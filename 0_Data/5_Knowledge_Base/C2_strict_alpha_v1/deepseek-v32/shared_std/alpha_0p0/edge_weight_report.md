# Rec-EvoGraph-RAG Feedback Weight Alpha KG Report

- graph_name: Rec-EvoGraph-RAG feedback-alpha security knowledge graph
- version: v0.2-feedback-alpha
- generated_at_utc: 2026-07-19T13:08:54.696001Z
- feedback_weight_alpha: 0.0
- base_graph_dir: `0_Data\5_Knowledge_Base\recevograph_rag`
- feedback_jsonl: `0_Data\5_Knowledge_Base\recevograph_rag_feedback\v0_2\feedback_events.jsonl`
- output_dir: `0_Data\5_Knowledge_Base\C2_strict_alpha_v1\deepseek-v32\shared_std\alpha_0p0`

## Leakage Policy

Node text and graph topology are unchanged from the static OWASP/ATLAS graph. Only edge weights are changed by split-filtered critic feedback and feedback_weight_alpha linear fusion.

## Feedback Usage

- input_event_count: 1413
- used_edge_count: 897
- skipped_split: 0
- skipped_no_edges: 0
- skipped_low_confidence: 0
- missing_edge_key_count: 0

## Alpha Fusion Policy

- formula: `feedback_weight = clip(base_weight * exp(eta * clipped_signal), min_weight, max_weight); fused_weight = clip((1 - feedback_weight_alpha) * base_weight + feedback_weight_alpha * feedback_weight, min_weight, max_weight)`
- feedback_weight_alpha: 0.0
- eta: 0.08
- min_weight: 0.35
- max_weight: 3.0
- max_abs_signal: 5.0
- negative_scale: 0.7
- split_filter: train

## Edge Weight Changes

- edge_count: 8613
- feedback_eligible_edge_count: 897
- updated_edge_count: 0
- increased_edge_count: 0
- decreased_edge_count: 0
- unchanged_after_alpha_fusion_count: 897
- no_feedback_edge_count: 7716
- max_abs_feedback_delta: 0.491825
- mean_abs_feedback_delta: 0.028925
- max_abs_alpha_delta: 0.000000
- mean_abs_alpha_delta: 0.000000

## Top Absolute Alpha Deltas

- edge::mitigated_by::2fbf0f222a10: mitigated_by base=1.000000 feedback=1.491825 fused=1.000000 alpha_delta=0.000000 events=87 source=risk::owasp::llm03_supplychain target=mitigation::only_use_models_from_verifiable_sources_and_use_third_party_model_int_afa7804934
- edge::mitigated_by::1535a7937022: mitigated_by base=1.000000 feedback=1.491825 fused=1.000000 alpha_delta=0.000000 events=84 source=risk::owasp::llm03_supplychain target=mitigation::implement_a_patching_policy_to_mitigate_vulnerable_or_outdated_compon_1984a65bab
- edge::mitigated_by::e61ad98876ee: mitigated_by base=1.000000 feedback=1.491825 fused=1.000000 alpha_delta=0.000000 events=29 source=risk::atlas_tactic::aml_ta0005 target=mitigation::agent_config_tools
- edge::mitigated_by::83df818dfb58: mitigated_by base=1.000000 feedback=1.491825 fused=1.000000 alpha_delta=0.000000 events=18 source=risk::owasp::llm04_datamodelpoisoning target=mitigation::monitor_training_loss_and_analyze_model_behavior_for_signs_of_poisoni_c4280cb595
- edge::mitigated_by::53bf0e1b244e: mitigated_by base=1.000000 feedback=1.491825 fused=1.000000 alpha_delta=0.000000 events=13 source=risk::atlas_tactic::aml_ta0010 target=mitigation::agent_config_tools
- edge::mitigated_by::580990682c28: mitigated_by base=1.000000 feedback=1.485041 fused=1.000000 alpha_delta=0.000000 events=24 source=risk::owasp::llm04_datamodelpoisoning target=mitigation::store_user_supplied_information_in_a_vector_database_allowing_adjustm_e7c478b4d1
- edge::mitigated_by::a781db4a730b: mitigated_by base=1.000000 feedback=1.411421 fused=1.000000 alpha_delta=0.000000 events=18 source=risk::owasp::llm03_supplychain target=mitigation::carefully_vet_data_sources_and_suppliers_including_t_and_cs_and_their_b356b9bf84
- edge::mitigated_by::486dca8e2dfe: mitigated_by base=1.000000 feedback=1.349892 fused=1.000000 alpha_delta=0.000000 events=18 source=risk::atlas_tactic::aml_ta0005 target=mitigation::agent_config_priv
- edge::mitigated_by::dd43379bfa67: mitigated_by base=1.000000 feedback=1.332821 fused=1.000000 alpha_delta=0.000000 events=36 source=risk::owasp::llm03_supplychain target=mitigation::anomaly_detection_and_adversarial_robustness_tests_on_supplied_models_d5c412d9da
- edge::mitigated_by::510ece273e8d: mitigated_by base=1.000000 feedback=1.332578 fused=1.000000 alpha_delta=0.000000 events=15 source=risk::owasp::llm03_supplychain target=mitigation::apply_comprehensive_ai_red_teaming_and_evaluations_when_selecting_a_t_9f423eb2ee
- edge::mitigated_by::0d60edbe1875: mitigated_by base=1.000000 feedback=1.271835 fused=1.000000 alpha_delta=0.000000 events=11 source=attack_pattern::owasp::llm04_datamodelpoisoning::training_data_integrity::40 target=mitigation::store_user_supplied_information_in_a_vector_database_allowing_adjustm_e7c478b4d1
- edge::mitigated_by::00a77461227e: mitigated_by base=1.000000 feedback=1.251327 fused=1.000000 alpha_delta=0.000000 events=12 source=attack_pattern::owasp::llm03_supplychain::external_data_source_validation::37 target=mitigation::only_use_models_from_verifiable_sources_and_use_third_party_model_int_afa7804934
- edge::exploits::adc3d69484ad: exploits base=1.000000 feedback=1.245911 fused=1.000000 alpha_delta=0.000000 events=20 source=attack_pattern::owasp::llm03_supplychain::external_data_source_validation::37 target=risk::owasp::llm03_supplychain
- edge::mitigated_by::525e36b30243: mitigated_by base=1.000000 feedback=1.236247 fused=1.000000 alpha_delta=0.000000 events=12 source=risk::owasp::llm03_supplychain target=mitigation::encrypt_models_deployed_at_ai_edge_with_integrity_checks_and_use_vend_ae81320183
- edge::exploits::559a22ced7ce: exploits base=1.000000 feedback=1.232794 fused=1.000000 alpha_delta=0.000000 events=21 source=attack_pattern::owasp::llm04_datamodelpoisoning::training_data_integrity::40 target=risk::owasp::llm04_datamodelpoisoning
- edge::mitigated_by::40f3dbe079a3: mitigated_by base=1.000000 feedback=1.219911 fused=1.000000 alpha_delta=0.000000 events=11 source=attack_pattern::owasp::llm03_supplychain::external_data_source_validation::37 target=mitigation::anomaly_detection_and_adversarial_robustness_tests_on_supplied_models_d5c412d9da
- edge::mitigated_by::40fb9f4276b3: mitigated_by base=1.000000 feedback=1.212402 fused=1.000000 alpha_delta=0.000000 events=22 source=risk::owasp::llm04_datamodelpoisoning target=mitigation::ensure_sufficient_infrastructure_controls_to_prevent_the_model_from_a_f5f03c7975
- edge::mitigated_by::dc0929e06d3d: mitigated_by base=1.000000 feedback=1.206118 fused=1.000000 alpha_delta=0.000000 events=9 source=risk::owasp::llm04_datamodelpoisoning target=mitigation::vet_data_vendors_rigorously_and_validate_model_outputs_against_truste_cdb24e164c
- edge::mitigated_by::60d9ed842c36: mitigated_by base=1.000000 feedback=1.205070 fused=1.000000 alpha_delta=0.000000 events=9 source=risk::atlas_tactic::aml_ta0005 target=mitigation::hitl_agent_actions
- edge::mitigated_by::b2ab32ed5de3: mitigated_by base=1.000000 feedback=1.197217 fused=1.000000 alpha_delta=0.000000 events=3 source=attack_pattern::owasp::llm03_supplychain::dynamic_code_execution::23 target=mitigation::implement_a_patching_policy_to_mitigate_vulnerable_or_outdated_compon_1984a65bab
- edge::mitigated_by::6a66391cdfb5: mitigated_by base=1.000000 feedback=1.193116 fused=1.000000 alpha_delta=0.000000 events=4 source=attack_pattern::owasp::llm04_datamodelpoisoning::training_data_integrity::40 target=mitigation::monitor_training_loss_and_analyze_model_behavior_for_signs_of_poisoni_c4280cb595
- edge::mitigated_by::1f6c802be205: mitigated_by base=1.000000 feedback=1.191898 fused=1.000000 alpha_delta=0.000000 events=5 source=attack_pattern::owasp::llm03_supplychain::external_dependency_trust::30 target=mitigation::only_use_models_from_verifiable_sources_and_use_third_party_model_int_afa7804934
- edge::exploits::8f23535a9b3f: exploits base=1.000000 feedback=1.185756 fused=1.000000 alpha_delta=0.000000 events=10 source=attack_pattern::atlas::aml_cs0047::step_005::aml_t0103 target=risk::atlas_tactic::aml_ta0005
- edge::mitigated_by::218a4009d129: mitigated_by base=1.000000 feedback=1.182968 fused=1.000000 alpha_delta=0.000000 events=5 source=risk::atlas_tactic::aml_ta0003 target=mitigation::verify_ml_artifacts
- edge::mitigated_by::5fe614707df3: mitigated_by base=1.000000 feedback=1.182064 fused=1.000000 alpha_delta=0.000000 events=9 source=risk::owasp::llm04_datamodelpoisoning target=mitigation::implement_strict_sandboxing_to_limit_model_exposure_to_unverified_dat_6b3392c1d8
- edge::mitigated_by::7b3044e8f01e: mitigated_by base=1.000000 feedback=1.180357 fused=1.000000 alpha_delta=0.000000 events=11 source=risk::atlas_tactic::aml_ta0006 target=mitigation::gen_ai_guidelines
- edge::mitigated_by::e96bcdc414bd: mitigated_by base=1.000000 feedback=1.173511 fused=1.000000 alpha_delta=0.000000 events=2 source=attack_pattern::owasp::llm04_datamodelpoisoning::external_data_injection::39 target=mitigation::monitor_training_loss_and_analyze_model_behavior_for_signs_of_poisoni_c4280cb595
- edge::mitigated_by::579088625829: mitigated_by base=1.000000 feedback=1.169277 fused=1.000000 alpha_delta=0.000000 events=16 source=risk::owasp::llm08_vectorandembeddingweaknesses target=mitigation::data_validation_and_source_authentication
- edge::exploits::300f4f3a18c6: exploits base=1.000000 feedback=1.154519 fused=1.000000 alpha_delta=0.000000 events=8 source=attack_pattern::atlas::aml_cs0048::step_009::aml_t0025 target=risk::atlas_tactic::aml_ta0010
- edge::mitigated_by::1ddd9a0de389: mitigated_by base=1.000000 feedback=1.150180 fused=1.000000 alpha_delta=0.000000 events=15 source=risk::owasp::llm09_misinformation target=mitigation::training_and_education
