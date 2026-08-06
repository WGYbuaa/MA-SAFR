# SAAFG-Bench v0.2

## Summary

This package provides the current v0.2 freeze for Security-Augmented Alternative Flow Generation (SAAFG).

It contains:

- a case registry
- split files
- cleaned functional flow inputs
- source-grounded silver threat records
- source-grounded silver security-augmented flows
- schema and protocol documents
- author-verified human-check artifacts under `0_Data/6_SAAFG/5_Gold_or_Human_Check`

## Counts

- total cases: 157
- train cases: 80
- dev cases: 20
- test cases: 57
- empty-core-threat cases: 0
- author-verified gold subset cases: 60
- legacy AI-reviewed gold subset seed cases: 12

## Human-Check Files

- `author_verified_subset.json`: current round revised-case subset with reviewed flows, threat records, and security-augmented flows.
- `author_verified_subset_notes.md`: scope and labeling notes for the expert-verified subset.
- `optional_anchor_adjudication.json`: ambiguity sidecar for acceptable alternate anchors that do not change the canonical benchmark anchor.
- `saafg_ai_reviewed_gold_subset_seed_v0_1.json`: legacy AI-reviewed seed subset retained for historical comparison.

## Important Note

The v0.2 author-verified subset is suitable for internal evaluation, ablation, and focused human-check workflows.
It should be described as author-verified rather than independent third-party expert gold unless additional external human confirmation is added later.
