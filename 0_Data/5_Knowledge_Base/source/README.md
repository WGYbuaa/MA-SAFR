# Knowledge Base README

本目录用于存放面向 RAG / Knowledge Base 构建的结构化数据文件。这里的文件不是原始数据，而是由 `0_Data/2_Red_team_scenario` 和 `0_Data/1_Processed_data` 以及部分 `0_Data/0_Raw_data` 中的源文件逐步整理、映射、扩充后生成的知识库版本。以后主要使用 `owasp_knowledge.json`和 `mitre_atlas_knowledge.json`。

## 文件概览

当前目录包含以下文件：

- `owasp_knowledge.json (后期制作知识库要用`)
- `atlas_requirement_with_procedure_v5.5.0.json`
- `mitre_atlas_knowledge.json`(后期制作知识库要用`)`

---

## 使用建议

如果面向 RAG 使用：

- `owasp_knowledge.json` 适合做 OWASP requirement 与 mitigation 联合检索
- `atlas_requirement_with_procedure_v5.5.0.json` 适合作为基础 requirement + case study 数据
- `mitre_atlas_knowledge.json` 最适合作为当前主版本的深度检索数据，因为它同时包含：
  - requirement 文本
  - case study procedure
  - tactic 结构化信息
  - technique 结构化信息
  - mitigation 映射

---

## 注意事项

- 本目录中的文件均为派生文件，建议不要将其视为唯一事实来源。
- 如果上游 OWASP / MITRE ATLAS 原始数据更新，需要重新执行映射和扩充流程。
- 对于 `mitigations = []` 的 technique，不能直接断定“没有缓解方案”，更合理的解释通常是“官方数据未显式给出映射”。
