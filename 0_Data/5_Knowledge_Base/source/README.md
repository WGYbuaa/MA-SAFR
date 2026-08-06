# Knowledge Base README

本目录用于存放面向 RAG / Knowledge Base 构建的结构化数据文件。这里的文件不是原始数据，而是由 `0_Data/2_Red_team_scenario` 和 `0_Data/1_Processed_data` 以及部分 `0_Data/0_Raw_data` 中的源文件逐步整理、映射、扩充后生成的知识库版本。以后主要使用 `owasp_knowledge.json`和 `mitre_atlas_knowledge.json`，但是后者有50.34%（225/447）的technique没有对应的mitigations；`atlas_requirement_with_procedure_tactic_nested_with_mitigations_v5.4.0.json`的这项数据是47.88%（192/401）。

## 文件概览

当前目录包含以下文件：

- `owasp_knowledge.json (后期制作知识库要用`)
- `atlas_requirement_with_procedure_v5.4.0.json`
- `atlas_requirement_with_procedure_tactic_nested_v5.4.0.json`
- `atlas_requirement_with_procedure_tactic_nested_with_mitigations_v5.4.0.json`
- `atlas_requirement_with_procedure_v5.5.0.json`
- `mitre_atlas_knowledge.json`(后期制作知识库要用`)`

---

## 1. owasp_knowledge.json

### 生成来源

- 需求文本来源：
  `0_Data/2_Red_team_scenario/2_unstructured_requirement_text.json`
- OWASP 结构化来源：
  `0_Data/1_Processed_data/0_OWASP.json`

### 生成过程

1. 从 `2_unstructured_requirement_text.json` 中筛选出 `dataset = "owasp"` 的所有项。
2. 使用每个项的 `id`，在 `0_OWASP.json -> entries` 中找到对应的 OWASP 条目。
3. 读取对应条目下：
   `sections -> Prevention and Mitigation Strategies -> subsections`
4. 将该内容写回对应 requirement 项中，形成新的知识库条目。
5. 同时补充来源信息，便于追溯原始出处。

### 文件内容

文件为一个 JSON 列表，每个元素对应一条 OWASP requirement 数据，主要字段包括：

- `global_id`
- `dataset`
- `id`
- `item_index`
- `original_text`
- `requirement_text`
- `business_value`
- `implicit_risk_hints`
- `source_title`
- `source_relative_path`
- `prevention_and_mitigation_strategies_subsections`

其中：

- `prevention_and_mitigation_strategies_subsections` 是一个字典，key 为 mitigation 小节标题，value 为对应描述。

---

## 2. atlas_requirement_with_procedure_v5.4.0.json

### 生成来源

- requirement 来源：
  `0_Data/2_Red_team_scenario/2_unstructured_requirement_text.json`
- ATLAS 处理后来源：
  `0_Data/1_Processed_data/1_ATLAS.json`

### 生成过程

1. 从 `2_unstructured_requirement_text.json` 中筛选出 `dataset = "atlas"` 的所有项。
2. 使用每个项的 `id`，在 `1_ATLAS.json` 中找到对应 case study。
3. 将对应 case study 的 `procedure` 内容补充到 requirement 项中。
4. 同时补充 `source_name` 和 `source_summary`，用于保存来源名称与摘要。

### 文件内容

文件为一个 JSON 列表，每个元素对应一条 ATLAS requirement 数据，主要字段包括：

- `global_id`
- `dataset`
- `id`
- `item_index`
- `original_text`
- `requirement_text`
- `business_value`
- `implicit_risk_hints`
- `source_name`
- `source_summary`
- `procedure`

其中：

- `procedure` 是一个列表，每个元素表示 case study 中的一个 procedure step。
- 在这个阶段，`procedure[].tactic` 和 `procedure[].technique` 仍然是原始引用字符串，例如 `{{reconnaissance.id}}`。

---

## 3. atlas_requirement_with_procedure_tactic_nested_v5.4.0.json

### 生成来源

- 上游文件：
  `0_Data/5_Knowledge_Base/atlas_requirement_with_procedure_v5.4.0.json`
- tactic 原始来源：
  `0_Data/0_Raw_data/1_MITRE ATLAS/atlas-data-main/data/tactics.yaml`

### 生成过程

1. 读取 `atlas_requirement_with_procedure_v5.4.0.json`。
2. 遍历每个 `procedure` step 中的 `tactic`，原始格式为 `{{reconnaissance.id}}`。
3. 提取其中的 anchor 名，例如 `reconnaissance`。
4. 在 `tactics.yaml` 中通过 YAML anchor（例如 `&reconnaissance`）找到对应 tactic。
5. 将 `tactic` 从字符串扩充为字典。

### 文件内容

相较于上游文件，本文件将 `procedure[].tactic` 改为了嵌套字典，结构如下：

- `raw`
- `anchor`
- `id`
- `name`
- `description`

`procedure[].technique` 在这个阶段仍然保持原始字符串形式。

---

## 4. atlas_requirement_with_procedure_tactic_nested_with_mitigations_v5.4.0.json

### 生成来源

- 上游文件：
  `0_Data/5_Knowledge_Base/atlas_requirement_with_procedure_tactic_nested_v5.4.0.json`
- mitigation 原始来源：
  `0_Data/0_Raw_data/1_MITRE ATLAS/atlas-data-main/data/mitigations.yaml`

### 生成过程

1. 读取 `atlas_requirement_with_procedure_tactic_nested_v5.4.0.json`。
2. 对每个 `procedure` step 的 `technique` 原始字符串做精确匹配。
3. 在 `mitigations.yaml` 中查找 `techniques[].id == procedure[].technique` 的所有 mitigation 项。
4. 对每个命中的 mitigation，取：
   - 顶层 YAML anchor 作为 key
   - 对应 technique 条目下的 `use` 作为 value
5. 将这些结果写入 `procedure[].mitigations` 列表。

### 文件内容

相较于上游文件，本文件新增：

- `procedure[].mitigations`

其结构为列表，列表示例：

```json
[
  {
    "limit_model_release": "Limiting the release of datasets can reduce an adversary's ability to target production models trained on the same or similar data."
  }
]
```

说明：

- 如果某个 technique 在官方 `mitigations.yaml` 中没有对应项，则 `mitigations` 为空列表 `[]`。
- 空列表不一定表示整理错误，也可能是官方数据未提供 mitigation 映射。

---

## 5. atlas_requirement_with_procedure_v5.5.0.json

### 生成来源

- requirement 来源：
  `0_Data/2_Red_team_scenario/2_unstructured_requirement_text_atlas_v5.5.0.json`
- ATLAS 处理后来源：
  `0_Data/1_Processed_data/2_ATLAS_v5.5.0.json`

### 生成过程

1. 先基于 `1_atlas_Red_team_scenario_v5.5.0.json` 生成
   `2_unstructured_requirement_text_atlas_v5.5.0.json`：
   - 新增从 0 开始的 `global_id`
   - 新增固定值 `dataset = "atlas"`
   - 将 `response_content` 中的 JSON 字符串解析为：
     - `requirement_text`
     - `business_value`
     - `implicit_risk_hints`
2. 再使用每个 requirement 项的 `id`，在 `2_ATLAS_v5.5.0.json` 中找到对应 case study。
3. 将 `procedure`、`source_name`、`source_summary` 补充到 requirement 项中。

### 文件内容

本文件是 v5.5.0 版本的基础 ATLAS knowledge base，字段与 v5.4.0 基础版类似：

- `global_id`
- `dataset`
- `id`
- `item_index`
- `original_text`
- `requirement_text`
- `business_value`
- `implicit_risk_hints`
- `source_name`
- `source_summary`
- `procedure`

在这个阶段：

- `procedure[].tactic` 仍为原始字符串
- `procedure[].technique` 仍为原始字符串

---

## 6. mitre_atlas_knowledge.json

### 生成来源

- 上游文件：
  `0_Data/5_Knowledge_Base/atlas_requirement_with_procedure_v5.5.0.json`
- tactic 来源：
  `0_Data/0_Raw_data/1_MITRE ATLAS/atlas-data-main_v5.5.0/atlas-data-main/data/tactics.yaml`
- technique 来源：
  `0_Data/0_Raw_data/1_MITRE ATLAS/atlas-data-main_v5.5.0/atlas-data-main/data/techniques.yaml`
- mitigation 来源：
  `0_Data/0_Raw_data/1_MITRE ATLAS/atlas-data-main_v5.5.0/atlas-data-main/data/mitigations.yaml`

### 生成过程

1. 读取 `atlas_requirement_with_procedure_v5.5.0.json`。
2. 将每个 `procedure[].tactic` 从原始引用字符串扩充为嵌套字典：
   - 使用 `{{xxx.id}}` 中的 `xxx`
   - 在 `tactics.yaml` 中匹配 `&xxx`
3. 将每个 `procedure[].technique` 从原始引用字符串扩充为嵌套字典：
   - 使用 `{{xxx.id}}` 中的 `xxx`
   - 在 `techniques.yaml` 中匹配 `&xxx`
4. 根据每个 step 的原始 `technique` 字符串，在 `mitigations.yaml` 中匹配所有 `techniques[].id` 相等的 mitigation 项。
5. 将所有匹配结果写入 `procedure[].mitigations`。

### 文件内容

这是当前目录中信息最完整的 v5.5.0 版 ATLAS knowledge base。除基础字段外，`procedure` 中每个 step 具有如下结构：

- `tactic`
  - `raw`
  - `anchor`
  - `id`
  - `name`
  - `description`
- `technique`
  - `raw`
  - `anchor`
  - `id`
  - `name`
  - `description`
  - `subtechnique_of`
  - `tactics`
- `description`
- `mitigations`

其中：

- `mitigations` 是一个列表
- 每个列表元素都是 `{ mitigation_anchor: use_text }` 形式的字典

说明：

- `tactic` 和 `technique` 已实现完整结构化映射
- `mitigations` 仅反映官方 `mitigations.yaml` 中显式给出的 technique-to-mitigation 对应关系
- 若某个 technique 的 `mitigations` 为空，通常表示官方数据未提供对应项，而不一定是生成过程错误

---

## 使用建议

如果面向 RAG 使用：

- `owasp_knowledge.json` 适合做 OWASP requirement 与 mitigation 联合检索
- `atlas_requirement_with_procedure_v5.4.0.json` 和 `atlas_requirement_with_procedure_v5.5.0.json` 适合作为基础 requirement + case study 数据
- `atlas_requirement_with_procedure_tactic_nested_with_mitigations_v5.4.0.json` 适合做 v5.4.0 的 tactic / mitigation 辅助检索
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
