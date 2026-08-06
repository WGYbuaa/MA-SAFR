# SAAFG split_2.0 Summary

## Construction Rule

test_2.0 selected from original train+dev using deterministic stratification over source groups and silver threat counts; train_2.0/dev_2.0 selected from remaining cases plus all original test cases.

The split is deterministic and does not use a random shuffle. `test_2.0` is drawn only from the original train+dev pool.

## train_2.0

- Cases: 80
- Silver threats: 128
- Dataset distribution: `{'owasp': 53, 'atlas': 27}`
- Old split distribution: `{'test': 45, 'train': 30, 'dev': 5}`
- Source group distribution: `{'LLM01_PromptInjection': 9, 'LLM02_SensitiveInformationDisclosure': 5, 'LLM03_SupplyChain': 7, 'LLM04_DataModelPoisoning': 3, 'LLM05_ImproperOutputHandling': 4, 'LLM06_ExcessiveAgency': 6, 'LLM07_SystemPromptLeakage': 5, 'LLM08_VectorAndEmbeddingWeaknesses': 2, 'LLM09_Misinformation': 2, 'LLM10_UnboundedConsumption': 10, 'atlas': 27}`
- Silver threat count distribution: `{1: 53, 2: 11, 3: 11, 4: 5}`

## dev_2.0

- Cases: 20
- Silver threats: 28
- Dataset distribution: `{'owasp': 14, 'atlas': 6}`
- Old split distribution: `{'test': 12, 'train': 6, 'dev': 2}`
- Source group distribution: `{'LLM01_PromptInjection': 2, 'LLM02_SensitiveInformationDisclosure': 1, 'LLM03_SupplyChain': 2, 'LLM04_DataModelPoisoning': 1, 'LLM05_ImproperOutputHandling': 1, 'LLM06_ExcessiveAgency': 1, 'LLM07_SystemPromptLeakage': 1, 'LLM08_VectorAndEmbeddingWeaknesses': 1, 'LLM09_Misinformation': 1, 'LLM10_UnboundedConsumption': 3, 'atlas': 6}`
- Silver threat count distribution: `{1: 16, 2: 1, 3: 2, 4: 1}`

## test_2.0

- Cases: 57
- Silver threats: 79
- Dataset distribution: `{'owasp': 33, 'atlas': 24}`
- Old split distribution: `{'train': 44, 'dev': 13}`
- Source group distribution: `{'LLM03_SupplyChain': 13, 'LLM04_DataModelPoisoning': 6, 'LLM05_ImproperOutputHandling': 6, 'LLM08_VectorAndEmbeddingWeaknesses': 5, 'LLM09_Misinformation': 3, 'atlas': 24}`
- Silver threat count distribution: `{1: 44, 2: 6, 3: 5, 4: 2}`
