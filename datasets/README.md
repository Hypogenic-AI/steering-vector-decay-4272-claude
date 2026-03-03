# Datasets for Steering Vector Decay Research

## Contents

### Downloaded Data
- `anthropic_evals/advanced_ai_risk/` - Anthropic model-written evaluations (human-generated) for 6 behaviors
- `anthropic_evals/sycophancy/` - Sycophancy evaluation datasets (3 topics, ~10k each)
- `truthfulqa/TruthfulQA.csv` - TruthfulQA generation benchmark (790 questions)
- `truthfulqa/mc_task.json` - TruthfulQA multiple-choice task (790 entries)

### Also Available Via Cloned Repos
- `../code/CAA/datasets/` - Pre-processed contrastive pairs for 7 behaviors (generate/test splits ready)
- `../code/CAA/vectors/` - Pre-computed steering vectors for Llama 2
- `../code/CAA/normalized_vectors/` - Normalized steering vectors

### Comprehensive Catalog
- `dataset_catalog.md` - Full catalog with loading code, format details, and experimental design notes

## Quick Start

### Load Anthropic Behavioral Evals
```python
import json

with open("datasets/anthropic_evals/advanced_ai_risk/survival-instinct.jsonl") as f:
    data = [json.loads(line) for line in f]

# Each example has:
# - "question": A/B multiple-choice dialogue prompt
# - "answer_matching_behavior": "(A)" or "(B)"
print(f"Loaded {len(data)} examples")
print(data[0]["question"][:200])
```

### Load TruthfulQA
```python
import json

with open("datasets/truthfulqa/mc_task.json") as f:
    data = json.loads(f.read())

# Each example has: question, mc1 (single correct), mc2 (multi correct)
print(f"Loaded {len(data)} questions")
```

### Load HuggingFace Datasets (requires `datasets` library)
```python
# pip install datasets
from datasets import load_dataset

# TruthfulQA generation config
ds = load_dataset("truthfulqa/truthful_qa", "generation")

# MMLU for capabilities preservation
ds = load_dataset("cais/mmlu", "all")

# Anthropic evals (full dataset)
ds = load_dataset("Anthropic/model-written-evals")
```

## Datasets Not Downloaded (too large, use HuggingFace streaming)
- **MMLU** (`cais/mmlu`): 14k test questions. Use `load_dataset("cais/mmlu", "all")`.
- **Natural Questions** (`google-research-datasets/natural_questions`): 300k+ questions. Use streaming mode.
- **AxBench CONCEPT500**: Via `stanfordnlp/axbench` GitHub repo.
- **MT-Bench**: Via `lm-sys/FastChat` GitHub repo.
