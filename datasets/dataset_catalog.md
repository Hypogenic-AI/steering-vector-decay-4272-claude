# Dataset Catalog for Steering Vector Decay Experiments

## Research Hypothesis
"The influence of an activation steering vector v on model hidden states decays over subsequent generation steps after v is no longer added."

---

## 1. PRIMARY DATASETS (for steering vector extraction and MCQ evaluation)

### 1a. Anthropic Model-Written Evaluations (MWE) -- Advanced AI Risk

- **HuggingFace**: `Anthropic/model-written-evals`
- **URL**: https://huggingface.co/datasets/Anthropic/model-written-evals
- **GitHub mirror**: https://github.com/anthropics/evals
- **License**: CC-BY-4.0
- **Paper**: Perez et al. (2022), "Discovering Language Model Behaviors with Model-Written Evaluations" (arXiv:2212.09251)

**Loading**:
```python
# Option 1: HuggingFace datasets library (loads the top-level metadata)
from datasets import load_dataset
ds = load_dataset("Anthropic/model-written-evals")

# Option 2: Direct JSONL file download (recommended for specific files)
import requests, json
base = "https://huggingface.co/datasets/Anthropic/model-written-evals/resolve/main"
# Example: advanced-ai-risk human-generated evals
url = f"{base}/advanced-ai-risk/human_generated_evals/survival-instinct.jsonl"
lines = requests.get(url).text.strip().split("\n")
data = [json.loads(line) for line in lines]
```

**Subdirectory: `advanced-ai-risk/human_generated_evals/`**

These are the files used by CAA (Panickssery et al. 2024) and SVF (Li et al. 2026):

| File | Size | Description |
|------|------|-------------|
| `coordinate-itself.jsonl` | 107 kB | AI coordinating with itself |
| `coordinate-other-ais.jsonl` | 143 kB | AI coordinating with other AIs |
| `coordinate-other-versions.jsonl` | 121 kB | AI coordinating with other versions |
| `corrigible-less-HHH.jsonl` | 107 kB | Corrigibility toward less HHH goal |
| `corrigible-more-HHH.jsonl` | 114 kB | Corrigibility toward more HHH goal |
| `corrigible-neutral-HHH.jsonl` | 102 kB | Corrigibility toward neutral HHH goal |
| `myopic-reward.jsonl` | 309 kB | Myopic reward seeking |
| `one-box-tendency.jsonl` | 417 kB | Newcomb's one-box tendency |
| `power-seeking-inclination.jsonl` | 356 kB | Power seeking |
| `self-awareness-general-ai.jsonl` | 59.5 kB | Self-awareness as general AI |
| `self-awareness-good-text-model.jsonl` | 122 kB | Self-awareness as good text model |
| `self-awareness-text-model.jsonl` | 56.1 kB | Self-awareness as text model |
| `self-awareness-training-architecture.jsonl` | 66.4 kB | Self-awareness of training |
| `self-awareness-web-gpt.jsonl` | 60.5 kB | Self-awareness as web GPT |
| `survival-instinct.jsonl` | 332 kB | Survival instinct / shutdown avoidance |
| `wealth-seeking-inclination.jsonl` | 319 kB | Wealth seeking |

**Format**: Each line is a JSON object with a `question` field (containing the A/B multiple-choice question as a dialogue prompt) and an `answer_matching_behavior` field (indicating which choice, "(A)" or "(B)", matches the target behavior). Approximately 200-500 examples per file.

**SVF paper categories used** (Tan et al. 2025 steerability profiles):
- WEALTH-SEEKING, POWER-SEEKING, MYOPIC-REWARD, SURVIVAL-INSTINCT (from advanced-ai-risk)
- INTEREST-IN-SCIENCE, BELIEVES-IT-IS-NOT-BEING-WATCHED, NARCISSISM (from persona)
- CORRIGIBLE (for open-ended generation)
- Split: 40/10/50 train/validation/test

**Subdirectory: `advanced-ai-risk/lm_generated_evals/`**

Contains ~24.5k LLM-generated multiple-choice questions across 16 categories (same topics as human_generated_evals but larger scale, lower quality).

**Suitability for decay experiments**: EXCELLENT. These are the standard datasets for steering vector extraction. The A/B format provides clean contrastive pairs for computing mean-difference steering vectors. Evaluating on held-out MCQs measures whether steering persists in logit space across generated tokens.

---

### 1b. Anthropic Model-Written Evaluations -- Sycophancy

- **Path within dataset**: `sycophancy/`
- **URL**: https://huggingface.co/datasets/Anthropic/model-written-evals/tree/main/sycophancy

| File | Size | Description |
|------|------|-------------|
| `sycophancy_on_nlp_survey.jsonl` | 9.73 MB | NLP research opinion questions (~10k examples) |
| `sycophancy_on_philpapers2020.jsonl` | 9.73 MB | Philosophy questions from PhilPapers 2020 (~10k examples) |
| `sycophancy_on_political_typology_quiz.jsonl` | 7.77 MB | Political opinion questions from Pew (~10k examples) |

**Format**: Each line is a JSON object containing a biography of a user with a particular view, followed by an A/B question. The model is tested on whether it agrees with the user's stated view.

**Loading**:
```python
import requests, json
url = "https://huggingface.co/datasets/Anthropic/model-written-evals/resolve/main/sycophancy/sycophancy_on_nlp_survey.jsonl"
lines = requests.get(url).text.strip().split("\n")
data = [json.loads(line) for line in lines]
```

**Suitability for decay experiments**: EXCELLENT. Sycophancy is the most-studied behavior in steering vector literature. The large size (~10k per topic) enables robust train/test splits. CAA uses a mixture of these with GPT-4-generated data for 1000 generate / 50 test pairs.

---

### 1c. Anthropic Model-Written Evaluations -- Persona

- **Path within dataset**: `persona/`
- **URL**: https://huggingface.co/datasets/Anthropic/model-written-evals/tree/main/persona

Contains datasets testing political views, religious views, personality traits, moral beliefs, and desires for dangerous goals. Includes categories like:
- `interest-in-science`, `narcissism`, `believes-it-is-not-being-watched-by-humans`
- Various political, religious, and personality tests

**Suitability for decay experiments**: GOOD for MCQ evaluation. SVF uses three persona categories. However, many prompts are Yes/No style rather than open-ended, making them less suitable for studying generation-level decay. Use primarily for MCQ-based decay measurement.

---

## 2. CAA PROCESSED DATASETS (ready-to-use contrastive pairs)

### 2a. nrimsky/CAA Repository

- **GitHub**: https://github.com/nrimsky/CAA
- **Paper**: Panickssery et al. (2024), "Steering Llama 2 via Contrastive Activation Addition" (arXiv:2312.06681, ACL 2024)

**Pre-processed datasets in `/datasets` directory**:

| Behavior | Generate Pairs | Test Pairs | Source |
|----------|---------------|------------|--------|
| coordinate-other-ais | 360 | 50 | Anthropic human-generated evals |
| corrigible-neutral-HHH | 290 | 50 | Anthropic human-generated evals |
| hallucination | 1000 | 50 | Custom (unprompted + context-triggered) |
| myopic-reward | 950 | 50 | Anthropic human-generated evals |
| survival-instinct | 903 | 50 | Anthropic human-generated evals |
| sycophancy | 1000 | 50 | Anthropic sycophancy + GPT-4 generated |
| refusal | 408 | 50 | Custom |

**Format**: Processed into contrastive pairs using `process_raw_datasets.py`. Each pair shares the same question context but differs in the behavior-indicating completion (answer A vs answer B). "Generate" pairs are used for computing the mean-difference steering vector; "test" pairs are held out for evaluation.

**Loading**:
```bash
git clone https://github.com/nrimsky/CAA.git
# Data is in CAA/datasets/
# Vectors are in CAA/vectors/ and CAA/normalized_vectors/
```

**Additional data**:
- `mmlu_full.json` and `mmlu.json` for capabilities evaluation
- Open-ended test questions (reformatted from A/B to free-form for some behaviors; GPT-4-generated for sycophancy)
- TruthfulQA evaluation data

**Suitability for decay experiments**: EXCELLENT. This is the most directly usable dataset for our experiments. The generate/test split is already prepared. The hallucination and sycophancy datasets are specifically designed for both MCQ and open-ended evaluation. The 7 behaviors provide diverse steering targets across which to measure decay rates.

---

## 3. TRUTHFULNESS EVALUATION

### 3a. TruthfulQA

- **HuggingFace**: `truthfulqa/truthful_qa`
- **URL**: https://huggingface.co/datasets/truthfulqa/truthful_qa
- **GitHub**: https://github.com/sylinrl/TruthfulQA
- **License**: Apache 2.0
- **Paper**: Lin et al. (2022), "TruthfulQA: Measuring How Models Mimic Human Falsehoods" (arXiv:2109.07958)

**Size**: 817 questions across 38 categories (health, law, finance, politics, etc.)

**Configurations**:

| Config | Split | Rows | Use Case |
|--------|-------|------|----------|
| `generation` | validation | 817 | Open-ended truthfulness evaluation |
| `multiple_choice` | validation | 817 | MC1 (single correct) and MC2 (multi correct) |

**Loading**:
```python
from datasets import load_dataset

# Generation config (for open-ended evaluation)
ds_gen = load_dataset("truthfulqa/truthful_qa", "generation")
# Fields: type, category, question, best_answer, correct_answers, incorrect_answers, source

# Multiple choice config
ds_mc = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
# Fields: question, mc1_targets (choices, labels), mc2_targets (choices, labels)
```

**Suitability for decay experiments**: EXCELLENT. Used by both ITI (Li et al. 2024) and SVF (Li et al. 2026) for truthfulness steering evaluation. The generation config is ideal for measuring how a truthfulness steering vector's effect decays over multi-token responses. The 817 questions provide enough statistical power. Also serves as an out-of-distribution test when using hallucination steering vectors trained on CAA data.

---

## 4. CAPABILITIES PRESERVATION (control benchmarks)

### 4a. MMLU (Massive Multitask Language Understanding)

- **HuggingFace**: `cais/mmlu` or `hails/mmlu_no_train` (faster loading, no auxiliary_train split)
- **URL**: https://huggingface.co/datasets/cais/mmlu
- **Paper**: Hendrycks et al. (2021)

**Size**: 57 subjects, ~14k test questions total

**Loading**:
```python
from datasets import load_dataset

# Full MMLU
ds = load_dataset("cais/mmlu", "all")

# Specific subject
ds = load_dataset("cais/mmlu", "abstract_algebra")

# Faster version without auxiliary_train
ds = load_dataset("hails/mmlu_no_train", "all")
```

**Suitability for decay experiments**: IMPORTANT CONTROL. Not for measuring decay per se, but essential for verifying that steering interventions (and their decay) do not degrade general capabilities. Multiple steering papers (CAA, WAS, SVF) use MMLU to measure capabilities preservation. Compare MMLU accuracy with vs. without steering at various token positions.

### 4b. Natural Questions (NQ)

- **HuggingFace**: `google-research-datasets/natural_questions`
- **URL**: https://huggingface.co/datasets/google-research-datasets/natural_questions
- **Paper**: Kwiatkowski et al. (2019)

**Size**: Very large (~300k+ questions from real Google searches)

**Loading**:
```python
from datasets import load_dataset
# Use streaming due to large size
ds = load_dataset("google-research-datasets/natural_questions", streaming=True)
```

**Suitability for decay experiments**: USEFUL for concept contamination testing. SVF (Li et al. 2026) uses NQ to measure whether steering causes concept leakage into unrelated question-answering. For decay experiments, can test whether contamination also decays over tokens.

---

## 5. GENERATION-LEVEL EVALUATION BENCHMARKS

### 5a. MT-Bench (Multi-Turn Benchmark)

- **GitHub**: https://github.com/lm-sys/FastChat (part of the FastChat repo)
- **HuggingFace Spaces**: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
- **Paper**: Zheng et al. (2023), "Judging LLM-as-a-Judge"

**Size**: 80 multi-turn questions across 8 categories (writing, roleplay, reasoning, math, coding, extraction, STEM, humanities)

**Loading**: Available through the FastChat library or as JSON from the repo.

**Suitability for decay experiments**: GOOD for measuring whether steered models maintain quality in multi-turn settings. Particularly relevant because multi-turn dialogue involves many generation steps, making it a natural testbed for decay. Several steering papers (KTS, CAA) report MT-Bench scores.

### 5b. AlpacaEval 2.0

- **GitHub**: https://github.com/tatsu-lab/alpaca_eval
- **URL**: https://tatsu-lab.github.io/alpaca_eval/

**Size**: 805 instructions

**Loading**:
```python
# pip install alpaca_eval
from alpaca_eval import evaluate
```

**Suitability for decay experiments**: MODERATE. Emerging as a standard for evaluating steered model output quality. HyperSteer and AlphaSteer use AlpacaEval. Less directly relevant to measuring decay per se, but useful for assessing whether models with decayed steering still produce coherent outputs.

### 5c. AxBench (CONCEPT500)

- **GitHub**: https://github.com/stanfordnlp/axbench
- **Paper**: Wu et al. (2025), "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders" (arXiv:2501.17148, ICML 2025)

**Size**: 500 concepts with synthetically generated training/validation datasets; 16K concept training data

**Loading**: Via the axbench library; datasets released on HuggingFace as "AxBench Collections"

**Suitability for decay experiments**: MODERATE. Useful as a comprehensive steering benchmark covering 500 diverse concepts. Could be used to test whether decay rates vary systematically across concept types. However, the synthetic nature of the data may limit ecological validity.

---

## 6. SPECIALIZED STEERING DATASETS

### 6a. ITI Truthfulness Probes (Li et al. 2024)

- **GitHub**: https://github.com/likenneth/honest_llama
- **Paper**: Li et al. (2024), "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model" (NeurIPS 2024)

**Data**: Uses TruthfulQA for probe training. A few hundred examples suffice to identify truthful attention heads. Also uses `nq_open` and `trivia_qa` for transfer evaluation.

**Suitability for decay experiments**: RELEVANT METHODOLOGY. ITI applies the same intervention autoregressively across all generated tokens, which directly relates to our decay hypothesis. If the intervention is removed after a few tokens, we can measure decay of truthful generation.

### 6b. Hallucination Dataset (from CAA)

- **Source**: Part of the nrimsky/CAA repository
- **Size**: 1000 generate / 50 test examples
- **Types**: Unprompted hallucination + context-triggered hallucination

**Suitability for decay experiments**: EXCELLENT. Hallucination control is one of the strongest steering behaviors. The context-triggered subset is especially interesting because it tests whether the model reverts to hallucination once the steering vector's influence decays.

---

## 7. STEERING VECTOR LIBRARY (for implementation)

### steering-vectors (Python package)

- **PyPI**: `pip install steering-vectors`
- **GitHub**: https://github.com/steering-vectors/steering-vectors
- **Docs**: https://steering-vectors.github.io/steering-vectors/
- **Version**: 0.12.2

```python
from steering_vectors import train_steering_vector

# Training pairs: list of (positive_prompt, negative_prompt)
training_pairs = [
    ("The capital of France is Paris", "The capital of France is London"),
    # ... more pairs
]
sv = train_steering_vector(model, tokenizer, training_pairs)

# Apply with controllable multiplier
sv.apply(model, multiplier=1.5)
# Generate with steering...
sv.remove(model)
```

Compatible with all HuggingFace decoder-only models (GPT, LLaMA, Gemma, Mistral, Pythia, etc.).

---

## RECOMMENDED DATASET SELECTION FOR DECAY EXPERIMENTS

### Tier 1: Must-Use (core experiments)

1. **CAA processed datasets** (nrimsky/CAA): Pre-built contrastive pairs for 7 behaviors. Use for steering vector extraction and primary decay measurement.
2. **TruthfulQA** (`truthfulqa/truthful_qa`, generation config): Open-ended generation evaluation. Ideal for measuring token-by-token decay of truthfulness steering.
3. **MWE advanced-ai-risk** (`Anthropic/model-written-evals`): MCQ evaluation for survival-instinct, corrigibility, sycophancy. The A/B format enables precise measurement of logit-gap changes across generation steps.

### Tier 2: Recommended (robustness and controls)

4. **MWE sycophancy** (`Anthropic/model-written-evals/sycophancy/`): Large-scale sycophancy data (10k+ per topic). Provides statistical power for measuring subtle decay effects.
5. **MMLU** (`cais/mmlu`): Capabilities preservation control. Verify steering and its decay do not degrade general knowledge.
6. **MT-Bench**: Multi-turn generation quality. Tests whether decay pattern affects conversational coherence.

### Tier 3: Extended (additional analyses)

7. **Natural Questions** (`google-research-datasets/natural_questions`): Concept contamination analysis.
8. **AxBench CONCEPT500**: Cross-concept decay rate comparison across 500 concepts.
9. **AlpacaEval**: Output quality assessment for steered+decayed models.

### Key Experimental Design Notes

- **For measuring decay**: Apply the steering vector for only the first K tokens of generation, then remove it. Measure the logit gap (target behavior probability minus anti-behavior probability) at each subsequent token position. The decay curve logit_gap(t) for t > K characterizes the temporal persistence of steering.
- **Token-level metrics** (logit gap, KL divergence from steered baseline) should be complemented by **sequence-level metrics** (LLM-as-judge scoring of full responses, MCQ accuracy on open-ended reformulations).
- **Pres et al. (2024)** (arXiv:2410.17245) explicitly found that steering effects in short-response settings may not carry over to longer continuations, providing direct motivation for studying decay.
- **SVF (Li et al. 2026)** addresses decay by "refreshing" steering directions during decoding, treating decay as a known failure mode of static steering vectors.
