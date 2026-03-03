# Resources Catalog: Steering Vector Decay Research

## Research Hypothesis
"The influence of an activation steering vector v on model hidden states decays over subsequent generation steps after v is no longer added."

---

## Papers (30 total)

All PDFs are in `papers/`. See `papers/README.md` for detailed descriptions.

### Directly Relevant to Decay Hypothesis

| # | Paper | arXiv | File | Key Contribution |
|---|-------|-------|------|------------------|
| 1 | Steering Vector Fields (Li et al., 2026) | 2602.01654 | `2602.01654_steering_vector_fields.pdf` | Shows static vectors misalign during generation; proposes context-dependent refreshing |
| 2 | PID Steering (Nguyen et al., 2025) | 2510.04309 | `2510.04309_activation_steering_feedback_controller.pdf` | Proves P-control has steady-state error; integral term addresses decay |
| 3 | Belief Dynamics (Bigelow et al., 2025) | 2511.00617 | `2511.00617_belief_dynamics_icl_steering.pdf` | Bayesian model: evidence accumulation overrides steering priors |
| 4 | Why Steering Works (Xu et al., 2026) | 2602.02343 | `2602.02343_why_steering_works.pdf` | Manifold projection counteracts perturbation; steering = bias |

### Foundational Methods

| # | Paper | arXiv | File | Key Contribution |
|---|-------|-------|------|------------------|
| 5 | ActAdd (Turner et al., 2023) | 2308.10248 | `2308.10248_activation_addition_steering.pdf` | Single-injection steering; layer-dependent effectiveness |
| 6 | CAA (Panickssery et al., 2024) | 2312.06681 | `2312.06681_contrastive_activation_addition.pdf` | Continuous injection; mean-difference vectors; 7 behaviors |
| 7 | RepE (Zou et al., 2023) | 2310.01405 | `2310.01405_representation_engineering.pdf` | Framework for reading/writing concept representations |
| 8 | Refusal Direction (Arditi et al., 2024) | 2406.11717 | `2406.11717_refusal_single_direction.pdf` | Single direction mediates refusal behavior |

### Mechanisms and Dynamics

| # | Paper | arXiv | File | Key Contribution |
|---|-------|-------|------|------------------|
| 9 | Hidden Life of Tokens (2025) | 2502.03628 | `2502.03628_hidden_life_tokens.pdf` | Info in hidden states decays across tokens |
| 10 | Pitfalls of Next-Token (2024) | 2403.06963 | `2403.06963_pitfalls_next_token_prediction.pdf` | Teacher forcing gap; error compounds exponentially |
| 11 | Steering in the Shadows (2025) | 2511.17194 | `2511.17194_steering_in_shadows.pdf` | BOS anchoring provides persistent steering bias |
| 12 | Token Uncertainty (2025) | 2511.04527 | `2511.04527_token_uncertainty_hidden_state.pdf` | Uncertainty-steerability correlation |
| 13 | Multiple Token Divergence (2025) | 2512.22944 | `2512.22944_multiple_token_divergence.pdf` | Token divergence accumulation patterns |
| 14 | Self-Reflective Generation (2025) | 2510.02919 | `2510.02919_self_reflective_generation.pdf` | Hidden state steering for error propagation |

### Steering Methodology and Evaluation

| # | Paper | arXiv | File | Key Contribution |
|---|-------|-------|------|------------------|
| 15 | Generalization/Reliability (Tan et al., 2024) | 2407.12404 | `2407.12404_generalization_reliability_steering.pdf` | Steerability is variable; spurious biases contribute |
| 16 | What Can We Steer? (Bas et al., 2025) | 2511.18284 | `2511.18284_what_can_we_steer.pdf` | 50-behavior study; inverted-U curve |
| 17 | Reliable Evaluation (Pres et al., 2024) | 2410.17245 | `2410.17245_reliable_evaluation_steering.pdf` | MCQ ≠ generation effectiveness; evaluation critique |
| 18 | Selective Steering (Dang & Ngo, 2026) | 2601.19375 | `2601.19375_selective_steering.pdf` | Norm-preserving rotation; discriminative layer selection |
| 19 | Understanding Steering Strength (2026) | 2506.06975 | `2506.06975_towards_understanding_steering_strength.pdf` | Steering strength determinants |

### Advanced Methods

| # | Paper | arXiv | File | Key Contribution |
|---|-------|-------|------|------------------|
| 20 | Steering Conceptors (2024) | 2410.16314 | `2410.16314_steering_conceptors.pdf` | Ellipsoidal regions; Boolean concept combination |
| 21 | SADI (Wang et al., 2024) | 2410.12299 | `2410.12299_semantics_adaptive_steering.pdf` | Dynamic semantic-adaptive vectors |
| 22 | Conditional Steering (Lee et al., 2024) | 2409.05907 | `2409.05907_conditional_activation_steering.pdf` | Selective application based on activation patterns |
| 23 | SAE-Targeted Steering (2024) | 2411.02193 | `2411.02193_sae_targeted_steering.pdf` | SAE-guided targeted vector construction |
| 24 | Improving Steering SAE Features (2024) | 2411.02790 | `2411.02790_improving_steering_sae_features.pdf` | SAE feature targeting improvements |
| 25 | FGAA (2025) | 2501.09929 | `2501.09929_feature_guided_activation_additions.pdf` | Feature-guided activation additions |
| 26 | ReFT (Wu et al., 2024) | 2405.01563 | `2405.01563_reft_representation_finetuning.pdf` | Representation finetuning |
| 27 | KV Cache Steering (2025) | 2505.18735 | `2505.18735_kv_cache_steering.pdf` | KV cache modification for persistent steering |

### Theoretical

| # | Paper | arXiv | File | Key Contribution |
|---|-------|-------|------|------------------|
| 28 | Linear Representation Hypothesis (2023) | 2311.03658 | `2311.03658_linear_representation_hypothesis.pdf` | Theoretical foundation for linear concept directions |
| 29 | Geometry of Truth (2023) | 2310.06824 | `2310.06824_geometry_of_truth.pdf` | Geometric structure of truth in activation space |
| 30 | Angular Steering (2025) | -- | Referenced in search results | Rotation-based steering |

---

## Datasets

All in `datasets/`. See `datasets/README.md` for quick start and `datasets/dataset_catalog.md` for comprehensive details.

### Downloaded Locally

| Dataset | Location | Size | Use Case |
|---------|----------|------|----------|
| Anthropic MWE (AI Risk) | `datasets/anthropic_evals/advanced_ai_risk/` | 6 files, ~5K examples | Contrastive pair extraction; MCQ evaluation |
| Anthropic MWE (Sycophancy) | `datasets/anthropic_evals/sycophancy/` | 3 files, ~30K examples | Sycophancy steering; large-scale evaluation |
| TruthfulQA (Generation) | `datasets/truthfulqa/TruthfulQA.csv` | 790 questions | Open-ended truthfulness evaluation |
| TruthfulQA (MC) | `datasets/truthfulqa/mc_task.json` | 790 entries | Multiple-choice truthfulness evaluation |
| CAA Processed Pairs | `code/CAA/datasets/` | 7 behaviors | Ready-to-use contrastive pairs (generate/test splits) |
| CAA Pre-computed Vectors | `code/CAA/vectors/` | Llama 2 vectors | Baseline steering vectors |

### Available via HuggingFace (load on demand)

| Dataset | HuggingFace ID | Loading Code |
|---------|---------------|--------------|
| MMLU | `cais/mmlu` | `load_dataset("cais/mmlu", "all")` |
| Natural Questions | `google-research-datasets/natural_questions` | `load_dataset("...", streaming=True)` |
| TruthfulQA (full) | `truthfulqa/truthful_qa` | `load_dataset("truthfulqa/truthful_qa", "generation")` |
| AxBench CONCEPT500 | Via stanfordnlp/axbench | See repo README |

---

## Code Repositories

All in `code/`. See `code/README.md` for detailed descriptions.

### Cloned Locally (8 repos, all shallow)

| Repo | URL | Primary Use |
|------|-----|-------------|
| `steering-vectors/` | github.com/steering-vectors/steering-vectors | Training & applying vectors; PyPI library |
| `TransformerLens/` | github.com/TransformerLensOrg/TransformerLens | Activation inspection & hook system |
| `CAA/` | github.com/nrimsky/CAA | Canonical CAA implementation; pre-computed vectors |
| `activation-steering/` | github.com/IBM/activation-steering | ActAdd + CAST; production-quality library |
| `selective-steering/` | github.com/knoveleng/steering | Norm-preserving steering; modular pipeline |
| `repeng/` | github.com/vgel/repeng | Fast RepE vector training; GGUF export |
| `SAE-TS/` | github.com/slavachalnev/SAE-TS | SAE-targeted vector construction |
| `EasySteer/` | github.com/ZJU-REAL/EasySteer | High-performance vLLM steering framework |

### Referenced but Not Cloned (no public code)

| Paper | Status | Notes |
|-------|--------|-------|
| PID Steering (arXiv:2510.04309) | No code released | Equations reimplementable from paper |
| SVF (arXiv:2602.01654) | No code released (Feb 2026) | Very recent |

### Additional Repositories of Interest

| Repo | URL | Description |
|------|-----|-------------|
| awesome-representation-engineering | github.com/chrisliu298/awesome-representation-engineering | Curated paper list |
| awesome-activation-engineering | github.com/ZFancy/awesome-activation-engineering | Curated activation engineering list |
| extending-activation-addition | github.com/TeunvdWeij/extending-activation-addition | ActAdd extensions |
| honest_llama (ITI) | github.com/likenneth/honest_llama | ITI truthfulness probes |
| stanfordnlp/axbench | github.com/stanfordnlp/axbench | AxBench steering benchmark |

---

## Recommended Experimental Stack

### For Steering Vector Extraction
- **Library:** `steering-vectors` (pip install) or `IBM/activation-steering`
- **Data:** CAA processed contrastive pairs (7 behaviors) + Anthropic MWE
- **Models:** Start with Llama 2-7B-chat (for comparability with CAA), extend to Gemma 2-2B and Llama 3-8B

### For Measuring Decay
- **Activation tracking:** TransformerLens hook system
- **Metrics:** Logit gap, KL divergence, cosine similarity of hidden states
- **Conditions:** Single injection vs. continuous, autoregressive vs. teacher-forced

### For Evaluation
- **MCQ:** TruthfulQA MC, Anthropic MWE A/B format, MMLU (capabilities)
- **Generation:** TruthfulQA generation, open-ended reformulations from CAA
- **Quality:** MT-Bench for multi-turn, AlpacaEval for instruction-following

---

## Project Structure

```
steering-vector-decay-4272-claude/
├── papers/                          # 30 downloaded PDFs
│   ├── README.md                    # Paper catalog with descriptions
│   └── pages/                       # Chunked PDFs for reading
├── datasets/                        # Downloaded evaluation data
│   ├── README.md                    # Quick start guide
│   ├── dataset_catalog.md           # Comprehensive catalog
│   ├── .gitignore                   # Excludes large binary files
│   ├── anthropic_evals/             # Anthropic model-written evals
│   │   ├── advanced_ai_risk/        # 6 behavior JSONL files
│   │   └── sycophancy/              # 3 topic JSONL files (~10k each)
│   └── truthfulqa/                  # TruthfulQA benchmark
│       ├── TruthfulQA.csv           # 790 generation questions
│       └── mc_task.json             # 790 MC entries
├── code/                            # 8 cloned repositories
│   ├── README.md                    # Repository descriptions
│   ├── steering-vectors/            # PyPI steering library
│   ├── TransformerLens/             # Mech interp activation hooks
│   ├── CAA/                         # Canonical CAA + pre-computed vectors
│   ├── activation-steering/         # IBM ActAdd + CAST
│   ├── selective-steering/          # Norm-preserving steering
│   ├── repeng/                      # RepE control vectors
│   ├── SAE-TS/                      # SAE-targeted steering
│   └── EasySteer/                   # vLLM steering framework
├── literature_review.md             # Synthesis of 30 papers
├── resources.md                     # This file
├── pyproject.toml                   # Project configuration
└── .resource_finder_complete        # Completion marker
```
