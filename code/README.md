# Code Repositories

Cloned repositories relevant to steering vector decay research. All repos are shallow-cloned (`--depth 1`) to minimize disk usage.

## Priority 1: Core Steering Libraries

### 1. steering-vectors/steering-vectors
- **URL:** https://github.com/steering-vectors/steering-vectors
- **Description:** PyTorch/HuggingFace library for training and applying steering vectors. Clean API for contrastive pair training, controllable magnitude application at specific layers. Supports GPT, LLaMA, Gemma, Mistral, Pythia.
- **PyPI:** `pip install steering-vectors` (v0.12.2)
- **Docs:** https://steering-vectors.github.io/steering-vectors
- **Decay relevance:** Magnitude control API enables per-layer decay schedule implementation. Most extensible option for adding decay measurement hooks.

### 2. TransformerLensOrg/TransformerLens
- **URL:** https://github.com/TransformerLensOrg/TransformerLens
- **Description:** Mechanistic interpretability library by Neel Nanda. Exposes all internal activations for 50+ models. Hook system for editing/reading activations during forward passes. Complete activation caching.
- **PyPI:** `pip install transformer-lens`
- **Decay relevance:** Gold standard for measuring activation propagation. Hook system ideal for tracking how steering perturbations decay through layers and across token positions.

### 3. nrimsky/CAA
- **URL:** https://github.com/nrimsky/CAA
- **Description:** Official code for "Steering Llama 2 via Contrastive Activation Addition" (ACL 2024, Panickssery et al.). Includes vector generation, normalization, and evaluation scripts. Pre-computed steering vectors included.
- **Key files:** `generate_vectors.py`, `normalize_vectors.py`, `prompting_with_steering.py`, `llama_wrapper.py`
- **Decay relevance:** Canonical baseline implementation. Applies vectors at ALL positions (continuous reinforcement prevents decay). Contrast with ActAdd's single injection.

### 4. IBM/activation-steering
- **URL:** https://github.com/IBM/activation-steering
- **Description:** General-purpose activation steering library (ICLR 2025 Spotlight). Implements ActAdd + Conditional Activation Steering (CAST). Works with most HuggingFace causal LMs.
- **Decay relevance:** Conditional application mechanism (steer only when condition is met) is conceptually adjacent to decay-aware steering. Well-engineered, good base for production experiments.

## Priority 2: Specialized Methods

### 5. knoveleng/steering (Selective Steering)
- **URL:** https://github.com/knoveleng/steering
- **Local dir:** `selective-steering/`
- **Description:** Norm-preserving steering through discriminative layer selection (Dang & Ngo, 2026). Addresses norm distortion from activation addition in models with LayerNorm/RMSNorm. Modular pipeline with extraction, direction, steering, hooks, and evaluation modules.
- **Decay relevance:** Core finding that norm distortion accumulates across layers is a specific decay mechanism. Norm-preserving approach directly addresses one cause of decay.

### 6. vgel/repeng
- **URL:** https://github.com/vgel/repeng
- **Description:** Python library for generating RepE control vectors. Train a vector in <60 seconds. Supports GGUF export for llama.cpp. Based on Zou et al. 2023.
- **PyPI:** `pip install repeng`
- **Decay relevance:** Fast vector training. GGUF export enables testing decay in quantized inference settings.

### 7. slavachalnev/SAE-TS
- **URL:** https://github.com/slavachalnev/SAE-TS
- **Description:** SAE-Targeted Steering. Uses Sparse Autoencoders to measure causal effects and construct targeted steering vectors. Includes EffectVis visualization tool.
- **Decay relevance:** Feature-level decomposition enables measuring how individual SAE features decay after steering. Causal measurement framework extensible to feature-level decay rates.

### 8. ZJU-REAL/EasySteer
- **URL:** https://github.com/ZJU-REAL/EasySteer
- **Description:** Unified framework for high-performance LLM steering on vLLM. 5.5-11.4x speedup. Supports token-level, position-specific, and multi-vector steering. Pre-computed vectors for 8 domains. OpenAI-compatible API.
- **Decay relevance:** Token-level and position-specific steering enables native implementation of position-dependent decay schedules. vLLM integration for production-throughput experiments.

## Papers Without Code (Relevant Theoretical Frameworks)

### PID Steering (arXiv:2510.04309)
- **Authors:** Dung V. Nguyen et al.
- **Status:** No public code released
- **Key idea:** Frames steering as PID control. Existing methods = P-controllers with steady-state error (= decay). Integral term addresses persistent corrections. Equations can be reimplemented from paper.

### SVF - Steering Vector Fields (arXiv:2602.01654)
- **Authors:** Jiaqian Li et al.
- **Status:** No public code released (Feb 2026)
- **Key idea:** Replaces static vectors with learned vector fields over activation space. Refreshes steering directions from evolving hidden states. Directly addresses static vector misalignment during generation.

## Recommended Experimental Setup

For steering vector decay experiments:
1. **Steering library:** `steering-vectors` or `IBM/activation-steering` for vector training and application
2. **Activation measurement:** `TransformerLens` for deep hook-based activation tracking
3. **Baseline vectors:** `CAA` pre-computed vectors and Anthropic behavioral eval datasets
4. **Evaluation:** `selective-steering` evaluation pipeline for norm-aware metrics
5. **Throughput:** `EasySteer` for large-scale generation experiments on vLLM
