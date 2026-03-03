# Paper Outline: Steering Vector Decay

## Title
Instant Forgetting: Steering Vectors Leave No Trace in Transformer Hidden States

## Abstract (150-250 words)
- Problem: Activation steering is used to control LLM behavior, but temporal dynamics of steering influence after removal are unknown
- Approach: We measure token-by-token hidden state dynamics after removing steering vectors in GPT-2-XL, comparing autoregressive vs. teacher-forced conditions
- Key results: Representational decay is instantaneous (<1 token); AR and TF conditions are identical; behavioral persistence operates entirely through generated tokens
- Significance: Continuous application is necessary; steering influence is carried by tokens, not hidden state momentum

## 1. Introduction
- Hook: Activation steering is widely used, but do practitioners need to apply vectors continuously?
- Gap: No prior work directly measures how quickly steering effects decay after removal
- Approach: Token-by-token measurement with 3 behaviors, ablations across layers/multipliers/durations
- Preview: Decay is instantaneous; AR = TF; behavioral persistence through tokens only
- Contributions:
  1. First empirical characterization of post-removal steering decay dynamics
  2. Decomposition of representational decay vs. behavioral persistence
  3. Comprehensive ablations showing universality of instantaneous decay

## 2. Related Work
- **Activation steering foundations**: ActAdd, CAA, RepE — continuous injection implicitly acknowledges decay
- **Evidence for decay**: SVF (misalignment during generation), PID (steady-state error), Belief Dynamics (Bayesian override)
- **Mechanisms**: LayerNorm absorption, manifold projection, hidden state drift
- **Methods to counteract decay**: SVF refresh, PID integral, CAST conditional, KV cache

## 3. Methodology
- Model: GPT-2-XL via TransformerLens
- Steering vector extraction: Mean-difference from 20 contrastive pairs per behavior
- Three behaviors: truthful, positive_sentiment, formal_style
- Six conditions: No-steer, Continuous, AR, TF, TF-clean, Random
- Metrics: Cosine similarity delta, projection, KL divergence
- Ablations: Layer (8,16,24,32), multiplier (1-5), duration (1,3,5,10)

## 4. Results
- **Main finding**: Delta drops from ~0.019 to ~0.000 in 1 token (Table 1, Figure 1)
- **AR vs TF**: Identical (Cohen's d = 0.000) — no hidden state memory (Table 2)
- **TF-clean**: Complete reversion to unsteered baseline — behavioral persistence is through tokens
- **Ablations**: All layers, multipliers, durations show instantaneous decay (Table 3)

## 5. Discussion
- Interpretation: LayerNorm and residual connections project perturbations back onto manifold
- Implication: Continuous injection is necessary, not just robust
- Tokens carry steering memory, not hidden states
- Limitations: Single model, single injection layer, greedy decoding, small contrastive sets

## 6. Conclusion
- Summary: Instantaneous representational decay, behavioral persistence through tokens
- Key takeaway: Steering vectors produce no persistent hidden state memory
- Future: Multi-layer steering, larger models, KV-cache steering

## Figures
- Figure 1: key_finding_representational_decay.png — Main decay curve
- Figure 2: key_finding_behavioral_vs_representational.png — Behavioral vs representational
- Figure 3: ablation_summary.png — All ablations

## Tables
- Table 1: Token-by-token cosine similarity delta (truthful direction)
- Table 2: AR vs TF comparison (statistical tests)
- Table 3: Ablation summary (layers, multipliers, durations)
