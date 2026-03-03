# Research Plan: Steering Vector Decay

## Motivation & Novelty Assessment

### Why This Research Matters
Activation steering is increasingly used to control LLM behavior at inference time. However, the temporal dynamics of steering—specifically how quickly steering influence dissipates after the vector is no longer applied—remain poorly characterized. Understanding decay rates is critical for practitioners who want to minimize computational overhead (applying vectors only when needed) and for theorists seeking to understand how models process and "wash out" perturbations.

### Gap in Existing Work
The literature review reveals that:
- **CAA** (Panickssery et al., 2024) applies vectors at every token position, implicitly acknowledging decay but never measuring it
- **SVF** (Li et al., 2026) observes that static vectors become misaligned during generation but focuses on proposing adaptive solutions, not characterizing decay dynamics
- **PID Steering** (Nguyen et al., 2025) proves P-control has steady-state error but doesn't empirically measure token-by-token decay curves
- **No paper directly measures** how quickly hidden states revert after steering is removed partway through generation
- **No paper compares** autoregressive vs. teacher-forced decay, which isolates representational decay from behavioral persistence

### Our Novel Contribution
We provide the **first direct empirical characterization** of steering vector decay dynamics:
1. Token-by-token measurement of how hidden states lose steering influence after removal
2. Quantification of decay functional form (exponential? power law?)
3. Direct comparison of autoregressive vs. teacher-forced conditions, isolating two decay mechanisms: representational decay (hidden state geometry reverting) vs. behavioral persistence (steered tokens maintaining influence through context)

### Experiment Justification
- **Experiment 1 (Representational Decay Curves):** Directly tests the core hypothesis by measuring cosine similarity and projection of hidden states onto the steering direction at each post-removal token. This answers: "How fast does the direct representational effect decay?"
- **Experiment 2 (Autoregressive vs. Teacher-Forced):** Tests whether the decay rate changes when we remove the feedback loop of steered tokens entering the context. This isolates representational decay from behavioral persistence.
- **Experiment 3 (Ablations):** Tests generalizability across steering durations, layers, behaviors, and multiplier strengths.

## Research Question
After applying a steering vector v for the first K tokens of generation, how quickly does v's influence on hidden states decay over subsequent tokens? Does this decay rate differ when tokens generated under steering are teacher-forced (model sees them but v is not added) vs. when the model generates freely?

## Hypothesis Decomposition
1. **H1:** Cosine similarity between steered and unsteered hidden states decreases monotonically after steering removal (representational decay occurs)
2. **H2:** Decay follows an approximately exponential curve (informed by PID steady-state error theory)
3. **H3:** Decay is faster under teacher-forcing than autoregressive generation (because autoregressive generation maintains behavioral persistence through steered tokens in context)
4. **H4:** Decay rate varies across behaviors (some concepts have more stable linear representations)

## Proposed Methodology

### Approach
Use TransformerLens to load a small-to-medium model (GPT-2-XL or Pythia-1.4B), extract steering vectors from contrastive pairs using the steering-vectors library, then measure hidden state dynamics token-by-token under controlled conditions.

**Model choice:** GPT-2-XL (1.5B params, 48 layers, d_model=1600). Well-supported by TransformerLens, fits easily on one A6000, and has been used in the original ActAdd paper. We'll also validate on Pythia-1.4B for generalizability.

**Why not Llama 2?** While CAA provides pre-computed Llama 2 vectors, TransformerLens support for Llama 2 is less mature and the model is larger. Our focus is on measuring dynamics, not achieving SOTA steering.

### Experimental Steps

#### Step 1: Steering Vector Extraction
- Create contrastive pairs for 3 behaviors: positive/negative sentiment, truthful/untruthful, formal/informal
- Use steering-vectors library to extract mean-difference vectors at all layers
- Validate vectors work by checking they shift logit distributions as expected

#### Step 2: Baseline Characterization
- For each behavior, generate 50 continuations from diverse prompts
- Measure baseline hidden states (no steering) at all layers for all tokens
- Establish metrics: cosine similarity to steering direction, projection magnitude, KL divergence of output distributions

#### Step 3: Experiment 1 — Representational Decay Curves
For each of {K=1, 3, 5} steering tokens:
- Apply steering vector at layer L for the first K generated tokens
- Remove steering vector and continue generating 20 more tokens
- At each token position, record:
  - Cosine similarity of hidden state (at steered layer) to the steering vector
  - Projection of hidden state onto steering direction
  - KL divergence of logit distribution vs. unsteered baseline
  - Cosine similarity of full hidden state to the continuously-steered hidden state

#### Step 4: Experiment 2 — Autoregressive vs. Teacher-Forced
Three conditions, all with K=3 steered tokens followed by 20 post-removal tokens:
1. **Autoregressive (AR):** Model generates freely after steering removal. Context includes tokens it generated under steering.
2. **Teacher-Forced (TF):** After steering removal, we feed the same tokens that AR generated, but compute forward passes without adding the steering vector. This isolates representational decay by providing the same token sequence but without steering.
3. **Teacher-Forced with Unsteered Tokens (TF-clean):** After steering removal, feed the tokens that would have been generated WITHOUT any steering. This measures whether steered tokens in context alone maintain steering influence.

#### Step 5: Experiment 3 — Ablations
- Vary steering layer: early (L=6), middle (L=24), late (L=42)
- Vary multiplier strength: 0.5x, 1x, 2x, 4x
- Vary steering duration: K=1, 3, 5, 10
- Vary behavior type across all 3 behaviors
- Measure decay half-life for each condition

### Baselines
- **No steering:** Baseline hidden states for comparison
- **Continuous steering:** Steering applied at all tokens (upper bound of influence)
- **Random vector:** Apply a random direction vector with same norm (control for perturbation vs. meaningful direction)

### Evaluation Metrics
1. **Cosine similarity to steering direction** (cos_sim): How aligned is the hidden state with v?
2. **Projection magnitude** (proj): How much of the hidden state lies along v?
3. **KL divergence from steered baseline** (KL_steered): How close is the output distribution to the continuously-steered model?
4. **KL divergence from unsteered baseline** (KL_unsteered): How close has the output distribution reverted to unsteered?
5. **Decay half-life** (t_half): Number of tokens for the metric to reach 50% of its initial (steered) value

### Statistical Analysis Plan
- Fit exponential decay curves: metric(t) = A * exp(-λt) + C
- Compare decay rates (λ) across conditions using paired t-tests
- Report 95% confidence intervals from bootstrap resampling (N=50 prompts per condition)
- Bonferroni correction for multiple comparisons
- Cohen's d effect sizes for AR vs. TF comparisons

## Expected Outcomes
- **Supports H1:** Cosine similarity to steering direction decreases monotonically after removal
- **Supports H2:** Decay is approximately exponential with a possible non-zero asymptote
- **Supports H3:** Teacher-forced decay is significantly faster (higher λ) than autoregressive decay
- **Supports H4:** Different behaviors show different decay half-lives, with more "natural" directions decaying slower

## Timeline and Milestones
1. Environment setup + vector extraction: 20 min
2. Baseline characterization: 15 min
3. Experiment 1 (decay curves): 30 min
4. Experiment 2 (AR vs TF): 30 min
5. Experiment 3 (ablations): 30 min
6. Analysis + visualization: 30 min
7. Documentation: 20 min

## Potential Challenges
- TransformerLens generation with custom token-by-token hooks may need manual implementation
- Teacher-forcing experiment requires careful implementation to feed correct tokens
- Decay may be too fast (< 3 tokens) or too slow (> 50 tokens) to characterize well
- GPU memory for caching activations across many tokens and conditions

## Success Criteria
1. Clear decay curves with quantified half-lives for at least 2 behaviors
2. Statistical comparison of AR vs. TF decay rates with p < 0.05
3. Fitted decay model with R² > 0.8
4. Reproducible results across at least 2 random seeds
