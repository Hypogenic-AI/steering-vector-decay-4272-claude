# Literature Review: Steering Vector Decay in Language Models

## Research Hypothesis

> "The influence of an activation steering vector **v** on model hidden states decays over subsequent generation steps after **v** is no longer added. The rate of this decay may differ if tokens generated with **v** are teacher-forced but not added to the hidden state."

This review synthesizes 30 papers relevant to understanding, measuring, and mitigating the temporal decay of steering vector influence during autoregressive generation.

---

## 1. Foundations of Activation Steering

### 1.1 The Linear Representation Hypothesis

The theoretical grounding for steering vectors comes from the **linear representation hypothesis** (Park et al., 2023; arXiv:2311.03658): high-level concepts correspond to approximately linear directions in transformer activation space. Marks & Tegmark (2023; arXiv:2310.06824) provide geometric evidence that truth values occupy a linear subspace, giving empirical support for the idea that adding a vector along a concept's direction should shift model behavior accordingly.

### 1.2 Activation Addition (ActAdd)

Turner et al. (2023; arXiv:2308.10248) introduced **Activation Addition (ActAdd)**, the foundational steering method. Key properties relevant to decay:

- **Single injection**: ActAdd computes a steering vector from the activation difference between a positive and negative prompt pair, then adds it at a **single layer and token position** during the forward pass
- **Layer-dependent effectiveness**: Effectiveness rises through early layers, peaks around layer 6 (for GPT-2-XL), then declines. This layer-dependent profile is itself a form of spatial decay
- **Minimal data**: Requires as few as 2 contrast prompts
- **Limitation**: The single-injection approach means the perturbation must propagate through all subsequent layers and token positions without reinforcement -- making it inherently subject to decay

### 1.3 Contrastive Activation Addition (CAA)

Panickssery et al. (2024; arXiv:2312.06681) developed **Contrastive Activation Addition (CAA)**, which differs from ActAdd in ways directly relevant to decay:

- **Continuous injection**: Steering vectors are added at **all token positions after the instruction**, not just one. This amounts to continuous reinforcement that prevents within-sequence decay by constantly re-applying the perturbation
- **Mean-difference vectors**: Uses hundreds of contrastive pairs (vs. ActAdd's single pair) and averages activation differences, producing more robust directions
- **7 alignment behaviors tested**: sycophancy, survival-instinct, corrigibility, coordinate-other-ais, myopic-reward, hallucination, and refusal
- **The design choice of continuous injection implicitly acknowledges that single-injection steering would decay** -- though the paper does not frame it this way

### 1.4 Representation Engineering (RepE)

Zou et al. (2023; arXiv:2310.01405) provided the broader **Representation Engineering** framework, establishing that concept directions can be extracted via various methods (PCA, mean difference, probing) and used for both reading and writing model representations. This framework treats steering as manipulating a learned representation space, making decay analogous to signal attenuation in that space.

---

## 2. Direct Evidence for Steering Vector Decay

### 2.1 Steering Vector Fields (SVF) -- The Primary Decay Paper

Li et al. (2026; arXiv:2602.01654) provide the **most direct evidence** for steering vector decay. Their key findings:

- **Static vectors become misaligned**: During autoregressive generation, hidden states drift through activation space. A static steering vector computed at one point becomes increasingly misaligned with the concept direction at later token positions
- **Long-form generation failure**: "Reliability degrades in long-form generation" with static steering vectors -- the effect diminishes as generation proceeds
- **Proposed solution**: Replace static vectors with a **learned vector field** over activation space. A differentiable concept scoring function's local gradient defines the steering direction at each activation state
- **Refreshing mechanism**: Re-compute the steering direction every K steps from the current hidden state, effectively counteracting decay by adapting to the model's evolving representation state
- **Evaluation**: Uses MWE (Anthropic model-written evals) for MCQ evaluation and TruthfulQA for open-ended generation, demonstrating that static vectors degrade on generation tasks while vector fields maintain effectiveness

**Relevance to hypothesis**: SVF directly demonstrates that steering influence decays during generation and proposes a mechanism (hidden state drift causing misalignment with a fixed direction). The decay is not about the vector's magnitude diminishing but about its **direction becoming irrelevant** as the model moves through activation space.

### 2.2 PID Steering -- Control-Theoretic Decay Framework

Nguyen et al. (2025; arXiv:2510.04309) provide the **strongest theoretical framework** for understanding decay, framing steering as a **control theory problem**:

- **Existing methods are P-controllers**: ActAdd, CAA, and directional ablation are all equivalent to proportional-only controllers in a dynamical system where layers are time steps
- **Proposition 1 (Steady-State Error)**: They formally prove that P-control steering has inherent **steady-state error** -- the activation state converges to a value that is offset from the target direction. This steady-state error is mathematically equivalent to incomplete steering, i.e., decay
- **Integral term counteracts decay**: Adding an integral component (I-term) accumulates past steering errors and provides persistent corrections. This directly addresses the temporal persistence problem
- **Derivative term prevents overshoot**: The D-term dampens oscillations that could occur from over-correction
- **Results**: PID steering achieves better toxicity reduction than P-only methods on Gemma2-2B and Llama3-8B while maintaining generation quality

**Relevance to hypothesis**: The PID framework formalizes decay as steady-state error in a P-control system. The integral term's necessity proves that single-application steering has inherent temporal limitations. This provides the most rigorous theoretical basis for our hypothesis.

### 2.3 Belief Dynamics and Bayesian Decay

Bigelow et al. (2025; arXiv:2511.00617) model steering through a **Bayesian lens**:

- **Steering alters concept priors**: A steering vector shifts the model's prior probability for a concept
- **ICL accumulates evidence**: In-context learning provides evidence that can reinforce or override the prior
- **Evidence accumulation can overwhelm steering**: As the model processes more tokens, the accumulated evidence from the actual text can override the steering-induced prior shift -- a form of Bayesian decay
- **Layer specificity**: Steering beliefs only works in specific layers, and the effective layers vary by concept

**Relevance to hypothesis**: This provides an information-theoretic perspective on decay. The steering vector sets a prior, but as generation proceeds and the model generates tokens consistent with its original tendency, each token provides evidence against the steered behavior. The decay rate would then depend on the model's likelihood ratio between steered and unsteered behaviors.

---

## 3. Mechanisms of Decay

### 3.1 Hidden State Drift

The **Hidden Life of Tokens** (2025; arXiv:2502.03628) demonstrates that information in hidden states decays across generated tokens even without steering. Visual information injected into hidden states diminishes as more tokens are generated. This suggests that **any perturbation to hidden states -- including steering vectors -- will naturally attenuate** as the autoregressive process progresses and the model's own dynamics dominate.

### 3.2 Error Propagation and Compounding

The **Pitfalls of Next-Token Prediction** (Bachmann & Nagarajan, 2024; arXiv:2403.06963) establishes the theoretical foundation for understanding why perturbations compound during autoregressive generation:

- **Teacher forcing vs. autoregressive gap**: During training, models see ground-truth prefixes (teacher forcing). During generation, models condition on their own (potentially erroneous) outputs
- **Exponential error accumulation**: Small perturbations in hidden states can compound exponentially through the autoregressive process
- **Relevance to decay hypothesis**: This suggests two competing effects: (a) the steering perturbation itself may decay through hidden state drift, but (b) any behavioral changes caused by early steered tokens may persist or even amplify as the model conditions on its own steered outputs. This creates a distinction between **representational decay** (the vector's direct effect on hidden states) and **behavioral persistence** (the indirect effect through generated tokens)

### 3.3 Token Divergence and Uncertainty

**Multiple Token Divergence** (2025; arXiv:2512.22944) and **Token-level Uncertainty and Hidden State Dynamics** (2025; arXiv:2511.04527) provide additional mechanisms:

- Divergence between steered and unsteered generation may accumulate or diminish depending on the model's confidence at each token position
- High-uncertainty tokens may be more susceptible to steering influence, while low-uncertainty tokens (where the model is confident) may resist steering regardless of vector magnitude

### 3.4 BOS Anchoring and Persistent Perturbation

**Steering in the Shadows** (2025; arXiv:2511.17194) studies how perturbations to the BOS (beginning-of-sequence) token propagate through subsequent layers:

- **BOS provides persistent bias**: Because the BOS token's representation is attended to by all subsequent tokens, perturbations to it provide a form of persistent steering
- **Propagation patterns**: The study characterizes how perturbations spread through the attention mechanism, providing a map of how steering effects might be maintained or lost

### 3.5 Norm Distortion as Decay Mechanism

Dang & Ngo (2026; arXiv:2601.19375) in their **Selective Steering** work identify a specific mechanism: adding a steering vector violates the activation norms expected by LayerNorm/RMSNorm, causing distribution shift. This norm violation compounds across layers (a spatial form of decay), and is especially problematic in sub-7B models. Their norm-preserving rotation approach demonstrates that maintaining geometric properties of activations prevents this specific decay pathway.

---

## 4. Why Steering Works (and Why It Might Stop Working)

### 4.1 Unified View of Steering

Xu et al. (2026; arXiv:2602.02343) provide a unifying perspective in **"Why Steering Works"**:

- **Steering = dynamic bias adjustment**: Adding a steering vector is mathematically equivalent to adding a bias term to the model's weights at the target layer
- **Activation manifold perspective**: Model activations live on a low-dimensional manifold. Steering pushes activations off this manifold. The model's subsequent layers then "project" the activations back onto the manifold -- this projection is what causes decay
- **Preference-utility tradeoff**: Stronger steering increases target behavior but degrades utility. This tradeoff curve's shape directly relates to how aggressively the model's dynamics counteract the perturbation

### 4.2 Steering Strength Analysis

**"Towards Understanding Steering Strength"** (2026; arXiv:2506.06975) analyzes what determines steering effectiveness:

- Effectiveness depends on the alignment between the steering direction and the model's internal concept representations
- Some concepts have clearer linear directions than others, leading to variable steerability
- The strength needed to overcome the model's default behavior varies by concept, suggesting that decay rates may also be concept-dependent

### 4.3 Steerability Limitations

Tan et al. (2024; arXiv:2407.12404) and Bas et al. (2025; arXiv:2511.18284) provide rigorous analyses:

- **Variable steerability**: Some concepts are highly steerable, others are not. Spurious biases can contribute to apparent steerability
- **Inverted-U curve**: Trait expression follows an inverted-U curve with steering coefficient -- too much steering actually reverses the effect. This suggests a nonlinear relationship between perturbation magnitude and behavioral outcome that complicates simple decay models
- **50-behavior study**: Tests across 50 behaviors show that steering effectiveness is behavior-dependent, implying decay rates likely vary across behaviors too

---

## 5. Methods for Counteracting Decay

### 5.1 Continuous Injection (CAA Approach)

The simplest anti-decay method: add the steering vector at every token position (Panickssery et al., 2024). This prevents temporal decay by brute-force reinforcement but has limitations:

- Does not adapt to changing context
- May cause cumulative norm distortion
- Prevents studying natural decay dynamics

### 5.2 Context-Dependent Refreshing (SVF)

Li et al. (2026) refresh the steering direction from the current hidden state. This adapts to hidden state drift but requires training a concept scoring function.

### 5.3 PID Control (Integral Accumulation)

Nguyen et al. (2025) accumulate past steering errors via an integral term. This provides persistent correction without requiring constant re-injection, making it more theoretically principled than continuous injection.

### 5.4 Conditional Steering (CAST)

Lee et al. (2024; arXiv:2409.05907) apply steering selectively based on input activation patterns. IBM's CAST (Conditional Activation Steering) extends this by applying steering only when a condition vector indicates relevance. This implicitly handles decay by re-checking at each position whether steering is still needed.

### 5.5 Semantics-Adaptive Intervention (SADI)

Wang et al. (2024; arXiv:2410.12299) adapt steering vectors to input semantics at inference time, addressing the problem that a fixed direction may not remain appropriate as context changes.

### 5.6 Norm-Preserving Steering

Dang & Ngo (2026) apply steering via rotation rather than addition, preserving activation norms and preventing norm-distortion-based decay.

### 5.7 SAE-Targeted Steering

Chalnev et al. (2024; arXiv:2411.02193) and related work (arXiv:2411.02790) use Sparse Autoencoders to construct steering vectors that target specific features while minimizing off-target effects. By reducing unintended perturbations, these methods may produce steering that decays more slowly because it is better aligned with the model's natural representation structure.

### 5.8 KV Cache Steering

KV Cache Steering (2025; arXiv:2505.18735) modifies the key-value cache rather than hidden states. Since the KV cache persists across all subsequent tokens, this may provide inherently more persistent steering than hidden-state injection.

---

## 6. Evaluation Methodology

### 6.1 MCQ vs. Generation Evaluation

Pres et al. (2024; arXiv:2410.17245) demonstrate that steering effectiveness measured via multiple-choice questions (MCQs) may not predict generation-level effectiveness:

- Short-response MCQ evaluations can overstate steering success
- Long-form generation is a stricter test where decay becomes apparent
- This distinction is critical for studying decay: MCQ evaluation measures the instantaneous effect, while generation evaluation measures the temporally-extended effect

### 6.2 Token-Level Metrics for Decay

Based on the literature, the following metrics are appropriate for measuring steering vector decay:

1. **Logit gap over time**: P(target behavior) - P(anti-behavior) at each token position after steering removal
2. **KL divergence from steered baseline**: How quickly the output distribution reverts to the unsteered model's distribution
3. **Cosine similarity of hidden states**: Between the steered model's hidden states and the unsteered model's hidden states at each layer and position
4. **Behavioral classifier scores**: Apply a behavior classifier to sliding windows of generated text

### 6.3 Capabilities Preservation

Multiple papers (CAA, WAS, SVF, Selective Steering) use MMLU to verify that steering does not degrade general capabilities. For decay experiments, this is important as a control: does the decay of steering influence correlate with recovery of baseline capabilities?

---

## 7. Synthesis: A Framework for Studying Decay

### 7.1 Two Types of Decay

The literature suggests two distinct mechanisms:

1. **Representational decay**: The direct effect of the steering vector on hidden state geometry diminishes as the model processes more tokens. Caused by hidden state drift (SVF), norm re-normalization (Selective Steering), and the model's inherent tendency to project activations back onto its learned manifold (Why Steering Works).

2. **Behavioral persistence**: Even after the representational effect decays, generated tokens that were influenced by steering remain in the context. These tokens provide "evidence" (in the Bayesian sense of Belief Dynamics) that may maintain the steered behavior even without continued representational intervention.

### 7.2 The Teacher Forcing Distinction

Our hypothesis specifically mentions that "the rate of this decay may differ if tokens generated with v are teacher-forced but not added to the hidden state." This relates to:

- **With autoregressive generation**: Steered tokens enter the context, potentially maintaining behavioral persistence even as representational decay occurs
- **With teacher forcing (ground truth tokens)**: The model sees unsteered tokens in its context, so there is no behavioral persistence -- only representational decay is measured
- This distinction is supported by Bachmann & Nagarajan (2024) on the teacher-forcing gap and by the Bayesian framework of Bigelow et al. (2025)

### 7.3 Predicted Decay Dynamics

Combining the theoretical frameworks:

1. **Immediate effect** (t=0): Full steering influence as measured by logit gap
2. **Rapid initial decay** (t=1 to ~5): The model's LayerNorm and subsequent transformations begin counteracting the perturbation. PID steady-state error theory predicts convergence to a non-zero but reduced level
3. **Plateau or continued decay** (t>5): Depends on whether behavioral persistence (through autoregressive feedback) maintains the effect. Under teacher forcing, expect continued decay. Under autoregressive generation, may plateau
4. **Concept-dependent variation**: Highly steerable concepts (clear linear directions) likely decay slower than weakly steerable ones

### 7.4 Key Experimental Comparisons

From the literature, the critical experimental conditions to compare are:

| Condition | Representational Decay | Behavioral Persistence |
|-----------|----------------------|----------------------|
| Single injection (ActAdd-style) + autoregressive | Yes | Possible |
| Single injection + teacher forcing | Yes | No |
| Continuous injection (CAA-style) | Counteracted | Yes |
| PID control | Actively corrected | Yes |
| No intervention (baseline) | N/A | N/A |

The difference between conditions 1 and 2 isolates the teacher-forcing effect predicted by our hypothesis.

---

## 8. Key Papers Summary Table

| Paper | Year | Key Finding for Decay | Type |
|-------|------|----------------------|------|
| SVF (Li et al.) | 2026 | Static vectors misalign during generation | Direct evidence |
| PID Steering (Nguyen et al.) | 2025 | P-control has inherent steady-state error | Theoretical framework |
| ActAdd (Turner et al.) | 2023 | Layer-dependent effectiveness (spatial decay) | Foundation |
| CAA (Panickssery et al.) | 2024 | Continuous injection prevents decay | Foundation |
| Belief Dynamics (Bigelow et al.) | 2025 | Bayesian evidence accumulation overrides steering | Theoretical |
| Why Steering Works (Xu et al.) | 2026 | Manifold projection counteracts perturbation | Mechanism |
| Hidden Life of Tokens | 2025 | Information in hidden states naturally decays | Mechanism |
| Pitfalls of Next-Token (Bachmann) | 2024 | Teacher forcing vs. autoregressive gap | Mechanism |
| Selective Steering (Dang & Ngo) | 2026 | Norm distortion accumulates across layers | Mechanism |
| Pres et al. | 2024 | Short-response ≠ generation effectiveness | Evaluation |
| Steering in the Shadows | 2025 | BOS perturbation propagation patterns | Mechanism |
| Refusal Direction (Arditi et al.) | 2024 | Single direction mediates refusal | Concept structure |
| SADI (Wang et al.) | 2024 | Fixed vectors fail across varying inputs | Anti-decay method |
| Tan et al. | 2024 | Steerability is highly variable | Limitations |
| Bas et al. | 2025 | Inverted-U curve with steering coefficient | Nonlinearity |

---

## 9. Open Questions

1. **What is the functional form of decay?** Exponential? Power law? Does it vary by concept?
2. **Is behavioral persistence sufficient to maintain steering without representational support?** If yes, at what confidence threshold does the model's own generation maintain the steered behavior?
3. **How does model scale affect decay rate?** Larger models have more layers for the perturbation to propagate through but may also have more stable representation manifolds
4. **Can decay rate predict steerability?** Concepts that decay slowly may be the ones most amenable to steering
5. **How does quantization affect decay?** Reduced numerical precision may accelerate decay through rounding
6. **What is the interaction between decay and attention?** Do attention patterns to steered tokens maintain or counteract the steering effect?

---

## References

Full papers available in `papers/` directory. See `papers/README.md` for complete listing with file names and arXiv IDs.
