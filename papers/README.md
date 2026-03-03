# Downloaded Papers

## Foundational Steering Vector Papers

1. **Steering Language Models With Activation Engineering** (Turner et al., 2023)
   - File: `2308.10248_activation_addition_steering.pdf`
   - arXiv: 2308.10248
   - Key: Introduces ActAdd — steering vectors computed from prompt pair activation differences, injected once at a specific layer/position. Shows early layers most effective for GPT-2-XL.

2. **Steering Llama 2 via Contrastive Activation Addition** (Panickssery et al., 2024)
   - File: `2312.06681_contrastive_activation_addition.pdf`
   - arXiv: 2312.06681
   - Key: CAA averages activation differences over many contrast pairs. Applies steering at ALL token positions after instruction (unlike ActAdd's single injection). Tests on 7 alignment-relevant behaviors.

3. **Representation Engineering: A Top-Down Approach to AI Transparency** (Zou et al., 2023)
   - File: `2310.01405_representation_engineering.pdf`
   - arXiv: 2310.01405
   - Key: Foundational framework for representation engineering. Various techniques for extracting and steering concept directions.

4. **Refusal in Language Models Is Mediated by a Single Direction** (Arditi et al., 2024)
   - File: `2406.11717_refusal_single_direction.pdf`
   - arXiv: 2406.11717
   - Key: Shows refusal behavior mediated by a single direction. Directional ablation as steering method.

## Papers Directly Relevant to Steering Vector Decay

5. **Steering Vector Fields for Context-Aware Inference-Time Control** (Li et al., 2026)
   - File: `2602.01654_steering_vector_fields.pdf`
   - arXiv: 2602.01654
   - Key: MOST DIRECTLY RELEVANT. Shows static steering vectors become misaligned as hidden states drift during generation. Proposes refreshing context-dependent directions every K steps to address decay. Demonstrates long-form control degradation with static vectors.

6. **Activation Steering with a Feedback Controller** (Nguyen et al., 2025)
   - File: `2510.04309_activation_steering_feedback_controller.pdf`
   - arXiv: 2510.04309
   - Key: HIGHLY RELEVANT. Frames steering as PID control. Shows existing methods (ActAdd, DirAblate, Mean-AcT) are P-controllers with steady-state error. Integral term addresses persistent corrections (anti-decay). Proves steady-state error exists in P-control steering.

7. **Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering** (Bigelow et al., 2025)
   - File: `2511.00617_belief_dynamics_icl_steering.pdf`
   - arXiv: 2511.00617
   - Key: Bayesian model of steering as belief updating. Steering alters concept priors; ICL accumulates evidence. Shows steering beliefs typically only works in specific layers.

8. **Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics** (Xu et al., 2026)
   - File: `2602.02343_why_steering_works.pdf`
   - arXiv: 2602.02343
   - Key: Unifies steering, LoRA, and weight editing as dynamic weight updates. Steering = bias adjustment. Preference-utility tradeoff analysis. Activation manifold perspective.

9. **Towards Understanding Steering Strength** (2026)
   - File: `2506.06975_towards_understanding_steering_strength.pdf`
   - arXiv: 2506.06975
   - Key: Analyzes what determines steering effectiveness and how strength varies.

## Steering Methodology Papers

10. **Analyzing the Generalization and Reliability of Steering Vectors** (Tan et al., 2024)
    - File: `2407.12404_generalization_reliability_steering.pdf`
    - arXiv: 2407.12404
    - Key: Rigorous evaluation of steering limitations. Shows steerability is highly variable. Some concepts unsteerable. Spurious biases contribute to effectiveness.

11. **What Can We Actually Steer?** (Bas et al., 2025)
    - File: `2511.18284_what_can_we_steer.pdf`
    - arXiv: 2511.18284
    - Key: Multi-behavior study across 50 behaviors. Trait expression follows inverted-U curve with steering coefficient.

12. **Towards Reliable Evaluation of Behavior Steering Interventions** (Pres et al., 2024)
    - File: `2410.17245_reliable_evaluation_steering.pdf`
    - arXiv: 2410.17245
    - Key: Evaluation methodology critique. Shows some steering results overstated.

13. **Selective Steering: Norm-Preserving Control** (Dang & Ngo, 2026)
    - File: `2601.19375_selective_steering.pdf`
    - arXiv: 2601.19375
    - Key: Norm-preserving rotation + discriminative layer selection. Layer-wise activation geometry analysis.

14. **Angular Steering: Behavior Control via Rotation** (Vu & Nguyen, 2025)
    - File: (referenced in paper-finder results)
    - Key: Steering via rotation in activation space rather than addition.

## Advanced Steering Methods

15. **Steering Conceptors** (Postmus & Abreu, 2024)
    - File: `2410.16314_steering_conceptors.pdf`
    - arXiv: 2410.16314
    - Key: Uses conceptors (ellipsoidal regions) instead of single vectors. Boolean operations for combining.

16. **Semantics-Adaptive Dynamic Intervention (SADI)** (Wang et al., 2024)
    - File: `2410.12299_semantics_adaptive_steering.pdf`
    - arXiv: 2410.12299
    - Key: Dynamic steering vectors that adapt to input semantics at inference time.

17. **Programming Refusal with Conditional Activation Steering** (Lee et al., 2024)
    - File: `2409.05907_conditional_activation_steering.pdf`
    - arXiv: 2409.05907
    - Key: Conditional steering — applies steering selectively based on input activation patterns.

18. **SAE-Targeted Steering** (Chalnev et al., 2024)
    - File: `2411.02193_sae_targeted_steering.pdf`
    - arXiv: 2411.02193
    - Key: Uses SAEs to measure causal effects and construct targeted steering vectors. ICLR 2025.

19. **Improving Steering Vectors by Targeting SAE Features** (2024)
    - File: `2411.02790_improving_steering_sae_features.pdf`
    - arXiv: 2411.02790

20. **Feature Guided Activation Additions (FGAA)** (2025)
    - File: `2501.09929_feature_guided_activation_additions.pdf`
    - arXiv: 2501.09929

21. **ReFT: Representation Finetuning** (Wu et al., 2024)
    - File: `2405.01563_reft_representation_finetuning.pdf`
    - arXiv: 2405.01563

22. **KV Cache Steering** (2025)
    - File: `2505.18735_kv_cache_steering.pdf`
    - arXiv: 2505.18735

## Theoretical/Analytical Papers

23. **The Linear Representation Hypothesis** (Park et al., 2023)
    - File: `2311.03658_linear_representation_hypothesis.pdf`
    - arXiv: 2311.03658

24. **The Geometry of Truth** (Marks & Tegmark, 2023)
    - File: `2310.06824_geometry_of_truth.pdf`
    - arXiv: 2310.06824

## Hidden State Dynamics & Token-Level Analysis

25. **The Hidden Life of Tokens** (2025)
    - File: `2502.03628_hidden_life_tokens.pdf`
    - arXiv: 2502.03628
    - Key: Visual info in hidden states decays across generated tokens. Uses steering vectors to counteract.

26. **Token-level Uncertainty and Hidden State Dynamics** (2025)
    - File: `2511.04527_token_uncertainty_hidden_state.pdf`
    - arXiv: 2511.04527
    - Key: Correlation between model uncertainty and steerability.

27. **Multiple Token Divergence** (2025)
    - File: `2512.22944_multiple_token_divergence.pdf`
    - arXiv: 2512.22944

28. **Self-Reflective Generation at Test Time** (2025)
    - File: `2510.02919_self_reflective_generation.pdf`
    - arXiv: 2510.02919
    - Key: Hidden state steering for error propagation mitigation during autoregressive generation.

29. **Steering in the Shadows** (2025)
    - File: `2511.17194_steering_in_shadows.pdf`
    - arXiv: 2511.17194
    - Key: BOS anchoring provides persistent steering bias. Studies how perturbations propagate through layers.

30. **The Pitfalls of Next-Token Prediction** (2024)
    - File: `2403.06963_pitfalls_next_token_prediction.pdf`
    - arXiv: 2403.06963
    - Key: Teacher forcing vs autoregressive gap. Error propagation compounds exponentially.
