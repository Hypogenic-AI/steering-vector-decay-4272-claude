"""
Steering Vector Decay Experiment
================================
Measures how quickly steering vector influence on hidden states decays
after the vector is no longer applied during generation.

Three experimental conditions:
1. Autoregressive (AR): Model generates freely after steering removal
2. Teacher-Forced (TF): Feed AR-generated tokens without steering vector
3. Teacher-Forced Clean (TF-clean): Feed unsteered tokens without steering vector

Metrics tracked at each token position:
- Cosine similarity of hidden state to steering direction
- Projection magnitude onto steering direction
- KL divergence from steered/unsteered baselines
"""

import os
os.environ.setdefault("USER", "researcher")

import json
import random
import time
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ─── Contrastive Pairs for Steering Vector Training ─────────────────────────

BEHAVIOR_PAIRS = {
    "positive_sentiment": [
        ("I think this movie is absolutely wonderful and amazing.", "I think this movie is absolutely terrible and awful."),
        ("The food at this restaurant was delicious and satisfying.", "The food at this restaurant was disgusting and revolting."),
        ("I had a great time at the party last night.", "I had a horrible time at the party last night."),
        ("This book is one of the best I've ever read.", "This book is one of the worst I've ever read."),
        ("The weather today is beautiful and perfect.", "The weather today is miserable and dreadful."),
        ("I love spending time with my family.", "I hate spending time with my family."),
        ("The concert was incredible and breathtaking.", "The concert was terrible and disappointing."),
        ("This new software is intuitive and helpful.", "This new software is confusing and useless."),
        ("The vacation was relaxing and enjoyable.", "The vacation was stressful and unpleasant."),
        ("I feel optimistic about the future.", "I feel pessimistic about the future."),
        ("The presentation was engaging and informative.", "The presentation was boring and uninformative."),
        ("This neighborhood is safe and welcoming.", "This neighborhood is dangerous and unwelcoming."),
        ("The customer service was excellent and friendly.", "The customer service was terrible and rude."),
        ("I'm excited about my new job opportunity.", "I'm dreading my new job situation."),
        ("The garden looks beautiful this spring.", "The garden looks terrible this spring."),
        ("My experience with this product was fantastic.", "My experience with this product was awful."),
        ("The team collaboration was smooth and productive.", "The team collaboration was chaotic and unproductive."),
        ("This city has a wonderful vibrant culture.", "This city has a dreary depressing atmosphere."),
        ("The results exceeded all expectations.", "The results fell far below expectations."),
        ("I'm grateful for everything in my life.", "I'm disappointed with everything in my life."),
    ],
    "formal_style": [
        ("I would like to express my sincere gratitude for your assistance.", "Thanks a bunch for helping me out, you're the best!"),
        ("It is with great pleasure that I accept your invitation.", "Sure, I'll totally come to your thing, sounds fun!"),
        ("Please find attached the documents for your review.", "Here are the docs, take a look when you get a chance."),
        ("We regret to inform you that your application was unsuccessful.", "Sorry dude, you didn't get the job this time."),
        ("I wish to inquire about the availability of the position.", "Hey, is that job still up for grabs?"),
        ("Your prompt attention to this matter would be appreciated.", "Can you get on this ASAP? Thanks!"),
        ("I am writing to formally request a meeting.", "Hey, can we chat sometime this week?"),
        ("The committee has reached a unanimous decision.", "Everyone agreed on what to do."),
        ("We acknowledge receipt of your correspondence.", "Got your message, thanks!"),
        ("I would be delighted to discuss this further at your convenience.", "Let's talk about this whenever you're free!"),
        ("The organization maintains strict adherence to established protocols.", "We pretty much follow the rules around here."),
        ("Pursuant to our previous discussion, I am providing the requested information.", "Like we talked about, here's the info you wanted."),
        ("It is imperative that we address this matter with utmost urgency.", "We really need to deal with this right now!"),
        ("I respectfully submit my resignation effective immediately.", "I'm out of here, starting now."),
        ("The data suggests a statistically significant correlation.", "The numbers show these things are totally connected."),
        ("We are committed to maintaining the highest standards of excellence.", "We try to do our best work all the time."),
        ("Your cooperation in this endeavor is greatly valued.", "Thanks for pitching in, it really helps!"),
        ("The fiscal quarter demonstrated robust financial performance.", "We made good money this quarter."),
        ("I shall endeavor to fulfill all obligations in a timely manner.", "I'll try to get everything done on time."),
        ("The board of directors convened to deliberate on strategic priorities.", "The bosses got together to figure out what to do next."),
    ],
    "truthful": [
        ("The Earth revolves around the Sun.", "The Sun revolves around the Earth."),
        ("Water boils at 100 degrees Celsius at sea level.", "Water boils at 50 degrees Celsius at sea level."),
        ("Humans need oxygen to survive.", "Humans need carbon dioxide to survive."),
        ("The speed of light is approximately 300,000 km per second.", "The speed of light is approximately 300 km per second."),
        ("DNA carries genetic information in living organisms.", "Proteins carry genetic information in living organisms."),
        ("Gravity causes objects to fall toward the ground.", "Magnetism causes objects to fall toward the ground."),
        ("The Pacific Ocean is the largest ocean on Earth.", "The Atlantic Ocean is the largest ocean on Earth."),
        ("Antibiotics treat bacterial infections, not viral infections.", "Antibiotics treat viral infections, not bacterial infections."),
        ("The Great Wall of China was built over many centuries.", "The Great Wall of China was built in a single year."),
        ("Photosynthesis converts sunlight into chemical energy in plants.", "Photosynthesis converts moonlight into chemical energy in plants."),
        ("The Moon causes tides on Earth through gravitational pull.", "The Sun alone causes tides on Earth through radiation."),
        ("Diamonds are made of carbon atoms.", "Diamonds are made of silicon atoms."),
        ("Sound travels faster in water than in air.", "Sound travels faster in air than in water."),
        ("The human body has 206 bones in adulthood.", "The human body has 106 bones in adulthood."),
        ("Lightning is caused by electrical discharge in the atmosphere.", "Lightning is caused by magnetic discharge in the atmosphere."),
        ("Iron is attracted to magnets.", "Copper is attracted to magnets."),
        ("The Amazon rainforest produces significant amounts of oxygen.", "The Sahara desert produces significant amounts of oxygen."),
        ("Vaccines help the immune system fight diseases.", "Vaccines weaken the immune system against diseases."),
        ("Mount Everest is the tallest mountain above sea level.", "Mount Kilimanjaro is the tallest mountain above sea level."),
        ("Computers process information using binary code.", "Computers process information using decimal code."),
    ],
}

GENERATION_PROMPTS = [
    "The most important thing to remember is that",
    "In my experience, I have found that",
    "When considering the matter carefully,",
    "It is worth noting that recently",
    "Looking at the situation from a different angle,",
    "One cannot deny the fact that",
    "The fundamental principle behind this is",
    "After careful consideration, it seems that",
    "The key takeaway from all of this is",
    "From a broader perspective, we can see that",
    "What many people don't realize is that",
    "The evidence clearly suggests that",
    "According to recent developments,",
    "When we examine the data closely,",
    "The underlying mechanism here is",
    "A critical aspect to consider is",
    "The current state of affairs indicates",
    "Research has consistently shown that",
    "The most compelling argument is that",
    "Given the circumstances, it appears that",
    "An interesting observation is that",
    "The primary concern regarding this is",
    "Historically, we have seen that",
    "The implication of these findings is",
    "What stands out most is that",
]


def load_model(model_name="gpt2-xl"):
    """Load model with TransformerLens."""
    from transformer_lens import HookedTransformer
    print(f"Loading {model_name}...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(
        model_name,
        device=DEVICE,
        dtype=torch.float16,
    )
    model.eval()
    print(f"Loaded {model_name} in {time.time()-t0:.1f}s")
    print(f"  Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")
    return model


def extract_steering_vector(model, behavior_name, layers=None):
    """
    Extract a steering vector using mean-difference of contrastive pairs.
    Uses TransformerLens run_with_cache to get activations.
    Returns dict mapping layer_idx -> steering vector (d_model,).
    """
    pairs = BEHAVIOR_PAIRS[behavior_name]
    if layers is None:
        # Use middle layers by default
        n = model.cfg.n_layers
        layers = list(range(n // 4, 3 * n // 4))

    steering_vectors = {}
    for layer_idx in layers:
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        pos_acts = []
        neg_acts = []

        for pos_text, neg_text in pairs:
            # Get activations for positive example
            _, cache_pos = model.run_with_cache(
                pos_text,
                names_filter=[hook_name],
            )
            # Last token activation
            pos_act = cache_pos[hook_name][0, -1, :].detach().float()
            pos_acts.append(pos_act)

            # Get activations for negative example
            _, cache_neg = model.run_with_cache(
                neg_text,
                names_filter=[hook_name],
            )
            neg_act = cache_neg[hook_name][0, -1, :].detach().float()
            neg_acts.append(neg_act)

        # Mean difference
        pos_mean = torch.stack(pos_acts).mean(dim=0)
        neg_mean = torch.stack(neg_acts).mean(dim=0)
        sv = pos_mean - neg_mean
        # Normalize to unit norm
        sv = sv / sv.norm()
        steering_vectors[layer_idx] = sv.to(DEVICE)

    return steering_vectors


def validate_steering_vector(model, sv_dict, behavior_name, layer_idx):
    """Quick validation: does the vector shift logits in the expected direction?"""
    test_prompt = "I think that"
    tokens = model.to_tokens(test_prompt)
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    sv = sv_dict[layer_idx]

    # Unsteered logits
    logits_base = model(tokens)[0, -1, :]

    # Steered logits (positive direction)
    def add_sv(act, hook, vec=sv, strength=3.0):
        act[:, :, :] = act[:, :, :] + strength * vec.half()
        return act

    logits_steered = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, add_sv)]
    )[0, -1, :]

    # Check KL divergence
    p = F.softmax(logits_base.float(), dim=-1)
    q = F.softmax(logits_steered.float(), dim=-1)
    kl = F.kl_div(q.log(), p, reduction='sum').item()
    print(f"  Validation KL(steered||base) for {behavior_name} at L{layer_idx}: {kl:.2f}")
    return kl > 0.1  # Should be meaningfully different


@dataclass
class DecayMeasurement:
    """Single measurement at one token position."""
    token_pos: int  # relative to steering removal (0 = first unsteered token)
    cos_sim_to_sv: float  # cosine similarity of hidden state diff to steering vec
    proj_onto_sv: float  # projection magnitude onto steering direction
    kl_from_steered: float  # KL divergence from continuously-steered distribution
    kl_from_unsteered: float  # KL divergence from unsteered distribution


@dataclass
class ExperimentResult:
    """Result from one experimental run."""
    behavior: str
    condition: str  # "AR", "TF", "TF_clean", "continuous", "no_steer", "random"
    steer_layer: int
    steer_duration: int  # K tokens steered
    multiplier: float
    prompt: str
    seed: int
    measurements: list = field(default_factory=list)
    generated_tokens: list = field(default_factory=list)


def generate_with_steering_and_measure(
    model,
    prompt,
    sv_dict,
    steer_layer,
    steer_duration,
    post_steer_tokens,
    multiplier=3.0,
    condition="AR",
    teacher_tokens=None,
):
    """
    Generate tokens, applying steering for steer_duration tokens, then measure decay.

    Args:
        model: HookedTransformer model
        prompt: text prompt to start generation
        sv_dict: dict of layer_idx -> steering vector
        steer_layer: which layer to apply steering at
        steer_duration: K = number of tokens to steer
        post_steer_tokens: number of tokens to generate after steering removal
        multiplier: steering strength
        condition: "AR" (autoregressive), "TF" (teacher forced with AR tokens),
                   "TF_clean" (teacher forced with unsteered tokens),
                   "continuous" (steer all tokens), "no_steer" (baseline)
        teacher_tokens: pre-generated token sequence for TF conditions

    Returns:
        ExperimentResult with token-by-token measurements
    """
    sv = sv_dict[steer_layer]
    hook_name = f"blocks.{steer_layer}.hook_resid_post"
    total_new_tokens = steer_duration + post_steer_tokens

    tokens = model.to_tokens(prompt)
    prompt_len = tokens.shape[1]

    # Storage for measurements
    all_hidden_states = []  # list of (token_pos, hidden_state_at_steer_layer)
    all_logits = []

    # We'll do manual autoregressive generation
    current_tokens = tokens.clone()

    for step in range(total_new_tokens):
        # Determine if we should steer at this step
        is_steered = False
        if condition == "continuous":
            is_steered = True
        elif condition == "no_steer" or condition == "random":
            is_steered = False
        elif condition in ("AR", "TF", "TF_clean"):
            is_steered = (step < steer_duration)

        # Build hooks for this step
        fwd_hooks = []

        # Storage for hidden state at this step
        step_hidden = {}

        def capture_hook(act, hook, storage=step_hidden):
            # act shape: [batch, seq_len, d_model]
            # Capture the last token's hidden state
            storage["h"] = act[0, -1, :].detach().float().clone()
            return None

        fwd_hooks.append((hook_name, capture_hook))

        if is_steered:
            def steer_hook(act, hook, vec=sv, s=multiplier):
                # Add steering vector to all positions (but mainly last token matters for generation)
                act[:, -1, :] = act[:, -1, :] + s * vec.half()
                return act

            # We need both hooks: capture first, then steer
            # Actually let's capture AFTER steering so we see the steered state
            fwd_hooks = []

            def capture_and_steer_hook(act, hook, vec=sv, s=multiplier, storage=step_hidden):
                storage["h_pre"] = act[0, -1, :].detach().float().clone()
                act[:, -1, :] = act[:, -1, :] + s * vec.half()
                storage["h"] = act[0, -1, :].detach().float().clone()
                return act

            fwd_hooks = [(hook_name, capture_and_steer_hook)]

        elif condition == "random" and step < steer_duration:
            # Apply random vector with same norm as steering vector
            rand_vec = torch.randn_like(sv)
            rand_vec = rand_vec / rand_vec.norm() * sv.norm()

            def random_hook(act, hook, vec=rand_vec, s=multiplier, storage=step_hidden):
                storage["h_pre"] = act[0, -1, :].detach().float().clone()
                act[:, -1, :] = act[:, -1, :] + s * vec.half()
                storage["h"] = act[0, -1, :].detach().float().clone()
                return act

            fwd_hooks = [(hook_name, random_hook)]

        # Forward pass
        logits = model.run_with_hooks(
            current_tokens,
            fwd_hooks=fwd_hooks,
        )
        last_logits = logits[0, -1, :].detach().float()

        all_hidden_states.append(step_hidden.get("h", step_hidden.get("h_pre")))
        all_logits.append(last_logits)

        # Get next token
        if condition == "TF" and step >= steer_duration and teacher_tokens is not None:
            # Teacher forcing: use pre-specified tokens
            tf_idx = step - steer_duration
            if tf_idx < len(teacher_tokens):
                next_token = teacher_tokens[tf_idx].unsqueeze(0).unsqueeze(0)
            else:
                next_token = torch.argmax(last_logits).unsqueeze(0).unsqueeze(0)
        elif condition == "TF_clean" and teacher_tokens is not None:
            if step < len(teacher_tokens):
                next_token = teacher_tokens[step].unsqueeze(0).unsqueeze(0)
            else:
                next_token = torch.argmax(last_logits).unsqueeze(0).unsqueeze(0)
        else:
            # Greedy decoding
            next_token = torch.argmax(last_logits).unsqueeze(0).unsqueeze(0)

        next_token = next_token.to(current_tokens.device)
        current_tokens = torch.cat([current_tokens, next_token], dim=1)

    return all_hidden_states, all_logits, current_tokens


def compute_decay_metrics(
    model,
    prompt,
    sv_dict,
    steer_layer,
    steer_duration,
    post_steer_tokens,
    multiplier=3.0,
    behavior_name="",
):
    """
    Run all conditions for one prompt and compute decay metrics.
    Returns dict of condition -> ExperimentResult.
    """
    sv = sv_dict[steer_layer]
    total = steer_duration + post_steer_tokens

    # 1. No steering baseline
    h_nosteer, logits_nosteer, tokens_nosteer = generate_with_steering_and_measure(
        model, prompt, sv_dict, steer_layer, steer_duration, post_steer_tokens,
        multiplier=multiplier, condition="no_steer",
    )

    # 2. Continuous steering (upper bound)
    h_cont, logits_cont, tokens_cont = generate_with_steering_and_measure(
        model, prompt, sv_dict, steer_layer, steer_duration, post_steer_tokens,
        multiplier=multiplier, condition="continuous",
    )

    # 3. AR condition: steer for K, then free generation
    h_ar, logits_ar, tokens_ar = generate_with_steering_and_measure(
        model, prompt, sv_dict, steer_layer, steer_duration, post_steer_tokens,
        multiplier=multiplier, condition="AR",
    )

    # Extract AR-generated tokens after steering for teacher forcing
    prompt_len = model.to_tokens(prompt).shape[1]
    ar_gen_tokens = tokens_ar[0, prompt_len + steer_duration:]

    # 4. TF condition: teacher-force with AR tokens (but no steering)
    h_tf, logits_tf, tokens_tf = generate_with_steering_and_measure(
        model, prompt, sv_dict, steer_layer, steer_duration, post_steer_tokens,
        multiplier=multiplier, condition="TF",
        teacher_tokens=ar_gen_tokens,
    )

    # 5. TF_clean: teacher-force with unsteered tokens
    nosteer_gen_tokens = tokens_nosteer[0, prompt_len:]
    h_tf_clean, logits_tf_clean, tokens_tf_clean = generate_with_steering_and_measure(
        model, prompt, sv_dict, steer_layer, steer_duration, post_steer_tokens,
        multiplier=multiplier, condition="TF_clean",
        teacher_tokens=nosteer_gen_tokens,
    )

    # 6. Random vector control
    h_rand, logits_rand, tokens_rand = generate_with_steering_and_measure(
        model, prompt, sv_dict, steer_layer, steer_duration, post_steer_tokens,
        multiplier=multiplier, condition="random",
    )

    # ─── Compute metrics ───
    conditions = {
        "no_steer": (h_nosteer, logits_nosteer, tokens_nosteer),
        "continuous": (h_cont, logits_cont, tokens_cont),
        "AR": (h_ar, logits_ar, tokens_ar),
        "TF": (h_tf, logits_tf, tokens_tf),
        "TF_clean": (h_tf_clean, logits_tf_clean, tokens_tf_clean),
        "random": (h_rand, logits_rand, tokens_rand),
    }

    results = {}
    for cond_name, (h_states, logit_list, gen_tokens) in conditions.items():
        measurements = []
        for step in range(total):
            h = h_states[step]
            if h is None:
                continue

            # 1. Cosine similarity to steering vector
            cos_sim = F.cosine_similarity(h.unsqueeze(0), sv.unsqueeze(0)).item()

            # 2. Projection onto steering direction
            proj = torch.dot(h, sv).item()

            # 3. KL from continuously-steered distribution
            p_cont = F.softmax(logits_cont[step], dim=-1).clamp(min=1e-10)
            p_this = F.softmax(logit_list[step], dim=-1).clamp(min=1e-10)
            kl_from_steered = F.kl_div(
                p_this.log(), p_cont, reduction='sum'
            ).item()

            # 4. KL from unsteered distribution
            p_nosteer = F.softmax(logits_nosteer[step], dim=-1).clamp(min=1e-10)
            kl_from_unsteered = F.kl_div(
                p_this.log(), p_nosteer, reduction='sum'
            ).item()

            rel_pos = step - steer_duration  # negative = during steering
            measurements.append(DecayMeasurement(
                token_pos=rel_pos,
                cos_sim_to_sv=cos_sim,
                proj_onto_sv=proj,
                kl_from_steered=kl_from_steered,
                kl_from_unsteered=kl_from_unsteered,
            ))

        result = ExperimentResult(
            behavior=behavior_name,
            condition=cond_name,
            steer_layer=steer_layer,
            steer_duration=steer_duration,
            multiplier=multiplier,
            prompt=prompt,
            seed=SEED,
            measurements=[asdict(m) for m in measurements],
            generated_tokens=model.to_str_tokens(gen_tokens[0])[
                model.to_tokens(prompt).shape[1]:
            ] if gen_tokens is not None else [],
        )
        results[cond_name] = result

    return results


def run_experiment_1(model, behaviors_to_test, steer_layer, n_prompts=20):
    """
    Experiment 1: Basic representational decay curves.
    Steer for K=3 tokens, then measure decay for 20 tokens.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Representational Decay Curves")
    print("="*70)

    all_results = {}
    for behavior in behaviors_to_test:
        print(f"\n--- Behavior: {behavior} ---")
        sv_dict = extract_steering_vector(model, behavior, layers=[steer_layer])
        valid = validate_steering_vector(model, sv_dict, behavior, steer_layer)
        if not valid:
            print(f"  WARNING: Steering vector for {behavior} may be weak")

        prompts = GENERATION_PROMPTS[:n_prompts]
        behavior_results = []

        for i, prompt in enumerate(tqdm(prompts, desc=f"  {behavior}")):
            result = compute_decay_metrics(
                model, prompt, sv_dict, steer_layer,
                steer_duration=3, post_steer_tokens=20,
                multiplier=3.0, behavior_name=behavior,
            )
            behavior_results.append(result)

        all_results[behavior] = behavior_results

    return all_results


def run_experiment_2(model, behavior, steer_layer, n_prompts=25):
    """
    Experiment 2: Autoregressive vs Teacher-Forced comparison.
    Focus on AR vs TF vs TF_clean conditions.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: AR vs Teacher-Forced Decay Comparison")
    print("="*70)

    sv_dict = extract_steering_vector(model, behavior, layers=[steer_layer])

    prompts = GENERATION_PROMPTS[:n_prompts]
    all_results = []

    for prompt in tqdm(prompts, desc=f"  {behavior} AR-vs-TF"):
        result = compute_decay_metrics(
            model, prompt, sv_dict, steer_layer,
            steer_duration=3, post_steer_tokens=20,
            multiplier=3.0, behavior_name=behavior,
        )
        all_results.append(result)

    return all_results


def run_experiment_3(model, behavior, n_prompts=10):
    """
    Experiment 3: Ablations across layers, durations, multipliers.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Ablation Studies")
    print("="*70)

    n_layers = model.cfg.n_layers
    # Test layers: early, middle, late
    test_layers = [n_layers // 6, n_layers // 3, n_layers // 2, 2 * n_layers // 3]
    test_durations = [1, 3, 5, 10]
    test_multipliers = [1.0, 2.0, 3.0, 5.0]

    prompts = GENERATION_PROMPTS[:n_prompts]

    # 3a: Layer ablation (fixed K=3, multiplier=3.0)
    print("\n  3a: Layer ablation")
    layer_results = {}
    for layer in test_layers:
        sv_dict = extract_steering_vector(model, behavior, layers=[layer])
        results = []
        for prompt in tqdm(prompts, desc=f"    Layer {layer}"):
            r = compute_decay_metrics(
                model, prompt, sv_dict, layer,
                steer_duration=3, post_steer_tokens=20,
                multiplier=3.0, behavior_name=behavior,
            )
            results.append(r)
        layer_results[layer] = results

    # 3b: Duration ablation (fixed layer, multiplier=3.0)
    print("\n  3b: Duration ablation")
    mid_layer = n_layers // 2
    sv_dict = extract_steering_vector(model, behavior, layers=[mid_layer])
    duration_results = {}
    for dur in test_durations:
        results = []
        for prompt in tqdm(prompts, desc=f"    K={dur}"):
            r = compute_decay_metrics(
                model, prompt, sv_dict, mid_layer,
                steer_duration=dur, post_steer_tokens=20,
                multiplier=3.0, behavior_name=behavior,
            )
            results.append(r)
        duration_results[dur] = results

    # 3c: Multiplier ablation (fixed layer, K=3)
    print("\n  3c: Multiplier ablation")
    mult_results = {}
    for mult in test_multipliers:
        results = []
        for prompt in tqdm(prompts, desc=f"    mult={mult}"):
            r = compute_decay_metrics(
                model, prompt, sv_dict, mid_layer,
                steer_duration=3, post_steer_tokens=20,
                multiplier=mult, behavior_name=behavior,
            )
            results.append(r)
        mult_results[mult] = results

    return {
        "layer_ablation": layer_results,
        "duration_ablation": duration_results,
        "multiplier_ablation": mult_results,
    }


def aggregate_measurements(results_list, conditions=None):
    """
    Aggregate measurements across prompts for each condition.
    Returns dict: condition -> {token_pos -> {metric -> (mean, std)}}
    """
    if conditions is None:
        conditions = ["AR", "TF", "TF_clean", "continuous", "no_steer", "random"]

    aggregated = {}
    for cond in conditions:
        pos_to_metrics = {}
        for prompt_results in results_list:
            if cond not in prompt_results:
                continue
            for m in prompt_results[cond].measurements:
                pos = m["token_pos"]
                if pos not in pos_to_metrics:
                    pos_to_metrics[pos] = {k: [] for k in [
                        "cos_sim_to_sv", "proj_onto_sv",
                        "kl_from_steered", "kl_from_unsteered"
                    ]}
                for k in pos_to_metrics[pos]:
                    pos_to_metrics[pos][k].append(m[k])

        # Compute stats
        stats = {}
        for pos, metrics in sorted(pos_to_metrics.items()):
            stats[pos] = {}
            for k, vals in metrics.items():
                arr = np.array(vals)
                stats[pos][k] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "n": len(vals),
                }
        aggregated[cond] = stats

    return aggregated


def save_results(results, filename):
    """Save results to JSON."""
    def convert(obj):
        if isinstance(obj, ExperimentResult):
            return asdict(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    path = RESULTS_DIR / filename
    with open(path, 'w') as f:
        json.dump(results, f, default=convert, indent=2)
    print(f"  Saved results to {path}")


def main():
    """Run all experiments."""
    print("="*70)
    print("STEERING VECTOR DECAY EXPERIMENT")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")

    # Load model
    model = load_model("gpt2-xl")

    # Config
    n_layers = model.cfg.n_layers
    mid_layer = n_layers // 2  # Layer 24 for GPT-2-XL
    behaviors = ["positive_sentiment", "formal_style", "truthful"]

    # ─── Experiment 1: Basic decay curves ───
    exp1_results = run_experiment_1(
        model, behaviors, steer_layer=mid_layer, n_prompts=20
    )

    # Aggregate and save
    exp1_agg = {}
    for beh, results_list in exp1_results.items():
        exp1_agg[beh] = aggregate_measurements(results_list)

    save_results(exp1_agg, "experiment1_decay_curves.json")

    # Save raw results too
    exp1_raw = {}
    for beh, results_list in exp1_results.items():
        exp1_raw[beh] = [
            {cond: asdict(r) for cond, r in prompt_res.items()}
            for prompt_res in results_list
        ]
    save_results(exp1_raw, "experiment1_raw.json")

    # ─── Experiment 2: AR vs TF ───
    exp2_results = run_experiment_2(
        model, "positive_sentiment", steer_layer=mid_layer, n_prompts=25
    )
    exp2_agg = aggregate_measurements(exp2_results)
    save_results(exp2_agg, "experiment2_ar_vs_tf.json")

    # ─── Experiment 3: Ablations ───
    exp3_results = run_experiment_3(
        model, "positive_sentiment", n_prompts=10
    )

    # Aggregate ablation results
    exp3_agg = {}

    # Layer ablation
    exp3_agg["layer_ablation"] = {}
    for layer, results_list in exp3_results["layer_ablation"].items():
        exp3_agg["layer_ablation"][str(layer)] = aggregate_measurements(
            results_list, conditions=["AR", "no_steer"]
        )

    # Duration ablation
    exp3_agg["duration_ablation"] = {}
    for dur, results_list in exp3_results["duration_ablation"].items():
        exp3_agg["duration_ablation"][str(dur)] = aggregate_measurements(
            results_list, conditions=["AR", "no_steer"]
        )

    # Multiplier ablation
    exp3_agg["multiplier_ablation"] = {}
    for mult, results_list in exp3_results["multiplier_ablation"].items():
        exp3_agg["multiplier_ablation"][str(mult)] = aggregate_measurements(
            results_list, conditions=["AR", "no_steer"]
        )

    save_results(exp3_agg, "experiment3_ablations.json")

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Results saved to: {RESULTS_DIR}")

    return exp1_agg, exp2_agg, exp3_agg


if __name__ == "__main__":
    main()
