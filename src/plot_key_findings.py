"""
Create publication-quality figures for key findings.
"""
import os
os.environ.setdefault("USER", "researcher")

import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def load_json(filename):
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)


def extract_series(agg_data, condition, metric):
    positions = []
    means = []
    stds = []
    for pos_str, metrics in sorted(agg_data[condition].items(), key=lambda x: int(x[0])):
        pos = int(pos_str)
        if metric in metrics:
            positions.append(pos)
            means.append(metrics[metric]["mean"])
            stds.append(metrics[metric]["std"])
    return np.array(positions), np.array(means), np.array(stds)


def compute_delta_series(agg_data, cond, baseline_cond, metric):
    """Compute the difference between a condition and baseline at each position."""
    pos1, means1, stds1 = extract_series(agg_data, cond, metric)
    pos2, means2, stds2 = extract_series(agg_data, baseline_cond, metric)
    # Align positions
    common = set(pos1.tolist()) & set(pos2.tolist())
    common = sorted(common)
    m1 = [means1[pos1.tolist().index(p)] for p in common]
    m2 = [means2[pos2.tolist().index(p)] for p in common]
    s1 = [stds1[pos1.tolist().index(p)] for p in common]
    s2 = [stds2[pos2.tolist().index(p)] for p in common]
    delta = np.array(m1) - np.array(m2)
    delta_std = np.sqrt(np.array(s1)**2 + np.array(s2)**2)
    return np.array(common), delta, delta_std


def exp_decay(t, A, lam, C):
    return A * np.exp(-lam * t) + C


# ─── Figure 1: Key Finding - Representational Decay is Instantaneous ─────

def plot_key_finding_1():
    """The central finding: representational decay within 1 token."""
    exp1 = load_json("experiment1_decay_curves.json")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Representational Decay of Steering Vectors After Removal", fontsize=15, y=1.02)

    behaviors = ["truthful", "positive_sentiment", "formal_style"]
    titles = ["Truthful Direction", "Positive Sentiment Direction", "Formal Style Direction"]

    for i, (beh, title) in enumerate(zip(behaviors, titles)):
        ax = axes[i]
        data = exp1[beh]

        # Compute delta (steered - unsteered) for each condition
        for cond, color, label in [
            ("continuous", "#ff7f00", "Continuous Steering"),
            ("AR", "#e41a1c", "Steer 3 tokens, then AR"),
            ("TF_clean", "#4daf4a", "Steer 3, then TF (clean tokens)"),
        ]:
            pos, delta, delta_std = compute_delta_series(
                data, cond, "no_steer", "cos_sim_to_sv"
            )
            ax.plot(pos, delta, color=color, label=label, linewidth=2.5, marker='o', markersize=4)
            ax.fill_between(pos, delta - delta_std, delta + delta_std,
                           color=color, alpha=0.12)

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4)
        ax.set_xlabel("Token position (0 = steering removed)")
        if i == 0:
            ax.set_ylabel("Cos. sim. delta (steered - unsteered)")
        ax.set_title(title)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.2)

        # Annotate the drop
        if beh == "truthful":
            ax.annotate("Immediate decay\n(~1 token)", xy=(0, 0.001),
                        xytext=(5, 0.012), fontsize=9,
                        arrowprops=dict(arrowstyle='->', color='red'),
                        color='red')

    plt.tight_layout()
    path = FIGURES_DIR / "key_finding_representational_decay.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ─── Figure 2: Behavioral Persistence vs Representational Decay ──────────

def plot_key_finding_2():
    """Shows that behavioral effects persist through tokens, not hidden states."""
    exp2 = load_json("experiment2_ar_vs_tf.json")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Behavioral Persistence vs. Representational Decay", fontsize=15, y=1.02)

    # Left: KL from unsteered (shows behavioral persistence)
    ax = axes[0]
    for cond, color, label in [
        ("AR", "#e41a1c", "AR (steered tokens in context)"),
        ("TF", "#377eb8", "TF (same steered tokens, no SV)"),
        ("TF_clean", "#4daf4a", "TF (clean tokens, no SV)"),
        ("continuous", "#ff7f00", "Continuous steering"),
    ]:
        if cond not in exp2:
            continue
        pos, means, stds = extract_series(exp2, cond, "kl_from_unsteered")
        ax.plot(pos, means, color=color, label=label, linewidth=2, marker='o', markersize=3)
        ax.fill_between(pos, means - stds, means + stds, color=color, alpha=0.12)

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Token position (0 = steering removed)")
    ax.set_ylabel("KL divergence from unsteered baseline")
    ax.set_title("Behavioral Divergence from Unsteered Model")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Right: Cosine similarity delta from unsteered (shows representational decay)
    ax = axes[1]
    for cond, color, label in [
        ("AR", "#e41a1c", "AR"),
        ("TF", "#377eb8", "TF (steered tokens)"),
        ("TF_clean", "#4daf4a", "TF (clean tokens)"),
        ("continuous", "#ff7f00", "Continuous steering"),
    ]:
        if cond not in exp2:
            continue
        pos, delta, delta_std = compute_delta_series(
            exp2, cond, "no_steer", "cos_sim_to_sv"
        )
        ax.plot(pos, delta, color=color, label=label, linewidth=2, marker='o', markersize=3)
        ax.fill_between(pos, delta - delta_std, delta + delta_std,
                       color=color, alpha=0.12)

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel("Token position (0 = steering removed)")
    ax.set_ylabel("Cos. sim. delta from unsteered")
    ax.set_title("Representational Effect (Hidden State Alignment)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = FIGURES_DIR / "key_finding_behavioral_vs_representational.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ─── Figure 3: Ablation Summary ─────────────────────────────────────────

def plot_ablation_summary():
    """Summarize ablation results in a compact figure."""
    exp3 = load_json("experiment3_ablations.json")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Ablation Studies: Steering Vector Decay Dynamics", fontsize=15)

    # Plot 1: Layer ablation
    ax = axes[0]
    layer_data = exp3.get("layer_ablation", {})
    cmap = plt.cm.viridis
    layers = sorted(layer_data.keys(), key=int)
    for i, layer in enumerate(layers):
        data = layer_data[layer]
        if "AR" not in data:
            continue
        pos, delta, delta_std = compute_delta_series(
            data, "AR", "no_steer", "cos_sim_to_sv"
        )
        color = cmap(i / max(1, len(layers)-1))
        ax.plot(pos, delta, color=color, label=f"Layer {layer}", linewidth=2)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Cos. sim. delta from unsteered")
    ax.set_title("By Injection Layer")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Plot 2: Duration ablation
    ax = axes[1]
    dur_data = exp3.get("duration_ablation", {})
    cmap = plt.cm.plasma
    durs = sorted(dur_data.keys(), key=int)
    for i, dur in enumerate(durs):
        data = dur_data[dur]
        if "AR" not in data:
            continue
        pos, delta, delta_std = compute_delta_series(
            data, "AR", "no_steer", "cos_sim_to_sv"
        )
        color = cmap(i / max(1, len(durs)-1))
        ax.plot(pos, delta, color=color, label=f"K={dur}", linewidth=2)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Cos. sim. delta from unsteered")
    ax.set_title("By Steering Duration (K)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Plot 3: Multiplier ablation
    ax = axes[2]
    mult_data = exp3.get("multiplier_ablation", {})
    cmap = plt.cm.inferno
    mults = sorted(mult_data.keys(), key=float)
    for i, mult in enumerate(mults):
        data = mult_data[mult]
        if "AR" not in data:
            continue
        pos, delta, delta_std = compute_delta_series(
            data, "AR", "no_steer", "cos_sim_to_sv"
        )
        color = cmap(i / max(1, len(mults)-1))
        ax.plot(pos, delta, color=color, label=f"mult={mult}", linewidth=2)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Cos. sim. delta from unsteered")
    ax.set_title("By Steering Multiplier")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = FIGURES_DIR / "ablation_summary.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ─── Figure 4: Decay Halflife Summary Bar Chart ─────────────────────────

def plot_halflife_summary():
    """Bar chart summarizing decay half-lives."""
    exp1 = load_json("experiment1_decay_curves.json")

    behaviors = ["truthful", "positive_sentiment", "formal_style"]
    conditions = ["AR", "continuous"]
    metric = "cos_sim_to_sv"

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(behaviors))
    width = 0.35

    for i, cond in enumerate(conditions):
        halflives = []
        for beh in behaviors:
            data = exp1[beh]
            pos, delta, _ = compute_delta_series(data, cond, "no_steer", metric)
            # Compute half-life from delta series
            mask = pos >= 0
            t = pos[mask].astype(float)
            y = np.abs(delta[mask])
            if len(y) > 2 and y[0] > 0:
                try:
                    popt, _ = curve_fit(exp_decay, t, y,
                                       p0=[y[0], 0.5, 0],
                                       maxfev=5000,
                                       bounds=([0, 0, -np.inf], [np.inf, 10, np.inf]))
                    hl = np.log(2) / popt[1] if popt[1] > 0 else 50
                    halflives.append(min(hl, 50))
                except:
                    halflives.append(0)
            else:
                halflives.append(0)

        bars = ax.bar(x + i * width, halflives, width,
                      label=cond.replace("_", " ").title(),
                      alpha=0.8)

    ax.set_xlabel("Behavior")
    ax.set_ylabel("Decay Half-Life (tokens)")
    ax.set_title("Representational Decay Half-Life by Behavior and Condition")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([b.replace("_", " ").title() for b in behaviors])
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    path = FIGURES_DIR / "halflife_summary.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating key finding visualizations...")
    plot_key_finding_1()
    plot_key_finding_2()
    plot_ablation_summary()
    plot_halflife_summary()
    print("Done.")
