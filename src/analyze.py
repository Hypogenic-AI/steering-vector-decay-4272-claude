"""
Analysis and Visualization for Steering Vector Decay Experiments
================================================================
Produces plots and statistical analyses from experimental results.
"""

import os
os.environ.setdefault("USER", "researcher")

import json
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_json(filename):
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)


def extract_series(agg_data, condition, metric):
    """Extract a time series (positions, means, stds) from aggregated data."""
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


def exp_decay(t, A, lam, C):
    """Exponential decay model: A * exp(-lam * t) + C"""
    return A * np.exp(-lam * t) + C


def fit_decay_curve(positions, values, metric_name=""):
    """Fit exponential decay to post-steering measurements."""
    # Only fit post-steering data (pos >= 0)
    mask = positions >= 0
    t = positions[mask].astype(float)
    y = values[mask]

    if len(t) < 3:
        return None

    # Initial guess
    A0 = y[0] - y[-1] if len(y) > 1 else y[0]
    C0 = y[-1] if len(y) > 1 else 0
    lam0 = 0.2

    try:
        popt, pcov = curve_fit(
            exp_decay, t, y,
            p0=[A0, lam0, C0],
            maxfev=5000,
            bounds=([-np.inf, 0, -np.inf], [np.inf, 10, np.inf])
        )
        y_pred = exp_decay(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        half_life = np.log(2) / popt[1] if popt[1] > 0 else float('inf')

        return {
            "A": float(popt[0]),
            "lambda": float(popt[1]),
            "C": float(popt[2]),
            "r_squared": float(r_squared),
            "half_life": float(half_life),
            "perr": np.sqrt(np.diag(pcov)).tolist() if pcov is not None else None,
        }
    except Exception as e:
        print(f"  Fit failed for {metric_name}: {e}")
        return None


# ─── Plot 1: Main Decay Curves ──────────────────────────────────────────────

def plot_decay_curves(exp1_data, behavior="positive_sentiment"):
    """Plot main decay curves for all conditions."""
    data = exp1_data[behavior]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Steering Vector Decay: {behavior.replace('_', ' ').title()}", fontsize=15)

    metrics = [
        ("cos_sim_to_sv", "Cosine Similarity to Steering Vector"),
        ("proj_onto_sv", "Projection onto Steering Direction"),
        ("kl_from_steered", "KL Divergence from Steered Baseline"),
        ("kl_from_unsteered", "KL Divergence from Unsteered Baseline"),
    ]

    conditions_to_plot = ["AR", "TF", "TF_clean", "continuous", "no_steer", "random"]
    colors = {
        "AR": "#e41a1c",
        "TF": "#377eb8",
        "TF_clean": "#4daf4a",
        "continuous": "#ff7f00",
        "no_steer": "#999999",
        "random": "#984ea3",
    }
    labels = {
        "AR": "Autoregressive",
        "TF": "Teacher-Forced (AR tokens)",
        "TF_clean": "Teacher-Forced (clean tokens)",
        "continuous": "Continuous Steering",
        "no_steer": "No Steering (baseline)",
        "random": "Random Vector",
    }

    for ax_idx, (metric, title) in enumerate(metrics):
        ax = axes[ax_idx // 2][ax_idx % 2]

        for cond in conditions_to_plot:
            if cond not in data:
                continue
            pos, means, stds = extract_series(data, cond, metric)
            if len(pos) == 0:
                continue

            ax.plot(pos, means, color=colors[cond], label=labels[cond],
                    linewidth=2, alpha=0.9)
            ax.fill_between(pos, means - stds, means + stds,
                           color=colors[cond], alpha=0.15)

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Steering removed')
        ax.set_xlabel("Token Position (relative to steering removal)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / f"decay_curves_{behavior}.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─── Plot 2: AR vs TF Comparison ────────────────────────────────────────────

def plot_ar_vs_tf(exp2_data):
    """Focused comparison of AR vs TF decay rates."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Autoregressive vs. Teacher-Forced Decay Comparison", fontsize=14)

    metrics = [
        ("cos_sim_to_sv", "Cosine Sim. to SV"),
        ("proj_onto_sv", "Projection onto SV"),
        ("kl_from_unsteered", "KL from Unsteered"),
    ]

    colors = {"AR": "#e41a1c", "TF": "#377eb8", "TF_clean": "#4daf4a"}
    labels = {"AR": "Autoregressive", "TF": "Teacher-Forced (AR tokens)", "TF_clean": "Teacher-Forced (clean)"}

    fit_results = {}
    for ax_idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]

        for cond in ["AR", "TF", "TF_clean"]:
            if cond not in exp2_data:
                continue
            pos, means, stds = extract_series(exp2_data, cond, metric)
            if len(pos) == 0:
                continue

            ax.plot(pos, means, color=colors[cond], label=labels[cond],
                    linewidth=2, marker='o', markersize=3)
            ax.fill_between(pos, means - stds, means + stds,
                           color=colors[cond], alpha=0.15)

            # Fit decay curve
            fit = fit_decay_curve(pos, means, f"{cond}_{metric}")
            if fit:
                fit_results[f"{cond}_{metric}"] = fit
                # Plot fitted curve
                post_mask = pos >= 0
                t_fit = np.linspace(0, pos[post_mask].max(), 100)
                y_fit = exp_decay(t_fit, fit["A"], fit["lambda"], fit["C"])
                ax.plot(t_fit, y_fit, color=colors[cond], linestyle='--', alpha=0.5)

                # Annotate half-life
                if fit["half_life"] < 100:
                    ax.annotate(
                        f't½={fit["half_life"]:.1f}',
                        xy=(fit["half_life"], exp_decay(fit["half_life"], fit["A"], fit["lambda"], fit["C"])),
                        fontsize=8, color=colors[cond],
                    )

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel("Token Position (rel. to steering removal)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "ar_vs_tf_comparison.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    return fit_results


# ─── Plot 3: Layer Ablation ─────────────────────────────────────────────────

def plot_layer_ablation(exp3_data):
    """Plot decay curves across different steering layers."""
    layer_data = exp3_data.get("layer_ablation", {})
    if not layer_data:
        print("  No layer ablation data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Decay Rate by Steering Layer", fontsize=14)

    cmap = plt.cm.viridis
    layers = sorted(layer_data.keys(), key=lambda x: int(x))
    colors = {l: cmap(i / max(1, len(layers)-1)) for i, l in enumerate(layers)}

    for metric_idx, (metric, ylabel) in enumerate([
        ("cos_sim_to_sv", "Cosine Sim. to SV"),
        ("proj_onto_sv", "Projection onto SV"),
    ]):
        ax = axes[metric_idx]
        for layer_str in layers:
            data = layer_data[layer_str]
            if "AR" not in data:
                continue
            pos, means, stds = extract_series(data, "AR", metric)
            if len(pos) == 0:
                continue
            ax.plot(pos, means, color=colors[layer_str],
                    label=f"Layer {layer_str}", linewidth=2)
            ax.fill_between(pos, means-stds, means+stds,
                           color=colors[layer_str], alpha=0.12)

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel("Token Position")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by Layer")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "layer_ablation.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 4: Duration Ablation ──────────────────────────────────────────────

def plot_duration_ablation(exp3_data):
    """Plot decay curves for different steering durations."""
    dur_data = exp3_data.get("duration_ablation", {})
    if not dur_data:
        print("  No duration ablation data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Decay by Steering Duration (K)", fontsize=14)

    cmap = plt.cm.plasma
    durations = sorted(dur_data.keys(), key=lambda x: int(x))
    colors = {d: cmap(i / max(1, len(durations)-1)) for i, d in enumerate(durations)}

    for metric_idx, (metric, ylabel) in enumerate([
        ("cos_sim_to_sv", "Cosine Sim. to SV"),
        ("proj_onto_sv", "Projection onto SV"),
    ]):
        ax = axes[metric_idx]
        for dur_str in durations:
            data = dur_data[dur_str]
            if "AR" not in data:
                continue
            pos, means, stds = extract_series(data, "AR", metric)
            if len(pos) == 0:
                continue
            ax.plot(pos, means, color=colors[dur_str],
                    label=f"K={dur_str}", linewidth=2)
            ax.fill_between(pos, means-stds, means+stds,
                           color=colors[dur_str], alpha=0.12)

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel("Token Position")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by Duration")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "duration_ablation.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 5: Multiplier Ablation ────────────────────────────────────────────

def plot_multiplier_ablation(exp3_data):
    """Plot decay curves for different multiplier strengths."""
    mult_data = exp3_data.get("multiplier_ablation", {})
    if not mult_data:
        print("  No multiplier ablation data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Decay by Steering Strength (Multiplier)", fontsize=14)

    cmap = plt.cm.inferno
    mults = sorted(mult_data.keys(), key=lambda x: float(x))
    colors = {m: cmap(i / max(1, len(mults)-1)) for i, m in enumerate(mults)}

    for metric_idx, (metric, ylabel) in enumerate([
        ("cos_sim_to_sv", "Cosine Sim. to SV"),
        ("proj_onto_sv", "Projection onto SV"),
    ]):
        ax = axes[metric_idx]
        for mult_str in mults:
            data = mult_data[mult_str]
            if "AR" not in data:
                continue
            pos, means, stds = extract_series(data, "AR", metric)
            if len(pos) == 0:
                continue
            ax.plot(pos, means, color=colors[mult_str],
                    label=f"mult={mult_str}", linewidth=2)
            ax.fill_between(pos, means-stds, means+stds,
                           color=colors[mult_str], alpha=0.12)

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel("Token Position")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by Multiplier")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "multiplier_ablation.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 6: Cross-behavior comparison ──────────────────────────────────────

def plot_behavior_comparison(exp1_data):
    """Compare decay across behaviors."""
    behaviors = list(exp1_data.keys())
    if not behaviors:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Decay Rate Comparison Across Behaviors", fontsize=14)

    cmap = plt.cm.Set1
    colors = {b: cmap(i / max(1, len(behaviors)-1)) for i, b in enumerate(behaviors)}

    for metric_idx, (metric, ylabel) in enumerate([
        ("cos_sim_to_sv", "Cosine Sim. to SV"),
        ("proj_onto_sv", "Projection onto SV"),
    ]):
        ax = axes[metric_idx]
        for beh in behaviors:
            data = exp1_data[beh]
            if "AR" not in data:
                continue
            pos, means, stds = extract_series(data, "AR", metric)
            if len(pos) == 0:
                continue
            ax.plot(pos, means, color=colors[beh],
                    label=beh.replace('_', ' ').title(), linewidth=2)
            ax.fill_between(pos, means-stds, means+stds,
                           color=colors[beh], alpha=0.12)

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel("Token Position")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by Behavior")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "behavior_comparison.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ─── Statistical Tests ──────────────────────────────────────────────────────

def statistical_analysis(exp2_data):
    """Run statistical tests comparing AR vs TF decay rates."""
    print("\n  Statistical Analysis: AR vs TF Decay")
    print("  " + "="*50)

    results = {}
    metrics = ["cos_sim_to_sv", "proj_onto_sv", "kl_from_unsteered"]

    for metric in metrics:
        print(f"\n  Metric: {metric}")

        # Fit decay curves for each condition
        fits = {}
        for cond in ["AR", "TF", "TF_clean"]:
            if cond not in exp2_data:
                continue
            pos, means, stds = extract_series(exp2_data, cond, metric)
            fit = fit_decay_curve(pos, means, f"{cond}_{metric}")
            if fit:
                fits[cond] = fit
                print(f"    {cond}: lambda={fit['lambda']:.4f}, "
                      f"half_life={fit['half_life']:.2f}, R²={fit['r_squared']:.4f}")

        # Compare AR vs TF using post-steering values
        for cond1, cond2 in [("AR", "TF"), ("AR", "TF_clean"), ("TF", "TF_clean")]:
            if cond1 not in exp2_data or cond2 not in exp2_data:
                continue
            pos1, means1, _ = extract_series(exp2_data, cond1, metric)
            pos2, means2, _ = extract_series(exp2_data, cond2, metric)

            # Use post-steering values
            mask1 = pos1 >= 0
            mask2 = pos2 >= 0
            v1 = means1[mask1]
            v2 = means2[mask2]
            min_len = min(len(v1), len(v2))
            if min_len < 3:
                continue

            t_stat, p_val = stats.ttest_rel(v1[:min_len], v2[:min_len])
            d = (np.mean(v1[:min_len]) - np.mean(v2[:min_len])) / np.std(v1[:min_len] - v2[:min_len]) if np.std(v1[:min_len] - v2[:min_len]) > 0 else 0

            print(f"    {cond1} vs {cond2}: t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d:.3f}")
            results[f"{cond1}_vs_{cond2}_{metric}"] = {
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(d),
            }

        results[f"fits_{metric}"] = fits

    return results


# ─── Summary Table ───────────────────────────────────────────────────────────

def generate_summary_table(exp1_data, exp2_data):
    """Generate summary table of decay half-lives."""
    print("\n  Summary Table: Decay Half-Lives")
    print("  " + "="*70)
    print(f"  {'Behavior':<20} {'Condition':<15} {'Metric':<18} {'Half-life':<10} {'R²':<8}")
    print("  " + "-"*70)

    all_fits = {}
    for beh in exp1_data:
        data = exp1_data[beh]
        for cond in ["AR", "TF", "TF_clean"]:
            if cond not in data:
                continue
            for metric in ["cos_sim_to_sv", "proj_onto_sv"]:
                pos, means, stds = extract_series(data, cond, metric)
                fit = fit_decay_curve(pos, means, f"{beh}_{cond}_{metric}")
                if fit:
                    key = f"{beh}_{cond}_{metric}"
                    all_fits[key] = fit
                    hl = f"{fit['half_life']:.2f}" if fit['half_life'] < 100 else ">100"
                    print(f"  {beh:<20} {cond:<15} {metric:<18} {hl:<10} {fit['r_squared']:.4f}")

    return all_fits


def main():
    """Run all analysis and generate plots."""
    print("="*70)
    print("ANALYSIS AND VISUALIZATION")
    print("="*70)

    # Load results
    try:
        exp1_data = load_json("experiment1_decay_curves.json")
    except FileNotFoundError:
        print("ERROR: experiment1_decay_curves.json not found")
        return

    try:
        exp2_data = load_json("experiment2_ar_vs_tf.json")
    except FileNotFoundError:
        print("WARNING: experiment2_ar_vs_tf.json not found")
        exp2_data = None

    try:
        exp3_data = load_json("experiment3_ablations.json")
    except FileNotFoundError:
        print("WARNING: experiment3_ablations.json not found")
        exp3_data = None

    # ─── Generate plots ───
    print("\nGenerating plots...")

    # Exp 1 plots
    for behavior in exp1_data:
        plot_decay_curves(exp1_data, behavior)

    plot_behavior_comparison(exp1_data)

    # Exp 2 plots
    if exp2_data:
        fit_results = plot_ar_vs_tf(exp2_data)
        stats_results = statistical_analysis(exp2_data)
        save_json = {
            "fit_results": fit_results,
            "statistical_tests": stats_results,
        }
        with open(RESULTS_DIR / "statistical_analysis.json", 'w') as f:
            json.dump(save_json, f, indent=2, default=float)
        print(f"  Saved statistical analysis to {RESULTS_DIR / 'statistical_analysis.json'}")

    # Exp 3 plots
    if exp3_data:
        plot_layer_ablation(exp3_data)
        plot_duration_ablation(exp3_data)
        plot_multiplier_ablation(exp3_data)

    # ─── Summary ───
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_fits = generate_summary_table(exp1_data, exp2_data if exp2_data else {})

    if exp2_data:
        print("\n  Key Finding: AR vs TF Decay Rate Comparison")
        stats_results = statistical_analysis(exp2_data)

    # Save all fit results
    with open(RESULTS_DIR / "decay_fits.json", 'w') as f:
        json.dump(all_fits, f, indent=2, default=float)

    print(f"\n  All figures saved to: {FIGURES_DIR}")
    print(f"  All results saved to: {RESULTS_DIR}")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
