# Steering Vector Decay

**Research question:** How quickly does a steering vector's influence on model hidden states decay after the vector is no longer applied during generation?

## Key Findings

- **Representational decay is instantaneous (< 1 token).** After removing a steering vector from GPT-2-XL generation, hidden state alignment with the steering direction drops from ~0.019 to ~0.000 in a single forward pass.
- **AR and teacher-forced conditions are identical.** Teacher-forcing the same steered tokens without the vector produces identical hidden states — the vector leaves no residual trace beyond the token sequence.
- **Behavioral persistence operates through tokens, not hidden states.** The KL divergence from the unsteered baseline remains elevated (half-life > 75 tokens) because steered tokens stay in the context, not because hidden states "remember" the perturbation.
- **Decay is invariant to layer, multiplier, and duration.** Whether you steer at layer 8 or 32, with 1x or 5x strength, for 1 or 10 tokens — the representational decay is always instantaneous.
- **Continuous injection is necessary for sustained effects.** This explains why CAA applies vectors at every token position.

## How to Reproduce

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run experiments (~40 min on A6000)
python src/experiment.py

# 3. Generate analysis and plots
python src/analyze.py
python src/plot_key_findings.py
```

### Requirements
- Python 3.10+
- CUDA-capable GPU (tested on NVIDIA RTX A6000, 49GB)
- ~5GB GPU memory for GPT-2-XL in float16

### Dependencies
- PyTorch 2.10+
- TransformerLens
- steering-vectors 0.12.2
- numpy, scipy, matplotlib, scikit-learn

## File Structure

```
steering-vector-decay-4272-claude/
├── REPORT.md                 # Full research report with results
├── README.md                 # This file
├── planning.md               # Research plan and motivation
├── literature_review.md      # Synthesis of 30 related papers
├── resources.md              # Catalog of available resources
├── src/
│   ├── experiment.py         # Main experiment code (all 3 experiments)
│   ├── analyze.py            # Statistical analysis and basic plots
│   └── plot_key_findings.py  # Publication-quality key finding figures
├── results/
│   ├── experiment1_decay_curves.json
│   ├── experiment1_raw.json
│   ├── experiment2_ar_vs_tf.json
│   ├── experiment3_ablations.json
│   ├── statistical_analysis.json
│   └── decay_fits.json
├── figures/
│   ├── key_finding_representational_decay.png
│   ├── key_finding_behavioral_vs_representational.png
│   ├── ablation_summary.png
│   ├── decay_curves_*.png
│   └── ...
├── papers/                   # 30 downloaded reference PDFs
├── datasets/                 # Downloaded evaluation datasets
└── code/                     # 8 cloned reference repositories
```

## Citation

See `REPORT.md` for full references and `literature_review.md` for comprehensive background.
