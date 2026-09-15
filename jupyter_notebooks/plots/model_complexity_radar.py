"""
Model Complexity Radar Chart  —  Fig 11c
=========================================
Single-polygon radar where each spoke = one model,
value = Relative Complexity Score (0–100).

Score basis (per model, on this dataset 42–36 samples, ~18 features):
  Training-time Big-O · number of tuneable hyperparameters ·
  memory footprint · black-box vs interpretable trade-off

Output: plots/Fig11c_ModelComplexity_Radar.png
        (does NOT override any existing Fig11a/Fig11b files)
"""

import os, numpy as np, matplotlib.pyplot as plt
import matplotlib.patheffects as pe

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = r'D:\My Projects\Py-DS-ML-Bootcamp-master\research paper'
OUT = os.path.join(_HERE, "plots")
os.makedirs(OUT, exist_ok=True)

# ── Relative complexity scores (0–100 scale) ──────────────────────────────────
# Higher = more complex (slower, more hyperparameters, less interpretable)
MODELS_SCORES = {
    "Decision\nTree":      22,   # O(n log n) train; ~5 params; fully interpretable
    "K-Means":             30,   # O(n·k·I); 2 key params; semi-interpretable
    "KNN":                 45,   # No training; O(n·d) per query; 2 params (k, metric)
    "Random\nForest":      70,   # 100 trees × DT; 8+ params; black-box
    "SVM":                 75,   # O(n²)–O(n³) kernel; 3–4 params; black-box
    "Gradient\nBoosting":  82,   # 100 sequential estimators; 8+ params; black-box
    "XGBoost":             88,   # Optimised GB + L1/L2 reg.; 12+ params; black-box
}

MODELS = list(MODELS_SCORES.keys())
VALUES = list(MODELS_SCORES.values())
N      = len(MODELS)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
vals   = VALUES + [VALUES[0]]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8.5),
                        subplot_kw=dict(polar=True),
                        facecolor="white")
fig.patch.set_facecolor("white")

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(MODELS, fontsize=11, fontweight="bold", color="#1a252f")
ax.set_rlabel_position(18)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20", "40", "60", "80", "100"],
                   fontsize=8, color="#777")
ax.set_ylim(0, 112)
ax.grid(color="#ccc", linestyle="--", linewidth=0.6, alpha=0.70)
ax.spines["polar"].set_color("#bbb")

FILL_COLOR = "#1565C0"   # deep blue matching paper color scheme

# Filled polygon
ax.plot(angles, vals, color=FILL_COLOR, linewidth=2.5, linestyle="-",
        zorder=3, solid_capstyle="round", solid_joinstyle="round")
ax.fill(angles, vals, color=FILL_COLOR, alpha=0.18, zorder=2)

# Vertex dots + score labels
for a, v, name in zip(angles[:-1], VALUES, MODELS):
    ax.plot(a, v, "o", ms=10, color=FILL_COLOR,
            mec="white", mew=1.8, zorder=4)
    # Score label just outside the dot
    ax.text(a, v + 8, str(v),
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=FILL_COLOR, zorder=5)

# ── Title & footnote ──────────────────────────────────────────────────────────
fig.suptitle(
    "Model Complexity Radar Chart\n"
    "Relative Algorithmic Complexity Score (0 – 100)",
    fontsize=13, fontweight="bold", color="#1a252f", y=1.03)

# Footnote below the polar axes
fig.text(
    0.50, 0.02,
    "Score = composite of: training-time complexity  ·  hyperparameter count  ·"
    "  memory  ·  interpretability (inverted)",
    ha="center", fontsize=7.5, color="#666", style="italic")

out_path = os.path.join(OUT, "Fig11c_ModelComplexity_Radar.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_path}")
