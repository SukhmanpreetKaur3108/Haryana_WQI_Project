"""
Fig 11 — Radar Chart: Fixed Legend + Separate Individual Model PNGs
====================================================================
Outputs:
  plots/Fig11a_Radar_Combined.png        — all models overlay (legend outside)
  plots/Fig11b_Radar_<ModelName>.png     — one PNG per model (7 files)
"""

import os, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
warnings.filterwarnings("ignore")

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = r'D:\My Projects\Py-DS-ML-Bootcamp-master\research paper'
OUT = os.path.join(_HERE, "plots")
os.makedirs(OUT, exist_ok=True)

# ── Metrics (from paper tables) ────────────────────────────────────────────
CATEGORIES = ["Accuracy", "Precision", "Recall", "F1", "MCC"]
N = len(CATEGORIES)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # close the loop

MODELS = {
    "KNN":               {"GW": [1.000,1.000,1.000,1.000,1.000], "SW": [0.778,0.790,0.778,0.781,0.715]},
    "Random Forest":     {"GW": [0.778,0.800,0.778,0.784,0.715], "SW": [0.889,0.900,0.889,0.891,0.862]},
    "SVM":               {"GW": [0.889,0.900,0.889,0.891,0.862], "SW": [0.333,0.360,0.333,0.319,0.183]},
    "Gradient Boosting": {"GW": [0.778,0.790,0.778,0.781,0.712], "SW": [0.778,0.790,0.778,0.781,0.715]},
    "Decision Tree":     {"GW": [0.778,0.780,0.778,0.775,0.710], "SW": [0.889,0.900,0.889,0.891,0.862]},
    "XGBoost":           {"GW": [0.667,0.670,0.667,0.661,0.578], "SW": [0.778,0.790,0.778,0.781,0.715]},
    "K-Means":           {"GW": [0.444,0.580,0.611,0.571,0.450], "SW": [0.444,0.530,0.556,0.527,0.380]},
}

COLORS = {
    "KNN":               "#1565C0",
    "Random Forest":     "#2E7D32",
    "SVM":               "#B71C1C",
    "Gradient Boosting": "#E65100",
    "Decision Tree":     "#4A148C",
    "XGBoost":           "#F9A825",
    "K-Means":           "#00695C",
}

def radar_axes(ax, title="", subtitle=""):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATEGORIES, fontsize=9, fontweight="bold", color="#333")
    ax.set_rlabel_position(30)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7, color="#777")
    ax.set_ylim(0, 1.0)
    ax.grid(color="#ccc", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines["polar"].set_color("#bbb")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=16, color="#222")
    if subtitle:
        ax.text(0, 1.22, subtitle, transform=ax.transAxes,
                ha="center", fontsize=8, color="#555", style="italic")

def plot_model_on_ax(ax, mname, data_gw, data_sw, alpha=0.85, lw=2.0, show_legend=True):
    color = COLORS[mname]
    # GW — solid
    vals_gw = data_gw + [data_gw[0]]
    ax.plot(angles, vals_gw, color=color, linewidth=lw, linestyle="-",
            label=f"{mname} — GW")
    ax.fill(angles, vals_gw, color=color, alpha=0.12)
    # SW — dashed
    vals_sw = data_sw + [data_sw[0]]
    ax.plot(angles, vals_sw, color=color, linewidth=lw, linestyle="--",
            label=f"{mname} — SW")
    ax.fill(angles, vals_sw, color=color, alpha=0.06)
    if show_legend:
        leg = ax.legend(loc="upper right", bbox_to_anchor=(1.52, 1.18),
                        fontsize=7.5, framealpha=0.9, edgecolor="#ccc",
                        title=mname, title_fontsize=8)

# ═══════════════════════════════════════════════════════════════════
# Fig 11a — COMBINED overlay (all models, legend placed OUTSIDE plot)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True),
                        facecolor="white")
radar_axes(ax, title="Model Complexity — All Classifiers\n(GW solid · SW dashed)")

handles, labels_ = [], []
for mname, vals in MODELS.items():
    color = COLORS[mname]
    vals_gw = vals["GW"] + [vals["GW"][0]]
    vals_sw = vals["SW"] + [vals["SW"][0]]
    ax.plot(angles, vals_gw, color=color, linewidth=2.0, linestyle="-")
    ax.fill(angles, vals_gw, color=color, alpha=0.07)
    ax.plot(angles, vals_sw, color=color, linewidth=2.0, linestyle="--")
    ax.fill(angles, vals_sw, color=color, alpha=0.04)
    # Legend handles (one per model, show color)
    import matplotlib.lines as mlines
    h = mlines.Line2D([], [], color=color, linewidth=2, label=mname)
    handles.append(h)
    labels_.append(mname)

# GW vs SW style indicators
gw_h = mlines.Line2D([], [], color="#333", linewidth=2, linestyle="-",  label="Groundwater (solid)")
sw_h = mlines.Line2D([], [], color="#333", linewidth=2, linestyle="--", label="Surface Water (dashed)")

# Legend OUTSIDE the polar axes — use figure-level legend
fig.legend(handles=handles + [gw_h, sw_h],
           labels=labels_ + ["── Groundwater", "-- Surface Water"],
           loc="center left", bbox_to_anchor=(0.78, 0.50),
           fontsize=8.5, framealpha=0.95, edgecolor="#ccc",
           title="Models", title_fontsize=9)

plt.tight_layout(rect=[0, 0, 0.74, 1])
path = os.path.join(OUT, "Fig11a_Radar_Combined.png")
plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: Fig11a_Radar_Combined.png")

# ═══════════════════════════════════════════════════════════════════
# Fig 11b–h — INDIVIDUAL model radars (one PNG each)
# ═══════════════════════════════════════════════════════════════════
for idx, (mname, vals) in enumerate(MODELS.items()):
    fig, ax = plt.subplots(figsize=(7.5, 7.0), subplot_kw=dict(polar=True),
                            facecolor="white")
    radar_axes(ax, title=f"{mname}\nClassifier Performance Radar")

    color = COLORS[mname]
    vals_gw = vals["GW"] + [vals["GW"][0]]
    vals_sw = vals["SW"] + [vals["SW"][0]]

    ax.plot(angles, vals_gw, color=color, linewidth=2.5, linestyle="-",  label="Groundwater")
    ax.fill(angles, vals_gw, color=color, alpha=0.20)
    ax.plot(angles, vals_sw, color=color, linewidth=2.5, linestyle="--", label="Surface Water")
    ax.fill(angles, vals_sw, color=color, alpha=0.08)

    # Compact metrics table below the radar (replaces vertex annotations that overlap)
    col_header = f"{'':10s}" + "  ".join(f"{c:>5s}" for c in CATEGORIES)
    gw_row     = f"{'GW':10s}" + "  ".join(f"{v:>5.2f}" for v in vals["GW"])
    sw_row     = f"{'SW':10s}" + "  ".join(f"{v:>5.2f}" for v in vals["SW"])
    table_str  = f"{col_header}\n{gw_row}\n{sw_row}"

    ax.text(0.50, -0.14, table_str,
            transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, fontfamily="monospace",
            color="#1a252f",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=color, alpha=0.93, linewidth=1.0))

    # Compact GW/SW line-style legend (no text collision)
    import matplotlib.lines as mlines
    gw_line = mlines.Line2D([], [], color=color, linewidth=2.0, linestyle="-",  label="Groundwater (GW)")
    sw_line = mlines.Line2D([], [], color=color, linewidth=2.0, linestyle="--", label="Surface Water (SW)")
    ax.legend(handles=[gw_line, sw_line],
              loc="upper right", bbox_to_anchor=(1.50, 1.05),
              ncol=1, fontsize=8.5, framealpha=0.95, edgecolor="#ccc",
              title=mname, title_fontsize=8)

    plt.tight_layout()
    safe_name = mname.replace(" ", "_").replace("/", "_")
    path = os.path.join(OUT, f"Fig11b_Radar_{safe_name}.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: Fig11b_Radar_{safe_name}.png")

print("\nAll Fig 11 radar charts saved.")
