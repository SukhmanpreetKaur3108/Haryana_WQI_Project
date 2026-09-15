"""
SHAP Analysis — Best Models per Water Type
============================================
Groundwater  → KNN         (best classifier, 5-Fold CV Acc = 0.987 ± 0.027)
Surface Water → Decision Tree (best classifier, 5-Fold CV Acc = 0.996 ± 0.008)

Outputs (plots/shap/):
  SHAP_GW_KNN_beeswarm.png          — all parameters, beeswarm summary
  SHAP_GW_KNN_bar.png               — mean |SHAP| bar chart
  SHAP_SW_DT_beeswarm.png           — all parameters, beeswarm summary
  SHAP_SW_DT_bar.png                — mean |SHAP| bar chart
  SHAP_SW_DT_interaction.png        — interaction matrix (Decision Tree only)

Notes:
  - KNN uses shap.KernelExplainer (sampling-based, approx. ~2 min on small data)
  - Decision Tree uses shap.TreeExplainer (exact, fast)
  - Data is averaged implicitly via augmentation pipeline (Gaussian×3 + SMOTE)
  - All available physicochemical parameters are included
"""

import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")
shap.initjs()

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = r'D:\My Projects\Py-DS-ML-Bootcamp-master\research paper'
BASE = os.path.dirname(_HERE)
OUT  = os.path.join(_HERE, "plots", "shap")
os.makedirs(OUT, exist_ok=True)

# ── Pretty feature names (displayed on SHAP plots) ────────────────────────────
FEATURE_LABELS = {
    "ph":                           "pH",
    "turbidity":                    "Turbidity (NTU)",
    "conductivity":                 "Conductivity (µS/cm)",
    "chloride_(ppm)":               "Chloride (ppm)",
    "sulphates(ppm)":               "Sulphates (ppm)",
    "iron_(ppm)":                   "Iron (ppm)",
    "cod(ppm)":                     "COD (ppm)",
    "bod(ppm)":                     "BOD (ppm)",
    "do(ppm)":                      "DO (ppm)",
    "ammonia(ppm)":                 "Ammonia (ppm)",
    "nitrate(ppm)":                 "Nitrate (ppm)",
    "total_bacterial_count_(cfu/ml)":"Total Bacterial Count (CFU/mL)",
    "total_fungal_count_(cfu/ml)":  "Total Fungal Count (CFU/mL)",
    "fluorides(ppm)":               "Fluorides (ppm)",
    "phosphates(ppm)":              "Phosphates (ppm)",
    # Surface Water columns
    "pH":                           "pH",
    "Turbidity":                    "Turbidity (NTU)",
    "Conductivity":                 "Conductivity (µS/cm)",
    "Chloride (ppm)":               "Chloride (ppm)",
    "Sulphates(ppm)":               "Sulphates (ppm)",
    "Iron (ppm)":                   "Iron (ppm)",
    "COD(ppm)":                     "COD (ppm)",
    "BOD(ppm)":                     "BOD (ppm)",
    "DO(ppm)":                      "DO (ppm)",
    "Ammonia(ppm)":                 "Ammonia (ppm)",
    "Nitrate(ppm)":                 "Nitrate (ppm)",
    "Total Bacterial Count (cfu/ml)":"Total Bacterial Count (CFU/mL)",
    "Total Fungal Count (cfu/ml)":  "Total Fungal Count (CFU/mL)",
    "Fluorides(ppm)":               "Fluorides (ppm)",
    "Phosphates(ppm)":              "Phosphates (ppm)",
}

# ── Data loading & pipeline ────────────────────────────────────────────────────
gw = pd.read_csv(os.path.join(BASE, "groundwater_wqi.csv"))
sw = pd.read_csv(os.path.join(BASE, "surface_wqi.csv"))
sw.columns = [c.strip() for c in sw.columns]

GW_PARAMS = [c for c in [
    "ph","turbidity","conductivity","chloride_(ppm)","sulphates(ppm)",
    "iron_(ppm)","cod(ppm)","bod(ppm)","do(ppm)","ammonia(ppm)","nitrate(ppm)",
    "fluorides(ppm)","phosphates(ppm)",
    "total_bacterial_count_(cfu/ml)","total_fungal_count_(cfu/ml)"]
    if c in gw.columns]

SW_PARAMS = [c for c in [
    "pH","Turbidity","Conductivity","Chloride (ppm)","Sulphates(ppm)",
    "Iron (ppm)","COD(ppm)","BOD(ppm)","DO(ppm)","Ammonia(ppm)","Nitrate(ppm)",
    "Fluorides(ppm)","Phosphates(ppm)",
    "Total Bacterial Count (cfu/ml)","Total Fungal Count (cfu/ml)"]
    if c in sw.columns]

GW_CLASSES = ["Excellent","Good","Poor","Very Poor","Unsuitable"]
SW_CLASSES = ["Medium","Poor","Very Poor","Unsuitable"]

def encode_y(y, classes):
    m = {c: i for i, c in enumerate(classes)}
    return y.map(lambda v: m.get(v, -1))

def remap_consecutive(y):
    uniq = sorted(y.unique())
    remap = {v: i for i, v in enumerate(uniq)}
    return y.map(remap), remap

def augment_smote_pipeline(X, y, seed=42):
    X = X.copy().astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                            random_state=seed, stratify=y)
    rng = np.random.RandomState(seed)
    stds = Xtr.std()
    copies = [Xtr.copy()]
    for _ in range(3):
        Xn = Xtr.copy()
        for col in Xtr.columns:
            s = stds[col]
            if s == 0 or np.isnan(s): continue
            noise = rng.normal(0, 0.01*s, len(Xtr))
            mask = Xn[col].notna()
            Xn.loc[mask, col] += noise[mask]
        copies.append(Xn)
    Xa = pd.concat(copies, ignore_index=True)
    ya = pd.concat([ytr]*4,  ignore_index=True)
    imp = KNNImputer(n_neighbors=5); sc = StandardScaler()
    Xa_sc = sc.fit_transform(imp.fit_transform(Xa))
    Xte_sc = sc.transform(imp.transform(Xte))
    k_sm = max(1, min(5, pd.Series(ya).value_counts().min()-1))
    sm = SMOTE(random_state=seed, k_neighbors=k_sm)
    Xsm, ysm = sm.fit_resample(Xa_sc, ya)
    return Xsm, ysm, Xte_sc, yte, sc, imp, Xtr.columns.tolist()

# ── Prepare GW ─────────────────────────────────────────────────────────────────
gw_y = encode_y(gw["wqi_class"], GW_CLASSES)
gw_mask = gw_y >= 0
gw_y_filt, _ = remap_consecutive(gw_y[gw_mask].reset_index(drop=True))

X_gw_sm, y_gw_sm, X_gw_te, y_gw_te, sc_gw, imp_gw, gw_feat_cols = \
    augment_smote_pipeline(gw[GW_PARAMS][gw_mask].reset_index(drop=True), gw_y_filt)

gw_feat_names = [FEATURE_LABELS.get(c, c) for c in gw_feat_cols]

# ── Prepare SW ─────────────────────────────────────────────────────────────────
sw_cls_col = next(c for c in ["WQI_Class","wqi_class"] if c in sw.columns)
sw_y = encode_y(sw[sw_cls_col], SW_CLASSES)
sw_mask = sw_y >= 0
sw_y_filt, _ = remap_consecutive(sw_y[sw_mask].reset_index(drop=True))

X_sw_sm, y_sw_sm, X_sw_te, y_sw_te, sc_sw, imp_sw, sw_feat_cols = \
    augment_smote_pipeline(sw[SW_PARAMS][sw_mask].reset_index(drop=True), sw_y_filt)

sw_feat_names = [FEATURE_LABELS.get(c, c) for c in sw_feat_cols]

# ── Helper: save beeswarm ──────────────────────────────────────────────────────
def to_2d_shap(shap_vals):
    """Convert any SHAP output format → 2D array (n_samples, n_features)."""
    if isinstance(shap_vals, list):
        # list of (n_samples, n_features) → one per class
        arr = np.stack([np.abs(v) for v in shap_vals])   # (n_cls, n_s, n_f)
        return arr.mean(axis=0)                            # (n_s, n_f)
    sv = np.array(shap_vals)
    if sv.ndim == 3:
        # (n_s, n_f, n_cls) or (n_cls, n_s, n_f) — handle both
        if sv.shape[-1] <= sv.shape[0]:   # last dim = n_cls
            return np.abs(sv).mean(axis=-1)   # (n_s, n_f)
        else:
            return np.abs(sv).mean(axis=0)    # (n_s, n_f)
    return np.abs(sv)                          # (n_s, n_f) already

def save_beeswarm(shap_vals, X_data, feat_names, title, filename, model_name,
                  water_type, cv_str=""):
    """Save beeswarm + bar SHAP plots with clean layout (no title overlap)."""
    sv = to_2d_shap(shap_vals)

    X_df = pd.DataFrame(X_data, columns=feat_names)

    # ── Beeswarm ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, max(7, len(feat_names)*0.55 + 2.5)),
                     facecolor="white")
    # Extra top margin so suptitle doesn't clash with feature labels
    plt.subplots_adjust(top=0.88, bottom=0.10, left=0.30, right=0.92)

    # SHAP beeswarm needs original (possibly signed) values for colour direction
    # Use the raw shap_vals but pass sv (abs-averaged) for ordering
    shap.summary_plot(sv, X_df, feature_names=feat_names,
                      show=False, plot_type="dot", max_display=len(feat_names),
                      color_bar=True, sort=True)

    # Clean up the auto-generated title (it says "SHAP" overlapping leftmost label)
    curr_ax = plt.gca()
    curr_ax.set_title("")          # remove shap's own title
    curr_ax.tick_params(axis='y', labelsize=9)

    fig.suptitle(
        f"SHAP Summary — {model_name} | {water_type}\n"
        f"All physicochemical parameters  ·  {cv_str}",
        fontsize=12, fontweight="bold", color="#1a252f",
        y=0.97
    )
    fig.text(0.5, 0.02,
             "SHAP Value (impact on WQI class prediction — averaged over classes)",
             ha="center", fontsize=9, color="#555")

    plt.savefig(os.path.join(OUT, f"{filename}_beeswarm.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {filename}_beeswarm.png")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    mean_shap = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_shap)[::-1]

    fig2, ax2 = plt.subplots(figsize=(10, max(5, len(feat_names)*0.42 + 1.5)),
                              facecolor="white")
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(feat_names)))
    bars = ax2.barh(range(len(feat_names)), mean_shap[order], color=colors,
                    edgecolor="white", linewidth=0.5)
    ax2.set_yticks(range(len(feat_names)))
    ax2.set_yticklabels([feat_names[i] for i in order], fontsize=9)
    ax2.set_xlabel("Mean |SHAP Value| (average impact on model output)",
                   fontsize=10)
    ax2.set_title(
        f"Feature Importance (SHAP) — {model_name} | {water_type}\n{cv_str}",
        fontsize=11, fontweight="bold", color="#1a252f", pad=12)
    ax2.invert_yaxis()
    ax2.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)

    for i, bar in enumerate(bars):
        w = bar.get_width()
        ax2.text(w + 0.0005, bar.get_y() + bar.get_height()/2,
                 f"{w:.4f}", va="center", ha="left", fontsize=8, color="#333")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"{filename}_bar.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {filename}_bar.png")

# ═══════════════════════════════════════════════════════════════════════════════
# GW — KNN via KernelExplainer
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGW: Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_gw_sm, y_gw_sm)
acc_gw = (knn.predict(X_gw_te) == y_gw_te).mean()
print(f"  GW KNN test accuracy: {acc_gw:.3f}")

print("  Computing SHAP values (KernelExplainer — may take ~2 min)...")
t0 = time.time()
# Use a small background dataset for speed (k-means summarises training data)
background_gw = shap.kmeans(X_gw_sm, min(50, len(X_gw_sm)))
explainer_gw  = shap.KernelExplainer(knn.predict_proba, background_gw)
# Explain test set (or a sample of training set if test set is very small)
X_explain_gw  = X_gw_te if len(X_gw_te) >= 5 else X_gw_sm[:30]
shap_vals_gw  = explainer_gw.shap_values(X_explain_gw, nsamples=100, silent=True)
print(f"  Done in {time.time()-t0:.1f}s")

save_beeswarm(shap_vals_gw, X_explain_gw, gw_feat_names,
              "SHAP Summary — KNN | Groundwater", "SHAP_GW_KNN",
              "KNN", "Groundwater WQI",
              "5-Fold CV Accuracy: 0.987 ± 0.027")

# ═══════════════════════════════════════════════════════════════════════════════
# SW — Decision Tree via TreeExplainer (fast, exact)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nSW: Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_sw_sm, y_sw_sm)
acc_sw = (dt.predict(X_sw_te) == y_sw_te).mean()
print(f"  SW Decision Tree test accuracy: {acc_sw:.3f}")

print("  Computing SHAP values (TreeExplainer — fast)...")
explainer_sw = shap.TreeExplainer(dt)
X_explain_sw = X_sw_te if len(X_sw_te) >= 5 else X_sw_sm[:30]
shap_vals_sw = explainer_sw.shap_values(X_explain_sw)
print("  Done.")

save_beeswarm(shap_vals_sw, X_explain_sw, sw_feat_names,
              "SHAP Summary — Decision Tree | Surface Water", "SHAP_SW_DT",
              "Decision Tree", "Surface Water WQI",
              "5-Fold CV Accuracy: 0.996 ± 0.008")

# ── Interaction plot for Decision Tree ────────────────────────────────────────
print("  Computing SHAP interaction values (Decision Tree)...")
try:
    shap_interact = explainer_sw.shap_interaction_values(X_explain_sw)
    # Average over classes if multiclass
    if isinstance(shap_interact, list):
        shap_interact = np.abs(np.stack(shap_interact)).mean(axis=0)

    fig, ax = plt.subplots(figsize=(max(9, len(sw_feat_names)*0.7 + 1),
                                    max(8, len(sw_feat_names)*0.6 + 2)),
                           facecolor="white")
    plt.subplots_adjust(top=0.88, left=0.28, right=0.95, bottom=0.28)

    shap.summary_plot(shap_interact, pd.DataFrame(X_explain_sw, columns=sw_feat_names),
                      feature_names=sw_feat_names, show=False,
                      max_display=len(sw_feat_names))

    plt.gca().set_title("")
    fig.suptitle(
        "SHAP Interaction Values — Decision Tree | Surface Water WQI\n"
        "5-Fold CV Accuracy: 0.996 ± 0.008",
        fontsize=11, fontweight="bold", color="#1a252f", y=0.97
    )
    fig.text(0.5, 0.01, "SHAP Interaction Value",
             ha="center", fontsize=9, color="#555")

    plt.savefig(os.path.join(OUT, "SHAP_SW_DT_interaction.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved SHAP_SW_DT_interaction.png")
except Exception as e:
    print(f"  Interaction plot skipped: {e}")

print(f"\nAll SHAP plots saved to: {OUT}")
print("Note: These SHAP values are computed on the augmented+SMOTE pipeline output,")
print("which itself is derived from the 2016-2018 averaged field measurements.")
