"""
Fig 12 — Individual Confusion Matrices: one PNG per model per water type
=========================================================================
Outputs (in plots/confusion_matrices/):
  GW_CM_KNN.png, GW_CM_RandomForest.png, ... GW_CM_KMeans.png  (7 files)
  SW_CM_KNN.png, SW_CM_RandomForest.png, ... SW_CM_KMeans.png  (7 files)
  + the original grid files Fig12a and Fig12b are preserved
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = r'D:\My Projects\Py-DS-ML-Bootcamp-master\research paper'
BASE = os.path.dirname(_HERE)
OUT  = os.path.join(_HERE, "plots", "confusion_matrices")
os.makedirs(OUT, exist_ok=True)

# ── Load & prep data ──────────────────────────────────────────────────────────
gw = pd.read_csv(os.path.join(BASE, "groundwater_wqi.csv"))
sw = pd.read_csv(os.path.join(BASE, "surface_wqi.csv"))

GW_PARAMS = ["ph","turbidity","conductivity","chloride_(ppm)","sulphates(ppm)",
             "iron_(ppm)","bod(ppm)","do(ppm)","ammonia(ppm)","nitrate(ppm)",
             "total_bacterial_count_(cfu/ml)","total_fungal_count_(cfu/ml)"]
SW_PARAMS = [c for c in ["pH","Turbidity","Conductivity","Chloride (ppm)",
             "Sulphates(ppm)","Iron (ppm)","BOD(ppm)","DO(ppm)",
             "Ammonia(ppm)","Nitrate(ppm)",
             "Total Bacterial Count (cfu/ml)","Total Fungal Count (cfu/ml)"]
             if c in sw.columns]

GW_CLASS_COL = "wqi_class"
SW_CLASS_COL = "WQI_Class"

GW_CLASSES = ["Excellent","Good","Poor","Very Poor","Unsuitable"]
SW_CLASSES = ["Medium","Poor","Very Poor","Unsuitable"]

MODEL_COLORS = {
    "KNN":"#1565C0","Random Forest":"#2E7D32","SVM":"#B71C1C",
    "Gradient Boosting":"#E65100","Decision Tree":"#4A148C",
    "XGBoost":"#F9A825","K-Means":"#00695C",
}
CMAPS = {
    "KNN":"Blues","Random Forest":"Greens","SVM":"Reds",
    "Gradient Boosting":"Oranges","Decision Tree":"Purples",
    "XGBoost":"YlOrBr","K-Means":"PuBuGn",
}

def augment_smote(X, y):
    X = X.copy().astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                            random_state=42, stratify=y)
    rng = np.random.RandomState(42)
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
    k = max(1, min(5, pd.Series(ya).value_counts().min()-1))
    sm = SMOTE(random_state=42, k_neighbors=k)
    Xsm, ysm = sm.fit_resample(Xa_sc, ya)
    return Xsm, ysm, Xte_sc, yte

def encode_classes(y, classes):
    mapping = {c: i for i, c in enumerate(classes)}
    return y.map(lambda v: mapping.get(v, -1))

# ── Encode & pipeline ────────────────────────────────────────────────────────
gw_params_ok = [c for c in GW_PARAMS if c in gw.columns]
sw_params_ok = [c for c in SW_PARAMS if c in sw.columns]

gw_y_enc = encode_classes(gw[GW_CLASS_COL], GW_CLASSES)
sw_y_enc = encode_classes(sw[SW_CLASS_COL], SW_CLASSES)

gw_mask = gw_y_enc >= 0; sw_mask = sw_y_enc >= 0

# Re-map to consecutive 0-based labels (required by XGBoost)
def remap_consecutive(y_series):
    uniq = sorted(y_series.unique())
    remap = {v: i for i, v in enumerate(uniq)}
    return y_series.map(remap), remap

gw_y_filt, gw_remap = remap_consecutive(gw_y_enc[gw_mask].reset_index(drop=True))
sw_y_filt, sw_remap = remap_consecutive(sw_y_enc[sw_mask].reset_index(drop=True))

X_gw_sm, y_gw_sm, X_gw_te, y_gw_te = augment_smote(gw[gw_params_ok][gw_mask].reset_index(drop=True), gw_y_filt)
X_sw_sm, y_sw_sm, X_sw_te, y_sw_te = augment_smote(sw[sw_params_ok][sw_mask].reset_index(drop=True), sw_y_filt)

# Rebuild present-label to class-name mapping after remap
gw_orig_classes = {v_new: GW_CLASSES[v_old] for v_old, v_new in gw_remap.items() if v_old < len(GW_CLASSES)}
sw_orig_classes = {v_new: SW_CLASSES[v_old] for v_old, v_new in sw_remap.items() if v_old < len(SW_CLASSES)}

# ── Train classifiers ────────────────────────────────────────────────────────
CLASSIFIERS = {
    "KNN":               KNeighborsClassifier(n_neighbors=5),
    "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":               SVC(kernel="rbf", C=1, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Decision Tree":     DecisionTreeClassifier(random_state=42),
    "XGBoost":           XGBClassifier(n_estimators=100, random_state=42,
                                        eval_metric="mlogloss", verbosity=0),
}

gw_models, sw_models = {}, {}
for mname, clf in CLASSIFIERS.items():
    gw_models[mname] = clf.__class__(**clf.get_params()).fit(X_gw_sm, y_gw_sm)
    sw_models[mname] = clf.__class__(**clf.get_params()).fit(X_sw_sm, y_sw_sm)
    print(f"  {mname} trained.")

# ── Draw single confusion matrix ──────────────────────────────────────────────
def plot_single_cm(mname, y_te, y_pred, classes_present, class_labels,
                   water_type, savepath):
    color  = MODEL_COLORS.get(mname, "#555")
    cmap   = CMAPS.get(mname, "Blues")
    n_cls  = len(classes_present)
    acc    = accuracy_score(y_te, y_pred)

    cm     = confusion_matrix(y_te, y_pred, labels=classes_present)
    cm_n   = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(5.5, 4.8), facecolor="white")
    im = ax.imshow(cm_n, cmap=cmap, vmin=0, vmax=1.8)

    # Cell annotations — always dark text (colours are light with vmax=1.8)
    for ri in range(n_cls):
        for ci in range(n_cls):
            ax.text(ci, ri, f"{cm[ri,ci]}\n({cm_n[ri,ci]*100:.0f}%)",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color="#1a252f")

    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(class_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(class_labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10, fontweight="bold")
    ax.set_ylabel("Actual",    fontsize=10, fontweight="bold")
    ax.set_title(f"{mname}  —  {water_type}\nAccuracy = {acc:.3f}",
                 fontsize=12, fontweight="bold", color=color, pad=10)

    plt.colorbar(im, ax=ax, shrink=0.7, label="Normalised proportion")
    plt.tight_layout()
    plt.savefig(savepath, dpi=180, bbox_inches="tight")
    plt.close()

# ── Get y_pred per model ──────────────────────────────────────────────────────
def get_preds_kmeans(X_tr, y_tr, X_te, n_cls):
    km = KMeans(n_clusters=n_cls, random_state=42, n_init=10)
    km.fit(X_tr)
    tr_cl = km.predict(X_tr)
    y_arr = y_tr if isinstance(y_tr, np.ndarray) else np.array(y_tr)
    cluster_lbl = {}
    for ci in np.unique(tr_cl):
        idx = np.where(tr_cl == ci)[0]
        cluster_lbl[ci] = int(pd.Series(y_arr[idx]).mode()[0])
    te_cl = km.predict(X_te)
    return np.array([cluster_lbl[c] for c in te_cl])

# ── Generate all individual PNGs ──────────────────────────────────────────────
for wtype, models_dict, X_tr, y_tr, X_te, y_te, cls_map in [
    ("Groundwater",  gw_models, X_gw_sm, y_gw_sm, X_gw_te, y_gw_te, gw_orig_classes),
    ("Surface Water",sw_models, X_sw_sm, y_sw_sm, X_sw_te, y_sw_te, sw_orig_classes),
]:
    present_labels = sorted(np.unique(y_te))
    present_names  = [cls_map.get(i, str(i)) for i in present_labels]
    prefix = "GW" if wtype == "Groundwater" else "SW"
    print(f"\n{wtype}:")

    model_order = list(CLASSIFIERS.keys()) + ["K-Means"]
    for mname in model_order:
        if mname == "K-Means":
            y_pred = get_preds_kmeans(X_tr, y_tr, X_te, len(set(y_tr)))
        else:
            y_pred = models_dict[mname].predict(X_te)

        safe  = mname.replace(" ", "_").replace("/","_")
        fpath = os.path.join(OUT, f"{prefix}_CM_{safe}.png")
        plot_single_cm(mname, y_te, y_pred, present_labels,
                       present_names, wtype, fpath)
        print(f"  Saved {prefix}_CM_{safe}.png  (acc={accuracy_score(y_te, y_pred):.3f})")

print("\nAll individual confusion matrices saved to plots/confusion_matrices/")
