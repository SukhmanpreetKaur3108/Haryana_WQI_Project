"""
Combined GW + SW WQI Year Maps
================================
3 publication maps (one per year: 2016, 2017, 2018).
Both GW (■ square) and SW (● circle) sites on one Haryana basemap.
Marker colour = WQI class.  Legend outside the map (right side).

Output files (plots/ — does NOT override existing individual maps):
  Combined_WQI_Map_2016.png
  Combined_WQI_Map_2017.png
  Combined_WQI_Map_2018.png
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
from matplotlib.image import imread
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = r'D:\My Projects\Py-DS-ML-Bootcamp-master\research paper'

BASE = os.path.dirname(_HERE)
OUT  = os.path.join(_HERE, "plots")
os.makedirs(OUT, exist_ok=True)

HARYANA_JPG = os.path.join(_HERE, "Haryana Irrigation Map.jpg")
# Geographic extent calibrated from known city positions
MAP_EXTENT = [74.24, 77.68, 27.32, 31.60]   # [lon_min, lon_max, lat_min, lat_max]

# ── DMS converter ─────────────────────────────────────────────────────────────
def dms(deg, mins, secs, d):
    v = float(deg) + float(mins)/60 + float(secs)/3600
    return round(-v if d in ('S', 'W') else v, 6)

# ── 14 site pairs: (display_name, gw_lat, gw_lon, sw_lat, sw_lon) ─────────────
# Same geographic locations; GW and SW markers offset ±0.05° lon for readability
SITE_PAIRS = [
    ("Ghaggar R.①\n[Pinjore]",
        dms(30,46,35,'N'), dms(76,54,51,'E'), dms(30,46,30,'N'), dms(76,54,50,'E')),
    ("Ghaggar R.②\n[Panchkula]",
        dms(30,41,37,'N'), dms(76,52,52,'E'), dms(30,41,39,'N'), dms(76,52,53,'E')),
    ("Ghaggar R.③\n[Ambala]",
        30.4762, 76.5906,  30.4766, 76.5905),
    ("Ghaggar R.④\n[Fatehabad]",
        dms(29,41, 1,'N'), dms(75,34,33,'E'), dms(29,41, 0,'N'), dms(75,34,30,'E')),
    ("Ghaggar R.⑤\n[Sirsa]",
        dms(29,29,21,'N'), dms(74,54,38,'E'), dms(29,29,21,'N'), dms(74,54,38,'E')),
    ("Markanda R.①\n[Shahabad]",
        30.1585, 76.8677,  30.1588, 76.8678),
    ("Yamuna R.①\n[Yamunanagar]",
        30.1290, 77.2674,  30.1290, 77.2674),
    ("Yamuna R.②\n[Panipat]",
        dms(29,14,48,'N'), dms(77,21, 5,'E'), dms(29,14,46,'N'), dms(77,21, 3,'E')),
    ("Yamuna R.③\n[Sonipat]",
        28.993, 77.027,   28.990, 77.022),
    ("Yamuna R.④\n[Faridabad]",
        28.4215, 77.3076,  28.4211, 77.3078),
    ("Agra Canal①\n[Palwal]",
        dms(28,33,56,'N'), dms(77,17,60,'E'), dms(28,33,59,'N'), dms(77,17,60,'E')),
    ("Brahma Sarovar\n[Kurukshetra]",
        29.57, 76.51,   29.58, 76.50),
    ("Karan Lake\n[Karnal]",
        dms(29,44,62,'N'), dms(76,58,58,'E'), dms(29,44,62,'N'), dms(76,58,54,'E')),
    ("Teekar Taal\n[Morni Hills]",
        30.6873, 77.0878,  30.6873, 77.0877),
]

# CSV site name → index in SITE_PAIRS
GW_CSV_TO_IDX = {
    "Ghaggar River(Site 1) (Ground Water)":    0,
    "Ghaggar River(Site 2) (Ground Water)":    1,
    "Ghaggar River(Site 3) (Ground Water)":    2,
    "Ghaggar River(Site 4) (Ground Water)":    3,
    "Ghaggar River(Site 5) (Ground Water)":    4,
    "Markanda River(Site 1) (Ground Water)":   5,
    "Yamuna River(Site 1) (Ground Water)":     6,
    "Yamuna River(Site 2) (Ground Water)":     7,
    "Yamuna River(Site 3) (Ground Water)":     8,
    "Yamuna River(Site 4) (Ground Water)":     9,
    "Agra Canal(Site 1) (Ground Water)":       10,
    "Brahma Sarovar(Site 2) (Ground Water )":  11,
    "Karan Lake (Site 1) (Ground Water)":      12,
    "Teekar Taal(Site 3) (Ground Water)":      13,
}
SW_CSV_TO_IDX = {
    "Ghaggar River(Site 1) (Surface Water)":   0,
    "Ghaggar River(Site 2) (Surface Water)":   1,
    "Ghaggar River(Site 3) (Surface Water)":   2,
    "Ghaggar River(Site 4) (Surface Water)":   3,
    "Ghaggar River(Site 5) (Surface Water)":   4,
    "Markanda River(Site 1) (Surface Water)":  5,
    "Yamuna River(Site 1) (Surface Water)":    6,
    "Yamuna River(Site 2) (Surface Water)":    7,
    "Yamuna River(Site 3) (Surface Water)":    8,
    "Yamuna River(Site 4) (Surface Water)":    9,
    "Agra Canal(Site 1) (Surface Water)":      10,
    "Brahma Sarovar(Site 2) (Surface Water)":  11,
    "Karan Lake (Site 1) (Surface Water)":     12,
    "Teekar Taal(Site 3) (Surface Water)":     13,
}

CLASS_COLORS = {
    "Excellent":  "#0D47A1",
    "Good":       "#1B5E20",
    "Medium":     "#F9A825",
    "Poor":       "#E65100",
    "Very Poor":  "#B71C1C",
    "Unsuitable": "#4A148C",
}

# Per-site label offsets (lon_delta, lat_delta)
# +lon = right, +lat = up
OFFSETS = {
    0:  (+0.04, +0.26),   # Ghaggar-1 (Pinjore) — UP
    1:  (+0.16, -0.24),   # Ghaggar-2 (Panchkula) — DOWN-RIGHT
    2:  (-0.62, +0.06),   # Ghaggar-3 (Ambala) — LEFT
    3:  (-0.68, +0.06),   # Ghaggar-4 (Fatehabad) — LEFT
    4:  (-0.68, +0.06),   # Ghaggar-5 (Sirsa) — LEFT
    5:  (-0.64, +0.06),   # Markanda (Shahabad) — LEFT
    6:  (+0.26, +0.05),   # Yamuna-1 (Yamunanagar) — RIGHT
    7:  (+0.26, +0.05),   # Yamuna-2 (Panipat) — RIGHT
    8:  (+0.26, +0.05),   # Yamuna-3 (Sonipat) — RIGHT
    9:  (+0.16, -0.19),   # Yamuna-4 (Faridabad) — DOWN-RIGHT
    10: (+0.16, +0.18),   # Agra Canal (Palwal) — UP-RIGHT
    11: (-0.64, +0.06),   # Brahma Sarovar — LEFT
    12: (+0.26, +0.05),   # Karan Lake (Karnal) — RIGHT
    13: (+0.28, +0.02),   # Teekar Taal — RIGHT
}

CITY_LABELS = {
    "Chandigarh": (30.74, 76.79), "Ambala":     (30.37, 76.78),
    "Karnal":     (29.69, 76.99), "Panipat":    (29.39, 76.97),
    "Hisar":      (29.15, 75.72), "Sirsa":      (29.53, 75.03),
    "Faridabad":  (28.41, 77.31), "Kurukshetra":(29.97, 76.88),
    "Yamunanagar":(30.13, 77.28), "Panchkula":  (30.69, 76.85),
}

# ── Load CSVs ─────────────────────────────────────────────────────────────────
gw_df = pd.read_csv(os.path.join(BASE, "groundwater_wqi.csv"))
sw_df = pd.read_csv(os.path.join(BASE, "surface_wqi.csv"))
sw_df.columns = [c.strip() for c in sw_df.columns]
sw_wqi_col  = next(c for c in ["WQI","wqi"]                         if c in sw_df.columns)
sw_cls_col  = next(c for c in ["WQI_Class","wqi_class","WQI Class"] if c in sw_df.columns)
sw_site_col = next(c for c in ["Site","site"]                       if c in sw_df.columns)
sw_year_col = next(c for c in ["Year","year"]                       if c in sw_df.columns)

# ── Load basemap once ─────────────────────────────────────────────────────────
basemap_img = None
if os.path.exists(HARYANA_JPG):
    basemap_img = imread(HARYANA_JPG)
    print(f"Basemap loaded: {HARYANA_JPG}")
else:
    print("Haryana JPG not found — plain coordinate background will be used.")

# ── Per-year map generator ─────────────────────────────────────────────────────
def make_combined_map(year):
    gw_yr = gw_df[gw_df["year"] == year].copy()
    sw_yr = sw_df[sw_df[sw_year_col] == year].copy()

    # Build site_data dict  {idx: {"gw": (wqi, cls), "sw": (wqi, cls)}}
    site_data = {}
    for csv_name, idx in GW_CSV_TO_IDX.items():
        row = gw_yr[gw_yr["site"].str.strip() == csv_name.strip()]
        if not row.empty:
            site_data.setdefault(idx, {})["gw"] = (
                float(row["wqi"].values[0]),
                str(row["wqi_class"].values[0]).strip())
    for csv_name, idx in SW_CSV_TO_IDX.items():
        row = sw_yr[sw_yr[sw_site_col].str.strip() == csv_name.strip()]
        if not row.empty:
            site_data.setdefault(idx, {})["sw"] = (
                float(row[sw_wqi_col].values[0]),
                str(row[sw_cls_col].values[0]).strip())

    if not site_data:
        print(f"  No data for {year} — skipping")
        return

    # ── Canvas ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 13), facecolor="white")

    if basemap_img is not None:
        ax.imshow(basemap_img, extent=MAP_EXTENT, aspect="auto",
                  zorder=0, alpha=0.93)
    else:
        ax.set_facecolor("#D6EAF8")

    # City reference labels
    for city, (clat, clon) in CITY_LABELS.items():
        ax.plot(clon, clat, "^", color="#888", ms=3.5, zorder=5,
                mec="white", mew=0.5)
        ax.text(clon + 0.03, clat + 0.025, city,
                fontsize=5.5, color="#888", fontstyle="italic", zorder=5,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

    # ── Site markers ──────────────────────────────────────────────────────────
    GW_DX = -0.055   # GW square: shift left of nominal coordinate
    SW_DX = +0.055   # SW circle: shift right

    for idx, data in site_data.items():
        if idx >= len(SITE_PAIRS):
            continue
        name, gw_lat, gw_lon, sw_lat, sw_lon = SITE_PAIRS[idx]
        dx, dy = OFFSETS.get(idx, (+0.12, +0.10))

        # Reference point for label arrow (midpoint between the two markers)
        ref_lat = (gw_lat + sw_lat) / 2
        ref_lon = (gw_lon + sw_lon) / 2

        # Determine label border color: worst WQI class of the pair
        cls_order = list(CLASS_COLORS.keys())   # Excellent … Unsuitable
        gw_cls = data.get("gw", (None, ""))[1]
        sw_cls = data.get("sw", (None, ""))[1]
        border_cls = max((gw_cls, sw_cls),
                         key=lambda c: cls_order.index(c) if c in cls_order else 0)
        border_col = CLASS_COLORS.get(border_cls, "#607D8B")

        # GW marker — filled square
        if "gw" in data:
            gw_wqi, gw_cls_str = data["gw"]
            gw_col = CLASS_COLORS.get(gw_cls_str, "#607D8B")
            ax.scatter(gw_lon + GW_DX, gw_lat,
                       s=170, marker="s", c=gw_col,
                       edgecolors="white", linewidths=1.5, zorder=8, alpha=0.93)

        # SW marker — filled circle
        if "sw" in data:
            sw_wqi, sw_cls_str = data["sw"]
            sw_col = CLASS_COLORS.get(sw_cls_str, "#607D8B")
            ax.scatter(sw_lon + SW_DX, sw_lat,
                       s=170, marker="o", c=sw_col,
                       edgecolors="white", linewidths=1.5, zorder=8, alpha=0.93)

        # Build combined label text
        label_lines = [name.replace("\n", " ")]
        if "gw" in data:
            gw_wqi, gw_cls_str = data["gw"]
            label_lines.append(f"■ GW: {gw_wqi:.0f}  ({gw_cls_str})")
        if "sw" in data:
            sw_wqi, sw_cls_str = data["sw"]
            label_lines.append(f"● SW: {sw_wqi:.0f}  ({sw_cls_str})")
        label_text = "\n".join(label_lines)

        ax.annotate(
            label_text,
            xy=(ref_lon, ref_lat),
            xytext=(ref_lon + dx, ref_lat + dy),
            fontsize=6.0, color="#1a252f", fontweight="bold", zorder=9,
            arrowprops=dict(arrowstyle="-", color=border_col,
                            lw=0.9, connectionstyle="arc3,rad=0.05"),
            bbox=dict(boxstyle="round,pad=0.28", fc="white",
                      ec=border_col, alpha=0.93, linewidth=1.2))

    # ── Map furniture ─────────────────────────────────────────────────────────
    ax.set_xlim(74.10, 77.90)
    ax.set_ylim(27.32, 31.55)
    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°N)",  fontsize=9)
    ax.set_title(
        f"Water Quality Index (WQI) — Monitoring Sites, {year}\n"
        f"Haryana, India  ·  BIS IS 10500:2012  ·  "
        f"■ Groundwater (GW)   ●  Surface Water (SW)",
        fontsize=11, fontweight="bold", pad=10, color="#1a252f")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.30,
            color="#aaa", zorder=1)
    ax.tick_params(labelsize=7)

    # North arrow
    ax.annotate("N",  xy=(77.68, 31.32), fontsize=11, fontweight="bold",
                color="#1a252f", ha="center", zorder=11)
    ax.annotate("▲",  xy=(77.68, 31.22), fontsize=14, color="#1a252f",
                ha="center", zorder=11)

    # ── Legend — outside the map on the right ─────────────────────────────────
    class_handles = [
        mpatches.Patch(color=c, label=k)
        for k, c in CLASS_COLORS.items()
    ]
    gw_h = mlines.Line2D([], [], marker="s", color="#555", mfc="#555",
                         markersize=9, linestyle="None",
                         label="■  Groundwater (GW)")
    sw_h = mlines.Line2D([], [], marker="o", color="#555", mfc="#555",
                         markersize=9, linestyle="None",
                         label="●  Surface Water (SW)")

    ax.legend(
        handles=class_handles + [gw_h, sw_h],
        title="WQI Class  /  Water Type",
        bbox_to_anchor=(1.02, 0.5), loc="center left",
        fontsize=8.5, title_fontsize=9,
        framealpha=0.97, edgecolor="#ccc",
        fancybox=True, borderpad=0.8, labelspacing=0.65)

    out_path = os.path.join(OUT, f"Combined_WQI_Map_{year}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    n_sites = len(site_data)
    print(f"Saved: Combined_WQI_Map_{year}.png  ({n_sites} site pairs)")


# ── Generate all 3 maps ───────────────────────────────────────────────────────
for yr in [2016, 2017, 2018]:
    make_combined_map(yr)

print("\nDone — 3 combined year maps saved to:", OUT)
print("Existing GW_WQI_Map_*/SW_WQI_Map_* files are untouched.")
