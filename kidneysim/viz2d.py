"""
2D figures (matplotlib) for every level of the model.

Every function returns the matplotlib Figure, so you can save it:
    fig = plot_tank(...); fig.savefig("tank.png", dpi=150)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

from . import constants as k
from . import geometry as geo
from .tubule import SEGMENTS, SEGMENT_NAMES

# ---- palette (fixed order, colour-blind checked) ------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
NORMAL_BAND = "#0ca30c"
BLUES = LinearSegmentedColormap.from_list(
    "kidney_blues", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                     "#256abf", "#184f95", "#0d366b"])
SEGMENT_SHADES = ["#f3f2ee", "#ffffff", "#f3f2ee", "#ffffff", "#f3f2ee"]


def style():
    """Light, quiet chart style used by all figures."""
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "legend.frameon": False,
        "lines.linewidth": 2.0, "font.size": 10,
        "axes.prop_cycle": plt.cycler(color=SERIES),
    })


style()


def _band(ax, lo, hi, label="normal range", axis="y"):
    if axis == "y":
        ax.axhspan(lo, hi, color=NORMAL_BAND, alpha=0.08, lw=0, label=label)
    else:
        ax.axvspan(lo, hi, color=NORMAL_BAND, alpha=0.08, lw=0, label=label)


# ---- Level 0 --------------------------------------------------------------------
def plot_tank(runs, solute="urea"):
    """runs: list of (label, t_days, C)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    scale = 1000 if solute == "creatinine" else 1
    lo, hi = k.NORMAL_RANGES["creatinine_umol_l" if solute == "creatinine" else "urea_mmol_l"]
    _band(ax, lo, hi)
    for i, (label, t, C) in enumerate(runs):
        ax.plot(t, C * scale, color=SERIES[i % 8], label=label)
        ax.annotate(f"{C[-1]*scale:.0f}", (t[-1], C[-1] * scale), xytext=(4, 0),
                    textcoords="offset points", va="center", color=TEXT2, fontsize=9)
    ax.set_xlabel("time (days)")
    ax.set_ylabel(f"plasma {solute} ({'µmol/L' if scale == 1000 else 'mmol/L'})")
    ax.set_title(f"Level 0 - plasma {solute}: V dC/dt = G - K C")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# ---- Level 1 --------------------------------------------------------------------
def plot_capillary(f):
    """Pressures along the glomerular capillary (output of filtration_along_capillary)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    x = f["x"]
    ax.plot(x, f["P_gc"], color=SERIES[0], label="P_GC  blood pressure (pushes out)")
    ax.plot(x, f["P_bs"] + f["pi_gc"], color=SERIES[1], label="P_BS + π_GC  (pull back)")
    ax.fill_between(x, f["P_bs"] + f["pi_gc"], f["P_gc"],
                    where=f["P_gc"] > f["P_bs"] + f["pi_gc"], color=SERIES[0], alpha=0.12,
                    label="net filtration pressure")
    ax.set_xlabel("position along capillary (0 = afferent, 1 = efferent)")
    ax.set_ylabel("pressure (mmHg)")
    ax.set_title("Starling forces along the capillary")
    ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    cum = (f["Q_plasma"][0] - f["Q_plasma"]) * 1000
    ax.plot(x, cum, color=SERIES[2])
    ax.set_xlabel("position along capillary")
    ax.set_ylabel("filtered so far (mL/min)")
    ax.set_title(f"GFR = {f['gfr']*1000:.0f} mL/min, filtration fraction {100*f['filtration_fraction']:.0f} %")
    fig.tight_layout()
    return fig


def plot_autoregulation(maps, gfr_auto, gfr_passive):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _band(ax, 80, 180, label="autoregulation range", axis="x")
    ax.plot(maps, np.asarray(gfr_auto) * 1000, color=SERIES[0], label="with autoregulation")
    ax.plot(maps, np.asarray(gfr_passive) * 1000, color=SERIES[1], ls="--", label="rigid vessels")
    ax.set_xlabel("mean arterial pressure (mmHg)")
    ax.set_ylabel("GFR (mL/min)")
    ax.set_ylim(0, 300)
    ax.set_title("Level 1 - autoregulation keeps filtration steady")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_sieving(radii, theta, molecules):
    """molecules: dict name -> (radius, theta)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(radii, theta, color=SERIES[0], label="neutral molecule")
    for name, (r, th) in molecules.items():
        ax.plot(r, th, "o", ms=8, color=SERIES[1], mec="#fcfcfb", mew=2)
        ax.annotate(name, (r, th), xytext=(6, 4), textcoords="offset points", color=TEXT2, fontsize=9)
    ax.set_xlabel("molecule radius (nm)")
    ax.set_ylabel("sieving coefficient θ  (filtrate / plasma)")
    ax.set_title("Which molecules pass the glomerular filter?")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ---- Level 2 --------------------------------------------------------------------
def _shade_segments(ax, show_names=True):
    ymin, ymax = ax.get_ylim()
    for j, seg in enumerate(SEGMENTS):
        ax.axvspan(j, j + 1, color=SEGMENT_SHADES[j], zorder=0, lw=0)
        if show_names:
            ax.text(j + 0.5, 1.0, seg, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=TEXT2)


def plot_nephron_profiles(r):
    """Four panels along the nephron for a tubule.run_nephron() result."""
    p = r["profile"]
    x = p["position"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    ax = axes[0, 0]
    ax.plot(x, 100 * p["F"] / r["gfr"], color=SERIES[0])
    ax.set_yscale("log")
    ax.set_ylabel("fluid flow (% of GFR, log)")
    ax.set_title("Water left in the tube")
    ax = axes[0, 1]
    ax.plot(x, p["osm"], color=SERIES[0], label="tubular fluid")
    ax.plot(x, p["osm_interstitium"], color=SERIES[1], ls="--", label="surrounding tissue")
    ax.set_ylabel("osmolarity (mOsm/L)")
    ax.set_title("Osmolarity: water follows salt")
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1, 0]
    for i, key in enumerate(["Na", "glucose", "urea"]):
        ax.plot(x, 100 * p["N_" + key] / r["filtered"][key], color=SERIES[i], label=key)
    ax.set_ylabel("amount left (% of filtered)")
    ax.set_title("Reabsorption of solutes")
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[1, 1]
    for i, key in enumerate(["Na", "urea"]):
        ax.plot(x, p["C_" + key], color=SERIES[i], label=key)
    ax.set_yscale("log")
    ax.set_ylabel("concentration (mmol/L, log)")
    ax.set_title("Concentrations in the tubular fluid")
    ax.legend(loc="upper left", fontsize=8)
    for ax in axes.flat:
        _shade_segments(ax, show_names=False)
        ax.set_xlim(0, len(SEGMENTS))
        ax.set_xticks([j + 0.5 for j in range(len(SEGMENTS))])
        ax.set_xticklabels([SEGMENT_NAMES[s].replace(" ", "\n") for s in SEGMENTS], fontsize=8)
    fig.suptitle(f"Level 2 - one pass through the nephron: urine {r['urine_flow_ml_min']:.2f} mL/min, "
                 f"{r['urine_osm']:.0f} mOsm/L", fontweight="bold", color=TEXT)
    fig.tight_layout()
    return fig


def plot_nephron_map(r, quantity="osm", ax=None, cmap=BLUES):
    """2D map of the nephron (x vs depth) coloured by a quantity of the profile."""
    p = r["profile"]
    x, y, z = geo.nephron_xyz(p)
    vals = p[quantity]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.5, 7))
    else:
        fig = ax.figure
    # background: tissue osmolarity by depth
    depth = np.linspace(0, 1, 100)
    from .tubule import interstitial_osm
    bg = interstitial_osm(depth, osm_max=p["osm_interstitium"].max())
    ax.imshow(bg[:, None], extent=[-0.25, 1.15, -1, 0], aspect="auto", cmap="Greys",
              alpha=0.25, origin="upper", vmin=0, vmax=1500)
    ax.axhline(-geo.CORTEX_DEPTH, color=TEXT2, lw=0.8, ls=":")
    ax.text(1.12, -0.15, "CORTEX", ha="right", color=TEXT2, fontsize=8)
    ax.text(1.12, -0.95, "MEDULLA", ha="right", color=TEXT2, fontsize=8)
    pts = np.column_stack([x, z]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidths=5, capstyle="round")
    lc.set_array(vals[:-1])
    ax.add_collection(lc)
    ax.add_patch(plt.Circle((0, -0.05), 0.07, color=SERIES[7], alpha=0.35))
    ax.text(0, -0.05, "glom.", ha="center", va="center", fontsize=7, color=TEXT)
    ax.set_xlim(-0.25, 1.15)
    ax.set_ylim(-1.05, 0.05)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"Nephron map coloured by {quantity}")
    fig.colorbar(lc, ax=ax, shrink=0.7, label=quantity)
    return fig


def plot_glucose_titration(g, filt, reab, exc):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(g, filt, color=SERIES[0], label="filtered")
    ax.plot(g, reab, color=SERIES[2], label="reabsorbed (SGLT)")
    ax.plot(g, exc, color=SERIES[1], label="excreted in urine")
    _band(ax, 4, 6, label="normal fasting glucose", axis="x")
    ax.set_xlabel("plasma glucose (mmol/L)")
    ax.set_ylabel("glucose (mmol/min)")
    ax.set_title("Glucose titration curve: transporters saturate")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# ---- Level 3 --------------------------------------------------------------------
def plot_countercurrent(res):
    hD = res["history_D"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    im = ax.imshow(hD.T, aspect="auto", cmap=BLUES, origin="upper",
                   extent=[0, hD.shape[0] - 1, 1, 0])
    ax.set_xlabel("cycle")
    ax.set_ylabel("depth (0 = cortex, 1 = papilla)")
    ax.set_title("Gradient building up (descending limb)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="mOsm/L")
    ax = axes[1]
    d = res["depth"]
    ax.plot(res["D"], d, "o-", color=SERIES[0], label="descending limb / tissue")
    ax.plot(res["A"], d, "o-", color=SERIES[1], label="ascending limb")
    ax.invert_yaxis()
    ax.set_xlabel("osmolarity (mOsm/L)")
    ax.set_ylabel("depth")
    ax.set_title(f"Steady state: tip {res['tip']:.0f}, out {res['outflow']:.0f} mOsm/L")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_loop_length(lengths, tips):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lengths, tips, "o-", color=SERIES[0])
    ax.axhline(1200, color=TEXT2, ls=":", lw=1)
    ax.text(lengths[0], 1220, "human maximum ~1200", color=TEXT2, fontsize=8)
    ax.set_xlabel("loop length (number of levels)")
    ax.set_ylabel("tip osmolarity (mOsm/L)")
    ax.set_title("Longer loops concentrate more")
    fig.tight_layout()
    return fig


# ---- Level 4 --------------------------------------------------------------------
def plot_body(runs):
    """runs: dict label -> coupled.simulate() output. Six small panels."""
    panels = [("V", "body water (L)", None), ("P_Na", "plasma Na (mmol/L)", (135, 145)),
              ("P_urea", "plasma urea (mmol/L)", k.NORMAL_RANGES["urea_mmol_l"]),
              ("GFR", "GFR (mL/min)", k.NORMAL_RANGES["gfr_ml_min"]),
              ("urine_flow", "urine flow (mL/min)", k.NORMAL_RANGES["urine_flow_ml_min"]),
              ("ADH", "ADH level (0-1)", None)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, (key, label, band) in zip(axes.flat, panels):
        if band:
            _band(ax, *band)
        for i, (name, out) in enumerate(runs.items()):
            ax.plot(out["t"], out[key], color=SERIES[i % 8], label=name)
        ax.set_title(label)
    for ax in axes[1]:
        ax.set_xlabel("time (days)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=9)
    fig.suptitle("Level 4 - body + kidney with hormone feedback", fontweight="bold", color=TEXT)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    return fig


# ---- Level 5 --------------------------------------------------------------------
def plot_dialyzer(prof):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(prof["x"], prof["C_blood"], color=SERIES[7], label="blood  (flows →)")
    ax.plot(prof["x"], prof["C_dialysate"], color=SERIES[0], label="dialysate  (flows ←)")
    ax.fill_between(prof["x"], prof["C_dialysate"], prof["C_blood"], color=SERIES[3], alpha=0.12,
                    label="driving difference  C_b - C_d")
    ax.set_xlabel("position along the fibre")
    ax.set_ylabel("urea (mmol/L)")
    ax.set_title(f"Countercurrent dialyzer: clearance {prof['clearance']*1000:.0f} mL/min")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_treatment(tr):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _band(ax, *k.NORMAL_RANGES["urea_mmol_l"])
    ax.plot(tr["t"], tr["C"], color=SERIES[0], label="plasma urea")
    on = tr["on"]
    ax.fill_between(tr["t"], 0, tr["C"].max() * 1.05, where=on, color=SERIES[1], alpha=0.15,
                    lw=0, label="dialysis session")
    ax.set_xlabel("time (days)")
    ax.set_ylabel("urea (mmol/L)")
    ax.set_title(f"Hemodialysis 3x/week - Kt/V per session = {tr['Kt_V'][0]:.2f}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_breakthrough(col, t_bt=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(col["t"], col["outlet"], color=SERIES[1], label="NH4+ leaving the column")
    ax.plot(col["t"], col["na_released"], color=SERIES[0], label="Na+ released (1:1 exchange)")
    ax.axhline(col["c_in"], color=TEXT2, ls=":", lw=1)
    if t_bt:
        ax.axvline(t_bt, color=TEXT2, lw=1)
        ax.text(t_bt, col["c_in"] * 0.5, f"  breakthrough\n  {t_bt:.0f} min", color=TEXT2, fontsize=8)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("concentration (mmol/L)")
    ax.set_title("Ion exchange column: breakthrough curve")
    ax.legend(loc="center left", fontsize=8)
    ax = axes[1]
    im = ax.imshow(col["q"].T, aspect="auto", origin="lower", cmap=BLUES,
                   extent=[col["t"][0], col["t"][-1], 0, 1])
    ax.set_xlabel("time (min)")
    ax.set_ylabel("position in column (0 inlet, 1 outlet)")
    ax.set_title("Resin loading q(z, t): the 'front' moves")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="NH4+ on resin (mmol/g)")
    fig.tight_layout()
    return fig
