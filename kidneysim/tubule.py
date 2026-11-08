"""
LEVEL 2 - The tubule: CHEMISTRY at the membrane (transporters + osmosis)
========================================================================

The filtrate (~125 mL/min) flows through a long tube, the nephron. Its wall
is made of cells packed with membrane transporters that take back what the
body wants to keep. We follow the fluid from start to end.

We model ONE "average" nephron scaled up to represent all ~2 million, so every
flow is for the whole pair of kidneys.

State along the tube (s = position inside a segment, 0 -> 1)
------------------------------------------------------------
    F(s)    tubular fluid flow                       [L/min]
    N_i(s)  amount of solute i flowing per minute    [mmol/min]
    C_i     = N_i / F   concentration                [mmol/L]
    Osm     = (2 N_Na + N_glucose + N_urea + N_drug) / F    (NaCl counts twice)

Governing equations (one-dimensional steady flow)
-------------------------------------------------
Solute reabsorption by a transporter (Michaelis-Menten, saturable):
    dN_i/ds = - J_max * C_i / (K_m + C_i)

Passive / "leaky" reabsorption (first order, proportional to amount):
    dN_i/ds = - k_i * N_i

Diffusion of urea across the wall (Fick's law):
    dN_urea/ds = - P_urea * (C_urea_lumen - C_urea_plasma)

Secretion of a drug/toxin by an EXCHANGER (organic anion transporter, OAT):
    dN_drug/ds = + S_max * C_plasma / (K_m + C_plasma)   (limited by blood supply)
    (OAT1 swaps one alpha-ketoglutarate from inside the cell for one organic
     anion from the blood: a real ion-exchange process in the kidney!)

Water follows by OSMOSIS (van 't Hoff / Kedem-Katchalsky with sigma = 1):
    dF/ds = - L_p * (Osm_interstitium - Osm_lumen)

    Water leaves the tube when the tissue around it is saltier.
    L_p is the water permeability. In the collecting duct L_p is controlled by
    the hormone ADH (vasopressin), which inserts water channels (aquaporin-2):
        L_p = L_p0 * (L_p,max / L_p0) ** ADH        ADH from 0 to 1
    (a ~30-fold range; every step of ADH multiplies the permeability)
    Real collecting ducts are not identical, so the final segment is split
    into 5 parallel ducts whose permeability differs (x0.15 ... x2.4).
    This makes the response to ADH gradual instead of all-or-nothing.

Tissue (interstitium) osmolarity around the tube
-------------------------------------------------
    Cortex:  290 mOsm/L
    Medulla: rises linearly from 290 to Osm_max (~1200) at the tip (papilla).
    This gradient is CREATED by the loop of Henle (see Level 3).

Segments and what each one does
-------------------------------
    PT   proximal tubule     65 % of Na and water back, ALL glucose (SGLT),
                             ~50 % of urea, secretes drugs (OAT exchanger)
    DL   descending limb     water out (permeable), salt stays in
    AL   ascending limb      salt pumped out (NKCC2), water cannot follow
    DCT  distal tubule       a bit more salt out (NCC, blocked by thiazides)
    CD   collecting duct     final salt (ENaC, aldosterone) and water (ADH)

Numerics: Euler steps along s for solutes; the water step is solved with a
backward (implicit) Euler step, which never makes F negative even when the
fluid becomes very concentrated.
"""

import numpy as np

from . import constants as k

SEGMENTS = ["PT", "DL", "AL", "DCT", "CD"]
SEGMENT_NAMES = {
    "PT": "Proximal tubule",
    "DL": "Descending limb",
    "AL": "Ascending limb",
    "DCT": "Distal tubule",
    "CD": "Collecting duct",
}

# Default parameters (whole-kidney values, tuned to textbook fractions)
DEFAULTS = dict(
    # proximal tubule
    k_Na_PT=1.05,          # 1/segment: exp(-1.05) -> 65 % reabsorbed
    Jmax_glu=1.7,          # mmol/min  SGLT capacity (sets glucose threshold)
    Km_glu=1.0,            # mmol/L
    P_urea_PT=0.2,         # L/min    urea permeability x area
    Lp_PT=0.02,            # L/min per mOsm/L (very water-permeable)
    Smax_drug=0.40,        # mmol/min  OAT secretion capacity (like PAH)
    Km_drug=0.10,          # mmol/L
    # descending limb
    Lp_DL=3.0e-5,
    # ascending limb
    Jmax_AL=6.0,           # mmol/min  NKCC2 capacity
    Km_AL=30.0,            # mmol/L
    # distal tubule
    Jmax_DCT=1.0,
    Km_DCT=20.0,
    # collecting duct
    k_Na_CD0=0.5,          # 1/segment, without aldosterone
    k_Na_CD=1.5,           # 1/segment, x aldosterone
    Lp_CD0=3.0e-5,         # water permeability with no ADH
    Lp_CDmax=1.0e-3,       # water permeability with maximal ADH (aquaporin-2 inserted)
    CD_spread=(0.15, 0.3, 0.6, 1.2, 2.4),   # relative permeability of 5 parallel ducts
)


def interstitial_osm(depth, osm_max=1200.0, cortex=290.0):
    """Tissue osmolarity. depth 0..1 = cortex surface .. tip of the papilla.
    The medulla starts at depth 0.3 in our geometry."""
    m = np.clip((np.asarray(depth, dtype=float) - 0.3) / 0.7, 0, 1)
    return cortex + (osm_max - cortex) * m


def _segment_depth(seg, s):
    """Depth (0 = kidney surface, 1 = papilla) of a point in a segment."""
    if seg == "PT":
        return 0.05 + 0.2 * s
    if seg == "DL":
        return 0.25 + 0.75 * s
    if seg == "AL":
        return 1.0 - 0.75 * s
    if seg == "DCT":
        return 0.25 - 0.15 * s
    if seg == "CD":
        return 0.10 + 0.90 * s
    raise ValueError(seg)


def _water_step(F, N_osm, osm_int, Lp_ds):
    """Backward-Euler solution of dF = -Lp ds (Osm_int - N_osm / F_new)."""
    if np.all(np.asarray(Lp_ds) <= 0):
        return F
    b = F - Lp_ds * osm_int
    return 0.5 * (b + np.sqrt(b * b + 4.0 * Lp_ds * N_osm))


def run_nephron(gfr=k.GFR_NORMAL,
                plasma=None,
                ADH=0.65,
                aldosterone=1.0,
                osm_max=1200.0,
                furosemide=0.0,
                thiazide=0.0,
                sglt2_inhibitor=0.0,
                renal_plasma_flow=None,
                nephron_fraction=1.0,
                n=400,
                params=None):
    """Follow the filtrate through the nephron.

    plasma : dict of plasma concentrations (mmol/L) with keys
             'Na', 'glucose', 'urea', 'drug'.
    ADH, aldosterone : hormone levels (0 none, 1 normal-high; aldosterone up to 3)
    furosemide, thiazide, sglt2_inhibitor : drug block fraction 0..1
    nephron_fraction : fraction of working nephrons (kidney disease < 1). Fewer
             nephrons means fewer transporters and less membrane area, so all
             capacities (J_max, S_max, L_p, P) are scaled by this fraction.
    Returns a dict with profiles along the nephron and urine results.
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    for key in p:
        if key.startswith(("Jmax", "Smax", "Lp", "P_urea")):
            p[key] *= nephron_fraction
    pl = {"Na": k.PLASMA_NA, "glucose": k.PLASMA_GLUCOSE,
          "urea": k.PLASMA_UREA, "drug": 0.02}
    if plasma:
        pl.update(plasma)
    if renal_plasma_flow is None:
        renal_plasma_flow = k.RENAL_BLOOD_FLOW * (1 - k.HEMATOCRIT)

    # loop diuretics weaken the medullary gradient too
    osm_max_eff = 290.0 + (osm_max - 290.0) * (1.0 - furosemide)

    # what enters the tubule = what is filtered (small solutes pass freely)
    # F and N are numpy arrays: one value per parallel tube (1 tube until the
    # collecting duct, where we split the flow into ducts of different
    # permeability, because real ducts are not all identical).
    F = np.array([gfr])
    N = {s: np.array([gfr * pl[s]]) for s in ["Na", "glucose", "urea", "drug"]}
    filtered = {s: float(N[s][0]) for s in N}
    spread = np.asarray(p["CD_spread"], dtype=float)

    # drug secretion: limited by transporter capacity AND by the plasma that
    # reaches the tubule after the glomerulus (peritubular capillaries)
    post_glom_plasma = max(renal_plasma_flow - gfr, 0.0)
    drug_secretion_total = min(p["Smax_drug"] * pl["drug"] / (p["Km_drug"] + pl["drug"]),
                               0.92 * post_glom_plasma * pl["drug"])

    ds = 1.0 / n
    rows = []          # one row per grid point
    pos = 0.0          # running position along the whole nephron (0..5)

    def osm(F, N):
        return (2 * N["Na"].sum() + N["glucose"].sum() + N["urea"].sum() + N["drug"].sum()) / F.sum()

    def record(seg, s):
        d = _segment_depth(seg, s)
        rows.append((pos + s, d, F.sum(), N["Na"].sum(), N["glucose"].sum(), N["urea"].sum(),
                     N["drug"].sum(), osm(F, N), float(interstitial_osm(d, osm_max_eff)),
                     SEGMENTS.index(seg)))

    for seg in SEGMENTS:
        if seg == "CD":                      # split into parallel ducts
            m = len(spread)
            F = np.repeat(F / m, m)
            N = {key: np.repeat(N[key] / m, m) for key in N}
        for i in range(n):
            s = i * ds
            record(seg, s)
            d = _segment_depth(seg, s + 0.5 * ds)
            osm_i = float(interstitial_osm(d, osm_max_eff))
            C = {key: N[key] / F for key in N}

            if seg == "PT":
                N["Na"] = N["Na"] - p["k_Na_PT"] * N["Na"] * ds
                J = (1 - sglt2_inhibitor) * p["Jmax_glu"] * C["glucose"] / (p["Km_glu"] + C["glucose"])
                N["glucose"] = np.maximum(N["glucose"] - J * ds, 0.0)
                N["urea"] = N["urea"] - np.maximum(p["P_urea_PT"] * (C["urea"] - pl["urea"]), 0.0) * ds
                N["drug"] = N["drug"] + drug_secretion_total * ds
                Lp = p["Lp_PT"]
            elif seg == "DL":
                Lp = p["Lp_DL"]
            elif seg == "AL":
                J = (1 - furosemide) * p["Jmax_AL"] * C["Na"] / (p["Km_AL"] + C["Na"])
                N["Na"] = np.maximum(N["Na"] - J * ds, 0.0)
                Lp = 0.0
            elif seg == "DCT":
                J = (1 - thiazide) * p["Jmax_DCT"] * C["Na"] / (p["Km_DCT"] + C["Na"])
                N["Na"] = np.maximum(N["Na"] - J * ds, 0.0)
                Lp = 0.0
            else:  # CD: each parallel duct has its own permeability (Lp / m per duct)
                N["Na"] = N["Na"] - (p["k_Na_CD0"] + p["k_Na_CD"] * aldosterone) * N["Na"] * ds
                Lp = p["Lp_CD0"] * (p["Lp_CDmax"] / p["Lp_CD0"]) ** ADH * spread / len(spread)
            N_osm = 2 * N["Na"] + N["glucose"] + N["urea"] + N["drug"]
            F = _water_step(F, N_osm, osm_i, Lp * ds)
        pos += 1.0
    record("CD", 1.0)
    F = F.sum()
    N = {key: float(N[key].sum()) for key in N}

    def osm(F, N):   # scalar version for the results below
        return (2 * N["Na"] + N["glucose"] + N["urea"] + N["drug"]) / F

    a = np.array(rows)
    prof = {
        "position": a[:, 0], "depth": a[:, 1], "F": a[:, 2],
        "N_Na": a[:, 3], "N_glucose": a[:, 4], "N_urea": a[:, 5], "N_drug": a[:, 6],
        "osm": a[:, 7], "osm_interstitium": a[:, 8], "segment": a[:, 9].astype(int),
    }
    for key in ["Na", "glucose", "urea", "drug"]:
        prof["C_" + key] = prof["N_" + key] / prof["F"]

    urine_flow = F
    excreted = dict(N)
    result = {
        "profile": prof,
        "urine_flow": urine_flow,                     # L/min
        "urine_flow_ml_min": urine_flow * 1000,
        "urine_l_day": urine_flow * 1440,
        "urine_osm": osm(F, N),
        "urine_conc": {key: N[key] / F for key in N},
        "filtered": filtered,
        "excreted": excreted,                          # mmol/min
        "fractional_excretion": {key: (excreted[key] / filtered[key] if filtered[key] > 0 else np.nan)
                                 for key in N},
        "clearance": {key: (excreted[key] / pl[key] if pl[key] > 0 else 0.0) for key in N},   # L/min
        "gfr": gfr,
        "plasma": pl,
    }
    # fraction of filtered Na (and water) still left at the END of each segment
    ends = {}
    for j, seg in enumerate(SEGMENTS):
        idx = np.where(prof["segment"] == j)[0][-1] + 1
        idx = min(idx, len(prof["F"]) - 1)
        ends[seg] = {"Na_left": prof["N_Na"][idx] / filtered["Na"],
                     "water_left": prof["F"][idx] / gfr,
                     "osm": prof["osm"][idx]}
    result["segment_end"] = ends
    return result


def glucose_titration(glucose_levels=None, **kw):
    """Classic 'titration curve': filtered, reabsorbed and excreted glucose
    (mmol/min) as plasma glucose rises."""
    if glucose_levels is None:
        glucose_levels = np.linspace(0, 40, 41)
    filt, reab, exc = [], [], []
    for g in glucose_levels:
        r = run_nephron(plasma={"glucose": g}, n=200, **kw)
        filt.append(r["filtered"]["glucose"])
        exc.append(r["excreted"]["glucose"])
        reab.append(filt[-1] - exc[-1])
    return np.asarray(glucose_levels), np.array(filt), np.array(reab), np.array(exc)


def summary(r):
    """Short text summary of a run_nephron() result."""
    lines = [
        f"GFR                   {r['gfr']*1000:7.1f} mL/min",
        f"Urine flow            {r['urine_flow_ml_min']:7.2f} mL/min  ({r['urine_l_day']:.2f} L/day)",
        f"Urine osmolarity      {r['urine_osm']:7.0f} mOsm/L",
        f"Water reabsorbed      {100*(1-r['urine_flow']/r['gfr']):7.2f} %",
    ]
    for key in ["Na", "glucose", "urea"]:
        lines.append(f"FE {key:<8s}           {100*r['fractional_excretion'][key]:7.2f} %")
    lines.append(f"Drug clearance        {r['clearance']['drug']*1000:7.0f} mL/min")
    return "\n".join(lines)
