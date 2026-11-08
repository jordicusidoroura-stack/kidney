"""
LEVEL 4 - Kidney + body together, with hormones (feedback loops)
=================================================================

Now we connect everything:

    Level 0 (body tank)  <->  Level 1 (glomerulus)  ->  Level 2 (tubule)
                                         ^                     |
                                         |  Level 3 sets the   |
                                         |  medulla gradient   |
                                         +------ hormones -----+

Body state (all body water treated as one tank)
-----------------------------------------------
    V    body water                      [L]
    S    salt osmoles in the body        [mOsm]   (Na+ and its anion)
    U    urea in the body                [mmol]
    Cr   creatinine in the body          [mmol]

Plasma values:
    P_Na     = S / (2 V)
    P_osm    = (S + U) / V + glucose          (what a lab measures)
    tonicity = S / V + glucose                (urea crosses cell membranes freely,
                                               so it does not pull water out of cells)

Balance equations (what comes in - what goes out)
-------------------------------------------------
    dV/dt  = water_intake + thirst - urine_flow - insensible_loss
    dS/dt  = 2 * (salt_intake - Na_excreted)
    dU/dt  = G_urea - urea_excreted
    dCr/dt = G_creatinine - GFR * P_creatinine

Hormones (negative feedback)
----------------------------
ADH (from the brain) rises when blood gets too concentrated:
    ADH = clip(0.65 + 0.08 * (tonicity - 285), 0, 1)

Thirst makes us drink more when tonicity rises (if water is available):
    thirst = clip(0.5 * (tonicity - 287), 0, 10)       [L/day]

Aldosterone (from the adrenal gland, via renin-angiotensin) rises when
body water / volume or blood pressure falls:
    aldosterone = clip(1 + 30 (V0 - V)/V0 + 0.05 (100 - MAP), 0, 3)

Blood pressure falls a little when volume falls:
    MAP = MAP_set * (1 + 2 * (V - V0) / V0)

Each time step we:
  1. compute plasma values and hormones,
  2. compute GFR (Level 1) and run the nephron (Level 2),
  3. update the body with Euler's method.
"""

import numpy as np

from . import constants as k
from . import glomerulus as glom
from . import tubule as tub
from . import countercurrent as cc

V0 = k.BODY_WATER_L
S0 = 2 * k.PLASMA_NA * V0

SCENARIOS = {
    "healthy":        dict(),
    "dehydration":    dict(water_intake=0.3, thirst=False),
    "drink_5_litres": dict(water_intake=5.0),
    "salty_diet":     dict(salt_intake=300.0),
    "diabetes":       dict(glucose=20.0),
    "kidney_failure": dict(nephron_fraction=0.15),
    "low_pressure":   dict(MAP_set=65.0),
    "furosemide":     dict(furosemide=0.3),
}


def simulate(days=5.0, dt_min=30.0,
             water_intake=2.3,          # L/day (drinks + food + metabolic water)
             salt_intake=150.0,         # mmol Na/day
             glucose=k.PLASMA_GLUCOSE,  # mmol/L (held constant)
             MAP_set=k.MAP,
             nephron_fraction=1.0,      # 1 = healthy, 0.1 = severe kidney failure
             furosemide=0.0, thiazide=0.0, sglt2_inhibitor=0.0,
             insensible_loss=0.6,       # L/day (breath, skin)
             thirst=True,               # False = no water available
             start=None, n_grid=150):
    """Simulate the body + kidney for a number of days.

    start: optional dict with initial V, S, U, Cr (e.g. the end of another run).
    Returns a dict of time series (time in days).
    """
    s = {"V": V0, "S": S0, "U": k.PLASMA_UREA * V0, "Cr": k.PLASMA_CREATININE * V0}
    if start:
        s.update(start)

    # Level 3: the loop sets the maximum medullary osmolarity.
    # (Furosemide weakens it further inside tubule.run_nephron.)
    osm_max = cc.multiplier()["tip"]

    n = int(days * 1440 / dt_min) + 1
    keys = ["t", "V", "P_Na", "P_osm", "P_urea", "P_creat", "ADH", "aldosterone", "drink",
            "MAP", "GFR", "urine_flow", "urine_osm", "Na_excr", "glucose_excr"]
    out = {key: np.zeros(n) for key in keys}

    for i in range(n):
        V = s["V"]
        P_na = s["S"] / (2 * V)
        P_urea = s["U"] / V
        P_cr = s["Cr"] / V
        P_osm = (s["S"] + s["U"]) / V + glucose
        tonicity = s["S"] / V + glucose

        ADH = float(np.clip(0.65 + 0.08 * (tonicity - 285.0), 0, 1))
        drink = float(np.clip(0.5 * (tonicity - 287.0), 0, 10)) if thirst else 0.0
        MAP = MAP_set * (1 + 2 * (V - V0) / V0)
        aldo = float(np.clip(1 + 30 * (V0 - V) / V0 + 0.05 * (100 - MAP), 0, 3))

        gfr = glom.gfr_from_map(MAP, Kf=glom.KF_NORMAL * nephron_fraction)
        if gfr > 1e-6:
            r = tub.run_nephron(gfr=gfr,
                                plasma={"Na": P_na, "glucose": glucose, "urea": P_urea},
                                ADH=ADH, aldosterone=aldo, osm_max=osm_max,
                                furosemide=furosemide, thiazide=thiazide,
                                sglt2_inhibitor=sglt2_inhibitor,
                                nephron_fraction=nephron_fraction, n=n_grid)
            urine, ex = r["urine_flow"], r["excreted"]
            uosm = r["urine_osm"]
        else:
            urine, ex, uosm = 0.0, {"Na": 0.0, "urea": 0.0, "glucose": 0.0}, np.nan

        row = dict(t=i * dt_min / 1440, V=V, P_Na=P_na, P_osm=P_osm, P_urea=P_urea,
                   P_creat=P_cr * 1000, ADH=ADH, aldosterone=aldo,
                   drink=water_intake + drink, MAP=MAP,
                   GFR=gfr * 1000, urine_flow=urine * 1000, urine_osm=uosm,
                   Na_excr=ex["Na"] * 1440, glucose_excr=ex["glucose"] * 1440)
        for key in keys:
            out[key][i] = row[key]

        # Euler update of the body
        s["V"] += dt_min * ((water_intake + drink - insensible_loss) / 1440 - urine)
        s["S"] += dt_min * 2 * (salt_intake / 1440 - ex["Na"])
        s["U"] += dt_min * (k.UREA_GENERATION - ex["urea"])
        s["Cr"] += dt_min * (k.CREATININE_GENERATION - gfr * P_cr)

    out["final_state"] = dict(s)
    out["units"] = {"t": "days", "V": "L", "P_Na": "mmol/L", "P_osm": "mOsm/L",
                    "P_urea": "mmol/L", "drink": "L/day", "P_creat": "umol/L", "MAP": "mmHg",
                    "GFR": "mL/min", "urine_flow": "mL/min", "urine_osm": "mOsm/L",
                    "Na_excr": "mmol/day", "glucose_excr": "mmol/day"}
    return out


def run_scenario(name, days=5.0, **extra):
    """Run one of the named SCENARIOS, starting from a healthy steady state."""
    params = dict(SCENARIOS[name])
    params.update(extra)
    return simulate(days=days, **params)
