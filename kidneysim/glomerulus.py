"""
LEVEL 1 - The glomerulus: a filter driven by PRESSURE
=====================================================

Every minute about 1.1 L of blood enters the two kidneys. In the glomerulus
(a ball of tiny capillaries) blood pressure pushes ~20 % of the plasma water
through a filter. Cells and big proteins stay in the blood; water, salts,
glucose and urea pass. No chemistry is needed: this step is pure physics.

1) Blood flow through two resistances in series (like an electric circuit)
--------------------------------------------------------------------------
    Q_blood = (MAP - P_vein) / (R_a + R_e)          "Ohm's law" for blood
    P_GC    = MAP - Q_blood * R_a                   pressure inside the glomerulus

    R_a  resistance of the afferent arteriole (vessel IN)
    R_e  resistance of the efferent arteriole (vessel OUT)

2) Starling equation: filtration along the capillary
----------------------------------------------------
    dQ_f/dx = K_f * [ (P_GC - P_BS) - (pi_GC(x) - pi_BS) ]      x from 0 to 1

    P_GC  hydrostatic pressure in the capillary (pushes water OUT)
    P_BS  pressure in Bowman's space (pushes back)
    pi_GC oncotic pressure of plasma proteins (pulls water IN)
    K_f   filtration coefficient = permeability x area

Because water leaves but protein stays, protein gets more concentrated
along the capillary and pi_GC rises:

    c_protein(x) = c_protein(0) * Q_plasma(0) / Q_plasma(x)
    pi(c) = 2.1 c + 0.16 c^2 + 0.009 c^3      (Landis-Pappenheimer, c in g/dL)

GFR is the total filtered flow:  GFR = integral of dQ_f/dx from 0 to 1.

3) Autoregulation (myogenic response)
-------------------------------------
When blood pressure rises, the afferent arteriole squeezes (R_a goes up) so
blood flow stays almost constant between about 80 and 180 mmHg:

    R_a = (MAP - P_vein) / Q_target - R_e      limited to [0.5, 3] x normal

4) Sieving: which molecules pass the filter?
--------------------------------------------
    lambda = r_molecule / r_pore
    theta  = (1 - lambda)^2 * (2 - (1 - lambda)^2)   (Ferry's pore model)
    theta is multiplied by a "charge penalty" for negatively charged
    molecules like albumin, because the filter itself is negatively charged.
"""

import numpy as np

from . import constants as k

# normal resistances chosen so that Q = 1.1 L/min and P_GC = 60 mmHg at MAP 100
_Q0 = k.RENAL_BLOOD_FLOW
R_A_NORMAL = (k.MAP - k.P_GLOMERULUS) / _Q0                 # ~36 mmHg/(L/min)
R_E_NORMAL = (k.P_GLOMERULUS - k.P_RENAL_VEIN) / _Q0        # ~51 mmHg/(L/min)
KF_NORMAL = 0.0105    # L/min/mmHg (both kidneys); gives GFR ~ 125 mL/min


def oncotic_pressure(c_g_dl):
    """Landis-Pappenheimer equation: protein concentration (g/dL) -> mmHg."""
    c = np.asarray(c_g_dl, dtype=float)
    return 2.1 * c + 0.16 * c**2 + 0.009 * c**3


def hemodynamics(MAP=k.MAP, R_a=R_A_NORMAL, R_e=R_E_NORMAL, autoregulation=False):
    """Return blood flow, plasma flow and glomerular pressure.

    If autoregulation=True, R_a is adjusted to keep blood flow at its normal value.
    """
    if autoregulation:
        R_a = (MAP - k.P_RENAL_VEIN) / _Q0 - R_e
        R_a = float(np.clip(R_a, 0.5 * R_A_NORMAL, 3.0 * R_A_NORMAL))
    Q_blood = max(MAP - k.P_RENAL_VEIN, 0.0) / (R_a + R_e)
    P_gc = MAP - Q_blood * R_a
    return {
        "Q_blood": Q_blood,
        "Q_plasma": Q_blood * (1 - k.HEMATOCRIT),
        "P_gc": P_gc,
        "R_a": R_a,
        "R_e": R_e,
    }


def filtration_along_capillary(P_gc=k.P_GLOMERULUS, Q_plasma=None, Kf=KF_NORMAL,
                               P_bs=k.P_BOWMAN, protein=k.PLASMA_PROTEIN,
                               pressure_drop=2.0, n=200):
    """Integrate the Starling equation along a glomerular capillary.

    Returns a dict with x, pressures along the capillary and the GFR (L/min).
    pressure_drop: how much P_GC falls from start to end of the capillary (mmHg).
    """
    if Q_plasma is None:
        Q_plasma = k.RENAL_BLOOD_FLOW * (1 - k.HEMATOCRIT)
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    Qp = np.empty(n)
    Qp[0] = Q_plasma
    P = P_gc - pressure_drop * x
    c = np.empty(n)
    pi = np.empty(n)
    nfp = np.empty(n)
    for i in range(n):
        c[i] = protein * Q_plasma / Qp[i]
        pi[i] = oncotic_pressure(c[i])
        nfp[i] = max((P[i] - P_bs) - pi[i], 0.0)    # filtration stops at equilibrium
        if i < n - 1:
            Qp[i + 1] = Qp[i] - Kf * nfp[i] * dx
    gfr = Q_plasma - Qp[-1]
    return {
        "x": x, "P_gc": P, "P_bs": np.full(n, P_bs), "pi_gc": pi,
        "net_pressure": nfp, "protein": c, "Q_plasma": Qp,
        "gfr": gfr, "filtration_fraction": gfr / Q_plasma,
    }


def gfr_from_map(MAP, autoregulation=True, Kf=KF_NORMAL, R_e=R_E_NORMAL):
    """GFR (L/min) for a given mean arterial pressure."""
    h = hemodynamics(MAP, R_e=R_e, autoregulation=autoregulation)
    f = filtration_along_capillary(P_gc=h["P_gc"], Q_plasma=h["Q_plasma"], Kf=Kf)
    return f["gfr"]


# ---- sieving -----------------------------------------------------------------
MOLECULES = {
    # name: (radius in nm, negatively charged?)
    "urea": (0.26, False),
    "glucose": (0.36, False),
    "inulin": (1.5, False),
    "myoglobin": (1.9, False),
    "hemoglobin": (3.25, False),
    "albumin": (3.6, True),
}


def sieving_coefficient(radius_nm, r_pore_nm=4.5, charged=False, charge_penalty=0.05):
    """Fraction of a molecule's plasma concentration that appears in the filtrate."""
    lam = np.clip(np.asarray(radius_nm, dtype=float) / r_pore_nm, 0, 1)
    phi = (1 - lam) ** 2
    theta = phi * (2 - phi)
    return theta * (charge_penalty if charged else 1.0)
