"""
LEVEL 0 - The body as one well-mixed tank
=========================================

Idea
----
Imagine all the water in the body (about 42 L) as one big tank.
The liver keeps adding urea to the tank, and the kidneys keep removing it.

Governing equation (mass balance)
---------------------------------
    V * dC/dt = G - K * C

    C  plasma concentration of the waste            [mmol/L]
    V  volume it is dissolved in (body water)       [L]
    G  generation rate (liver for urea)             [mmol/min]
    K  clearance = volume of plasma the kidney      [L/min]
       "cleans completely" every minute

Steady state (dC/dt = 0):   C_ss = G / K
Time constant:              tau  = V / K

Solved numerically with Euler's method:
    C(t + dt) = C(t) + dt * (G - K * C(t)) / V
"""

import numpy as np

from . import constants as k


def euler_tank(G, K, V, C0, t_end_min, dt_min=10.0, K_of_t=None):
    """Integrate V dC/dt = G - K C with Euler steps.

    K_of_t : optional function t -> clearance (L/min), used for dialysis
             sessions or a kidney that gets worse over time.
    Returns (t in minutes, C).
    """
    n = int(t_end_min / dt_min) + 1
    t = np.linspace(0.0, t_end_min, n)
    C = np.empty(n)
    C[0] = C0
    for i in range(n - 1):
        Ki = K_of_t(t[i]) if K_of_t is not None else K
        C[i + 1] = C[i] + dt_min * (G - Ki * C[i]) / V
    return t, C


def urea_clearance(gfr=k.GFR_NORMAL, fraction_excreted=0.45):
    """Urea is freely filtered, then ~55 % diffuses back -> K = FE * GFR."""
    return fraction_excreted * gfr


def simulate_waste(days=14, kidney_function=1.0, solute="urea", C0=None, dt_min=10.0):
    """Simulate plasma urea or creatinine for a given fraction of kidney function.

    kidney_function = 1.0 healthy, 0.5 = half the nephrons, 0.1 = kidney failure
    Returns (t_days, C [mmol/L]).
    """
    V = k.BODY_WATER_L
    if solute == "urea":
        G = k.UREA_GENERATION
        K = urea_clearance(k.GFR_NORMAL * kidney_function)
        C0 = k.PLASMA_UREA if C0 is None else C0
    elif solute == "creatinine":
        G = k.CREATININE_GENERATION
        K = k.GFR_NORMAL * kidney_function   # creatinine: clearance ~ GFR
        C0 = k.PLASMA_CREATININE if C0 is None else C0
    else:
        raise ValueError("solute must be 'urea' or 'creatinine'")
    t, C = euler_tank(G, K, V, C0, days * 1440.0, dt_min)
    return t / 1440.0, C


def steady_state(G, K):
    """C_ss = G / K."""
    return G / K
