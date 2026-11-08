"""
LEVEL 3 - The loop of Henle as a COUNTERCURRENT MULTIPLIER
===========================================================

How can the kidney make urine 4x saltier than blood? The loop of Henle is a
hairpin: fluid goes DOWN one side and back UP the other. The ascending side
pumps salt out (but water cannot follow); the descending side lets water out
(so it equilibrates with the tissue). Repeating this small "single effect"
while the fluid flows multiplies it into a big gradient along the loop.

Box model (classic textbook algorithm)
--------------------------------------
Divide the loop into n levels (top = cortex, bottom = papilla). At level i:
    D_i  osmolarity of the descending limb (= tissue around it)
    A_i  osmolarity of the ascending limb

Repeat many cycles:

 1. SINGLE EFFECT (NKCC2 pumps in the ascending limb). At each level the
    pumps create a difference "delta" between the limbs, conserving the total:
        m_i = (D_i + A_i) / 2
        D_i = m_i + delta / 2
        A_i = m_i - delta / 2

 2. WASHOUT. Blood vessels (vasa recta) carry a little salt away:
        D_i = D_i - w * (D_i - 300)
        A_i = A_i - w * (A_i - 300)

 3. FLOW. Everything moves one box along the hairpin:
        fresh fluid at 300 mOsm/L enters the top of D,
        the bottom of D turns the corner into the bottom of A,
        the top of A leaves the loop (now DILUTE, ~100-200 mOsm/L).

Without washout the tip grows by about "delta" per level, so longer loops
and stronger pumps make more concentrated urine (desert animals have very
long loops). Washout limits the gradient, like in a real kidney.
"""

import numpy as np


def multiplier(n_levels=10, delta=200.0, n_cycles=300, inflow=300.0, washout=0.02):
    """Run the box model.

    Returns dict with:
        history_D, history_A : arrays (cycles+1, n_levels), state after the
                               pumping step of every cycle
        D, A      : final profiles (top -> bottom)
        tip       : osmolarity at the bend
        outflow   : osmolarity leaving the ascending limb
        depth     : depth of each level (0 top, 1 bottom)
    """
    D = np.full(n_levels, inflow, dtype=float)
    A = np.full(n_levels, inflow, dtype=float)
    hD, hA = [D.copy()], [A.copy()]
    for _ in range(n_cycles):
        # 1) single effect
        m = 0.5 * (D + A)
        D = m + delta / 2
        A = m - delta / 2
        # 2) washout by blood flow
        D = D - washout * (D - inflow)
        A = A - washout * (A - inflow)
        hD.append(D.copy())
        hA.append(A.copy())
        # 3) flow one box around the hairpin
        bottom = D[-1]
        D = np.concatenate(([inflow], D[:-1]))
        A = np.concatenate((A[1:], [bottom]))
    D, A = hD[-1], hA[-1]
    return {"history_D": np.array(hD), "history_A": np.array(hA),
            "D": D, "A": A, "tip": D[-1], "outflow": A[0],
            "depth": (np.arange(n_levels) + 0.5) / n_levels}


def tip_vs_loop_length(lengths=range(2, 31, 2), delta=200.0, washout=0.02):
    """Tip osmolarity for loops of different length (number of levels)."""
    lengths = np.array(list(lengths))
    tips = np.array([multiplier(n, delta, n_cycles=40 * n, washout=washout)["tip"]
                     for n in lengths])
    return lengths, tips


def levels_needed(target_tip=1200.0, delta=200.0, washout=0.02, max_levels=200):
    """Smallest loop (number of levels) that reaches target_tip, or None."""
    for n in range(2, max_levels):
        if multiplier(n, delta, n_cycles=40 * n, washout=washout)["tip"] >= target_tip:
            return n
    return None
