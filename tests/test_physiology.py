"""
Checks that the model gives textbook values for a healthy adult.
Run with:  python -m pytest tests   (or simply: python tests/test_physiology.py)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import kidneysim as ks


def test_level0_steady_state():
    t, C = ks.body.simulate_waste(days=14, kidney_function=1.0)
    assert 2.5 < C[-1] < 7.5                       # normal plasma urea
    t, C = ks.body.simulate_waste(days=30, kidney_function=0.5)
    assert abs(C[-1] - 2 * 5.0) < 0.6              # half function -> double urea


def test_level1_gfr():
    f = ks.glomerulus.filtration_along_capillary()
    assert 110 < f["gfr"] * 1000 < 140             # mL/min
    assert 0.15 < f["filtration_fraction"] < 0.25
    # autoregulation: GFR almost constant between 80 and 180 mmHg
    g = [ks.glomerulus.gfr_from_map(m) for m in (90, 120, 170)]
    assert max(g) - min(g) < 0.01


def test_level1_sieving():
    assert ks.glomerulus.sieving_coefficient(0.26) > 0.95          # urea passes
    assert ks.glomerulus.sieving_coefficient(3.6, charged=True) < 0.01   # albumin stays


def test_level2_normal_urine():
    r = ks.tubule.run_nephron()
    assert 0.4 < r["urine_flow_ml_min"] < 2.0
    assert r["fractional_excretion"]["glucose"] < 0.001
    assert 0.003 < r["fractional_excretion"]["Na"] < 0.01
    assert 0.35 < r["fractional_excretion"]["urea"] < 0.6
    assert 0.55 < 1 - r["segment_end"]["PT"]["Na_left"] < 0.75    # PT reabsorbs ~65 % Na


def test_level2_adh():
    dilute = ks.tubule.run_nephron(ADH=0.0)
    conc = ks.tubule.run_nephron(ADH=1.0)
    assert dilute["urine_osm"] < 100 and dilute["urine_flow_ml_min"] > 10
    assert conc["urine_osm"] > 1000 and conc["urine_flow_ml_min"] < 0.6


def test_level2_glucose_threshold():
    g, filt, reab, exc = ks.tubule.glucose_titration(np.arange(6, 20, 1.0))
    threshold = g[np.argmax(exc > 0.005)]
    assert 9 <= threshold <= 14


def test_level3_gradient():
    res = ks.countercurrent.multiplier()
    assert 1000 < res["tip"] < 1400
    assert res["outflow"] < 290                     # ascending limb dilutes


def test_level4_homeostasis():
    out = ks.coupled.simulate(days=2)
    assert abs(out["P_Na"][-1] - 140) < 3
    assert abs(out["V"][-1] - 42) < 0.5


def test_level5_dialysis():
    K = ks.dialysis.dialyzer_clearance(0.3, 0.5, 1.0)
    assert 0.22 < K < 0.30
    prof = ks.dialysis.dialyzer_profile(0.3, 0.5, 1.0)
    assert abs(prof["clearance"] - K) < 1e-6        # numeric profile = Michaels formula
    tr = ks.dialysis.treatment(weeks=1)
    assert tr["Kt_V"][0] > 1.2
    col = ks.dialysis.ion_exchange_column(t_end_min=480)
    assert 150 < ks.dialysis.breakthrough_time(col) < 320


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
