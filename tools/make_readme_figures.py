"""Regenerates the PNG figures used in README.md:  python tools/make_readme_figures.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import kidneysim as ks
from kidneysim import viz2d, viz3d

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "img")
os.makedirs(out, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(out, name), dpi=110, bbox_inches="tight")
    matplotlib.pyplot.close(fig)
    print("saved", name)

runs = [(f"{int(f*100)} % kidney function", *ks.body.simulate_waste(14, f)) for f in [1, 0.5, 0.15]]
save(viz2d.plot_tank(runs), "level0_tank.png")
f = ks.glomerulus.filtration_along_capillary()
save(viz2d.plot_capillary(f), "level1_starling.png")
save(viz3d.glomerulus_3d(f, backend="matplotlib"), "level1_glomerulus_3d.png")
save(viz3d.gfr_surface(backend="matplotlib"), "level1_gfr_surface_3d.png")
r = ks.tubule.run_nephron()
save(viz2d.plot_nephron_profiles(r), "level2_profiles.png")
save(viz3d.nephron_3d(r, backend="matplotlib"), "level2_nephron_3d.png")
save(viz2d.plot_glucose_titration(*ks.tubule.glucose_titration()), "level2_glucose.png")
cc = ks.countercurrent.multiplier(n_cycles=150)
save(viz2d.plot_countercurrent(cc), "level3_countercurrent.png")
save(viz3d.countercurrent_surface(cc, backend="matplotlib"), "level3_surface_3d.png")
save(viz2d.plot_body({n: ks.coupled.run_scenario(n, days=3)
                      for n in ["healthy", "dehydration", "drink_5_litres", "kidney_failure"]}), "level4_body.png")
p = ks.dialysis.dialyzer_profile()
save(viz2d.plot_dialyzer(p), "level5_dialyzer.png")
save(viz2d.plot_treatment(ks.dialysis.treatment()), "level5_treatment.png")
col = ks.dialysis.ion_exchange_column()
save(viz2d.plot_breakthrough(col, ks.dialysis.breakthrough_time(col)), "level5_ion_exchange.png")
save(viz3d.column_surface(col, backend="matplotlib"), "level5_column_3d.png")
