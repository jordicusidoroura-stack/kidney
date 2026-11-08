"""
Builds the Colab notebooks in ../notebooks from the cell lists below.
Edit the text here and run:   python tools/build_notebooks.py
(Students and teachers can also just edit the .ipynb files directly.)
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "notebooks")
REPO = "YOUR-USERNAME/kidney-sim"   # <- change to your GitHub user/repo

SETUP = '''# --- Setup: run this cell first --------------------------------------------
import os, sys, subprocess
REPO_URL = "https://github.com/%s.git"   # change if you forked the project
if "google.colab" in sys.modules and not os.path.exists("kidneysim"):
    if not os.path.exists("kidney-sim"):
        subprocess.run(["git", "clone", "-q", REPO_URL], check=True)
    os.chdir("kidney-sim")
for path in [".", ".."]:
    if os.path.exists(os.path.join(path, "kidneysim")):
        sys.path.insert(0, os.path.abspath(path))
        break

import numpy as np
import matplotlib.pyplot as plt
import kidneysim as ks
from kidneysim import viz2d, viz3d
from kidneysim.ui import sliders
print("kidneysim", ks.__version__, "| interactive 3D (plotly):", viz3d.HAS_PLOTLY)''' % REPO


def badge(name):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name})")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n")}


def write(name, cells):
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}},
        "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i}"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", name)


# =============================================================================
NB0 = "00_body_tank.ipynb"
write(NB0, [
    md(f"""
# Level 0 - The body as a tank of water
{badge(NB0)}

**Goal:** understand *why* we need kidneys with the simplest possible model.

**You will learn:** mass balance, clearance, steady state, Euler's method.
"""),
    code(SETUP),
    md(r"""
## 1. The idea

The liver produces **urea** all the time (it is how we get rid of the nitrogen from proteins).
Urea dissolves in all the water of the body, about **42 L**. The kidneys remove it.

Think of a bathtub: the tap is the liver, the drain is the kidney.

## 2. The governing equation (mass balance)

$$ V\,\frac{dC}{dt} = G - K\,C $$

| symbol | meaning | normal value |
|---|---|---|
| $C$ | plasma urea concentration | 5 mmol/L |
| $V$ | body water | 42 L |
| $G$ | urea made by the liver | 0.28 mmol/min |
| $K$ | **clearance**: litres of plasma completely cleaned per minute | 0.056 L/min |

When nothing changes any more ($dC/dt = 0$) we reach the **steady state**:
$$ C_{ss} = \frac{G}{K} $$

## 3. Solving it with Euler's method

A computer cannot do calculus directly, so we take small time steps $\Delta t$:

$$ C(t+\Delta t) = C(t) + \Delta t\,\frac{G - K\,C(t)}{V} $$
"""),
    code('''
# Euler's method written out by hand
V = 42.0        # L
G = 0.28        # mmol/min
K = 0.056       # L/min  (healthy kidneys)
dt = 10.0       # min
C = 5.0         # mmol/L, starting value

times, values = [0.0], [C]
for step in range(int(14 * 1440 / dt)):      # 14 days
    C = C + dt * (G - K * C) / V
    times.append((step + 1) * dt / 1440)     # in days
    values.append(C)

print(f"after 14 days: C = {C:.2f} mmol/L,  steady state G/K = {G/K:.2f} mmol/L")
'''),
    md("""
## 4. What happens if the kidneys fail?

The package function `ks.body.simulate_waste` does exactly the loop above.
Let's compare healthy kidneys, half of the kidney (e.g. after donating one kidney) and kidney failure.
"""),
    code('''
runs = []
for f in [1.0, 0.5, 0.15]:
    t, C = ks.body.simulate_waste(days=14, kidney_function=f)
    runs.append((f"{int(f*100)} % kidney function", t, C))
viz2d.plot_tank(runs);
'''),
    md("""
### Try it yourself
Move the slider. Creatinine (from muscles) is the substance doctors use to estimate kidney function.
"""),
    code('''
def experiment(kidney_function=1.0, solute=["urea", "creatinine"]):
    t, C = ks.body.simulate_waste(days=21, kidney_function=kidney_function, solute=solute)
    viz2d.plot_tank([(f"{kidney_function:.0%} function", t, C)], solute=solute)
    plt.show()

sliders(experiment, kidney_function=(0.05, 1.0, 0.05, 1.0), solute=["urea", "creatinine"])
'''),
    md("""
## 5. Questions
1. If kidney function halves, what happens to the steady-state urea? Can you explain it using $C_{ss}=G/K$?
2. Why does it take *days* to reach the new value? (Hint: the time constant is $\\tau = V/K$.)
3. A high-protein diet increases $G$. What happens to urea?

## What is missing?
We just *chose* $K$. But **how** does the kidney clean the blood? Next level: the filter.
""")
])

# =============================================================================
NB1 = "01_glomerulus_pressure.ipynb"
write(NB1, [
    md(f"""
# Level 1 - The glomerulus: a filter driven by pressure
{badge(NB1)}

**Goal:** compute the GFR (glomerular filtration rate) from physics: pressures and flows.

**Answer to the question "does pressure matter in the body?":** in the glomerulus, pressure is *everything*.
"""),
    code(SETUP),
    md(r"""
## 1. Blood flow through the kidney = an electric circuit

Blood enters through the **afferent** arteriole, crosses the glomerulus and leaves through the **efferent** arteriole.
Each vessel is a resistance:

$$ Q_{blood} = \frac{MAP - P_{vein}}{R_a + R_e} \qquad\qquad P_{GC} = MAP - Q_{blood}\,R_a $$

The efferent vessel is narrow, so it keeps the pressure inside the glomerulus high (~60 mmHg).

## 2. Starling equation along the capillary

$$ \frac{dQ_f}{dx} = K_f\Big[\underbrace{(P_{GC}-P_{BS})}_{\text{pushes out}} - \underbrace{(\pi_{GC}-\pi_{BS})}_{\text{pulls back}}\Big] $$

* $P_{GC}$ blood pressure in the capillary, $P_{BS}$ pressure in Bowman's space (15 mmHg)
* $\pi_{GC}$ **oncotic pressure** of proteins. As water leaves, proteins get concentrated:

$$ c(x) = c_0\,\frac{Q_p(0)}{Q_p(x)}, \qquad \pi = 2.1c + 0.16c^2 + 0.009c^3 \;\;(\text{Landis–Pappenheimer}) $$
"""),
    code('''
f = ks.glomerulus.filtration_along_capillary()
print(f"GFR = {f['gfr']*1000:.0f} mL/min  ({f['gfr']*1440:.0f} L/day!)")
print(f"filtration fraction = {100*f['filtration_fraction']:.0f} % of the plasma")
viz2d.plot_capillary(f);
'''),
    md("""
### 3D: the capillary ball, coloured by net filtration pressure
Rotate it with the mouse (in Colab). Red = strong filtration, blue = almost none.
"""),
    code('''
viz3d.show(viz3d.glomerulus_3d(f))
'''),
    md(r"""
## 3. Autoregulation

If blood pressure goes up, the afferent arteriole contracts (myogenic response) so the flow stays the same:

$$ R_a = \frac{MAP-P_{vein}}{Q_{target}} - R_e \qquad (\text{limited to } 0.5\text{–}3\times \text{normal}) $$
"""),
    code('''
maps = np.linspace(40, 200, 81)
auto = [ks.glomerulus.gfr_from_map(m, autoregulation=True) for m in maps]
rigid = [ks.glomerulus.gfr_from_map(m, autoregulation=False) for m in maps]
viz2d.plot_autoregulation(maps, auto, rigid);
'''),
    md("""
### 3D: GFR versus blood pressure *and* filter health
$K_f$ falls when glomeruli are damaged (diabetes, high blood pressure over years).
"""),
    code('''
viz3d.show(viz3d.gfr_surface())
'''),
    md(r"""
## 4. What passes the filter? (sieving)

$$ \lambda = \frac{r_{molecule}}{r_{pore}},\qquad \theta = (1-\lambda)^2\,\big(2-(1-\lambda)^2\big) $$

Albumin is also **negatively charged** and so is the filter: it is pushed away (extra penalty).
"""),
    code('''
radii = np.linspace(0, 4.5, 200)
mols = {name: (r, ks.glomerulus.sieving_coefficient(r, charged=c))
        for name, (r, c) in ks.glomerulus.MOLECULES.items()}
viz2d.plot_sieving(radii, ks.glomerulus.sieving_coefficient(radii), mols);
'''),
    code('''
def experiment(MAP=100.0, efferent_resistance=1.0, filter_health=1.0):
    h = ks.glomerulus.hemodynamics(MAP, R_e=ks.glomerulus.R_E_NORMAL * efferent_resistance,
                                   autoregulation=True)
    f = ks.glomerulus.filtration_along_capillary(P_gc=h["P_gc"], Q_plasma=h["Q_plasma"],
                                                 Kf=ks.glomerulus.KF_NORMAL * filter_health)
    viz2d.plot_capillary(f)
    plt.show()

sliders(experiment, MAP=(40, 200, 5, 100), efferent_resistance=(0.5, 2.0, 0.1, 1.0),
        filter_health=(0.1, 1.0, 0.05, 1.0))
'''),
    md("""
## Questions
1. What happens to GFR during a severe haemorrhage (MAP 50 mmHg)? Why is that dangerous for the kidney?
2. Squeezing the **efferent** arteriole (angiotensin II does this) changes $P_{GC}$. Does GFR go up or down?
3. Why is protein in the urine (albuminuria) a sign of a damaged filter?

## What is missing?
180 L/day are filtered but we only pee ~1.5 L. Where does the rest go? → Level 2.
""")
])

# =============================================================================
NB2 = "02_tubule_transport.ipynb"
write(NB2, [
    md(f"""
# Level 2 - The tubule: chemistry at the membrane
{badge(NB2)}

**Goal:** follow the filtrate along the nephron and see how transporters and osmosis turn
180 L/day of filtrate into ~1.5 L of urine.

Here the **chemical** part of the model lives: pumps, co-transporters and **exchangers**.
"""),
    code(SETUP),
    md(r"""
## 1. Governing equations along the tube (position $s$)

Flow of water $F$ and of each solute $N_i = F\,C_i$:

| process | equation | example |
|---|---|---|
| saturable transporter | $\dfrac{dN_i}{ds} = -J_{max}\dfrac{C_i}{K_m + C_i}$ | glucose (SGLT), salt (NKCC2, NCC) |
| leaky / proportional | $\dfrac{dN_i}{ds} = -k\,N_i$ | Na in proximal tubule, collecting duct (aldosterone) |
| diffusion (Fick) | $\dfrac{dN_u}{ds} = -P\,(C_{u,lumen}-C_{u,plasma})$ | urea |
| **exchanger** secretion | $\dfrac{dN_d}{ds} = +S_{max}\dfrac{C_{plasma}}{K_m + C_{plasma}}$ | drugs/toxins via OAT1 (swaps α-ketoglutarate for the toxin) |
| osmosis | $\dfrac{dF}{ds} = -L_p\,(Osm_{tissue}-Osm_{lumen})$ | water follows salt |

with $Osm_{lumen} = (2N_{Na} + N_{glucose} + N_{urea})/F$.
In the collecting duct the water permeability is set by the hormone **ADH**: $L_p = L_{p0} + L_{p,ADH}\cdot ADH$.
"""),
    code('''
r = ks.tubule.run_nephron()          # normal conditions
print(ks.tubule.summary(r))
viz2d.plot_nephron_profiles(r);
'''),
    md("""
### 2D map and 3D nephron
The grey wall behind the nephron shows the **tissue** osmolarity: 290 in the cortex, up to 1200 mOsm/L deep in the medulla.
Change `quantity` to `"C_urea"`, `"C_Na"`, `"C_glucose"` or `"F"`.
"""),
    code('''
viz2d.plot_nephron_map(r, quantity="osm");
'''),
    code('''
viz3d.show(viz3d.nephron_3d(r, quantity="osm"))
'''),
    md("""
## 2. Glucose: when transporters saturate (diabetes)
Above ~10–12 mmol/L of blood glucose, the SGLT transporters are full and glucose appears in the urine.
"""),
    code('''
g, filt, reab, exc = ks.tubule.glucose_titration()
viz2d.plot_glucose_titration(g, filt, reab, exc);
'''),
    md("""
## 3. Experiment: hormones and drugs
* **ADH** (0 → 1): no ADH = lots of dilute urine; max ADH = little concentrated urine.
* **aldosterone**: more salt kept.
* **furosemide** blocks NKCC2 (ascending limb); **thiazide** blocks NCC (distal tubule); **SGLT2 inhibitor** blocks glucose reabsorption.
"""),
    code('''
def experiment(ADH=0.65, aldosterone=1.0, glucose=5.0, furosemide=0.0, thiazide=0.0, sglt2_inhibitor=0.0):
    r = ks.tubule.run_nephron(ADH=ADH, aldosterone=aldosterone, plasma={"glucose": glucose},
                              furosemide=furosemide, thiazide=thiazide, sglt2_inhibitor=sglt2_inhibitor)
    print(ks.tubule.summary(r))
    viz2d.plot_nephron_profiles(r)
    plt.show()

sliders(experiment, ADH=(0, 1, 0.05, 0.65), aldosterone=(0, 3, 0.1, 1.0), glucose=(3, 30, 1, 5),
        furosemide=(0, 1, 0.1, 0.0), thiazide=(0, 1, 0.1, 0.0), sglt2_inhibitor=(0, 1, 0.1, 0.0))
'''),
    md("""
## Questions
1. Where is most of the water reabsorbed? And most of the salt?
2. Why does a person with uncontrolled diabetes pee a lot (*diabetes mellitus* = "sweet siphon")?
3. Why is the fluid at the end of the ascending limb *more dilute* than blood?
4. The drug clearance is close to the renal plasma flow (~600 mL/min). Which transport process explains this?

## What is missing?
We *assumed* a salt gradient in the medulla. Who builds it? → Level 3.
""")
])

# =============================================================================
NB3 = "03_loop_of_henle_countercurrent.ipynb"
write(NB3, [
    md(f"""
# Level 3 - The loop of Henle: a countercurrent multiplier
{badge(NB3)}

**Goal:** see how a hairpin tube with salt pumps creates a gradient from 300 to 1200 mOsm/L.
"""),
    code(SETUP),
    md(r"""
## 1. Box model

Split the loop into $n$ levels. $D_i$ = descending limb (and tissue), $A_i$ = ascending limb. Repeat:

1. **Single effect** (pumps):  $m_i = \tfrac{D_i + A_i}{2}$, $\;D_i = m_i + \tfrac{\delta}{2}$, $\;A_i = m_i - \tfrac{\delta}{2}$
2. **Washout** by blood: $D_i \leftarrow D_i - w\,(D_i - 300)$ (same for $A_i$)
3. **Flow**: shift everything one box around the hairpin; fresh fluid at 300 mOsm/L enters.

The pumps can only make a difference of $\delta \approx 200$ mOsm/L *at one level*, but flow **multiplies** it along the loop.
"""),
    code('''
res = ks.countercurrent.multiplier(n_levels=10, delta=200, n_cycles=150)
print(f"tip of the loop: {res['tip']:.0f} mOsm/L, fluid leaving: {res['outflow']:.0f} mOsm/L")
viz2d.plot_countercurrent(res);
'''),
    md("### 3D: the gradient growing, cycle after cycle"),
    code('''
viz3d.show(viz3d.countercurrent_surface(res))
'''),
    md("""
## 2. Why do desert animals have long loops?
"""),
    code('''
lengths, tips = ks.countercurrent.tip_vs_loop_length()
viz2d.plot_loop_length(lengths, tips);
'''),
    code('''
def experiment(n_levels=10, delta=200.0, washout=0.02):
    res = ks.countercurrent.multiplier(int(n_levels), delta, n_cycles=200, washout=washout)
    viz2d.plot_countercurrent(res)
    plt.show()

sliders(experiment, n_levels=(2, 30, 1, 10), delta=(0, 300, 10, 200), washout=(0, 0.1, 0.005, 0.02))
'''),
    md("""
## 3. Connect to Level 2
The tip value is the `osm_max` used by the tubule model. A weak loop (e.g. furosemide) → the kidney cannot concentrate urine.
"""),
    code('''
for delta in [200, 100, 50]:
    tip = ks.countercurrent.multiplier(delta=delta)["tip"]
    r = ks.tubule.run_nephron(ADH=1.0, osm_max=tip)
    print(f"pump strength {delta:3d} -> tip {tip:5.0f} mOsm/L -> max urine osmolarity {r['urine_osm']:5.0f} mOsm/L")
'''),
    md("""
## Questions
1. What happens if the pumps stop ($\\delta = 0$)?
2. What does washout represent physically? What would happen if blood flow in the medulla increased a lot?
3. Kangaroo rats can make urine of ~5000 mOsm/L. What would they need in this model?
""")
])

# =============================================================================
NB4 = "04_whole_body_feedback.ipynb"
write(NB4, [
    md(f"""
# Level 4 - Kidney + body + hormones
{badge(NB4)}

**Goal:** put all previous levels together and add the **feedback loops** that keep the body stable
(homeostasis): ADH, thirst and aldosterone.
"""),
    code(SETUP),
    md(r"""
## 1. Balance equations

$$\frac{dV}{dt} = \text{drink} - \text{urine} - \text{insensible loss}$$
$$\frac{dS}{dt} = 2\,(\text{salt intake} - \text{Na excreted}) \qquad \frac{dU}{dt} = G_{urea} - \text{urea excreted}$$

Plasma: $P_{Na} = S/2V$, $\;$ tonicity $= S/V + \text{glucose}$.

## 2. Feedback (hormones)

$$ADH = \text{clip}\big(0.65 + 0.08\,(\text{tonicity}-285),\,0,\,1\big)$$
$$\text{thirst} = \text{clip}\big(0.5\,(\text{tonicity}-287),\,0,\,10\big)\ \text{L/day}$$
$$\text{aldosterone} = \text{clip}\Big(1 + 30\,\frac{V_0 - V}{V_0} + 0.05\,(100 - MAP),\,0,\,3\Big)$$

Every 30 minutes the model computes GFR (Level 1), runs the nephron (Level 2) with the gradient from Level 3, and updates the body.
"""),
    code('''
print("available scenarios:", list(ks.coupled.SCENARIOS))
runs = {name: ks.coupled.run_scenario(name, days=3)
        for name in ["healthy", "dehydration", "drink_5_litres", "diabetes"]}
viz2d.plot_body(runs);
'''),
    code('''
runs = {name: ks.coupled.run_scenario(name, days=5)
        for name in ["healthy", "kidney_failure", "low_pressure", "furosemide", "salty_diet"]}
viz2d.plot_body(runs);
'''),
    md("### Your own experiment (each run takes a few seconds)"),
    code('''
def experiment(water_intake=2.3, salt_intake=150.0, glucose=5.0, nephron_fraction=1.0, MAP_set=100.0):
    out = ks.coupled.simulate(days=3, water_intake=water_intake, salt_intake=salt_intake,
                              glucose=glucose, nephron_fraction=nephron_fraction, MAP_set=MAP_set)
    viz2d.plot_body({"your patient": out, "healthy": ks.coupled.simulate(days=3)})
    plt.show()

sliders(experiment, water_intake=(0, 6, 0.1, 2.3), salt_intake=(20, 400, 10, 150), glucose=(4, 30, 1, 5),
        nephron_fraction=(0.05, 1, 0.05, 1.0), MAP_set=(50, 160, 5, 100))
'''),
    md("""
## Questions
1. When you drink 5 L, what keeps plasma Na normal? Which hormone changes?
2. Without water, ADH saturates at 1. What happens to plasma Na after that? Why is this dangerous?
3. In kidney failure, why does urea keep rising while Na stays almost normal?
4. Find one **negative feedback loop** in the model and draw it as a diagram.
""")
])

# =============================================================================
NB5 = "05_artificial_kidney.ipynb"
write(NB5, [
    md(f"""
# Level 5 - The artificial kidney: membranes + ion exchange
{badge(NB5)}

**Goal:** model a dialysis machine. Here the idea of *ion exchange* is exactly how the machine works.
"""),
    code(SETUP),
    md(r"""
## 1. The dialyzer: diffusion through a membrane

Blood and dialysate flow in **opposite** directions (countercurrent, like the loop of Henle!):

$$Q_b\frac{dC_b}{dx} = -K_oA\,(C_b - C_d) \qquad Q_d\frac{dC_d}{dx} = -K_oA\,(C_b - C_d)$$

Solution (Michaels): $\;Z = Q_b/Q_d,\; NTU = K_oA/Q_b$

$$E = \frac{1-e^{NTU(1-Z)}}{Z-e^{NTU(1-Z)}}, \qquad K_D = E\,Q_b$$

Water is removed by **pressure** (ultrafiltration): $Q_{uf} = K_{uf}\cdot TMP$.
"""),
    code('''
prof = ks.dialysis.dialyzer_profile(Qb=0.3, Qd=0.5, KoA=1.0, C_in=25)
print(f"clearance = {ks.dialysis.dialyzer_clearance()*1000:.0f} mL/min "
      f"(healthy kidneys: ~56 mL/min for urea, but they work 24 h/day)")
print(f"ultrafiltration at TMP 80 mmHg: {ks.dialysis.ultrafiltration(80):.2f} L/h")
viz2d.plot_dialyzer(prof);
'''),
    code('''
viz3d.show(viz3d.dialyzer_3d(prof))
'''),
    md(r"""
## 2. The patient: 3 sessions of 4 hours per week

Level 0 again, with $K = K_D + K_{residual}$ during dialysis. Dose: $Kt/V$ (target ≥ 1.2) and
urea reduction ratio $URR = 1 - C_{end}/C_{start}$ (target ≥ 65 %).
"""),
    code('''
tr = ks.dialysis.treatment(weeks=2)
viz2d.plot_treatment(tr);
'''),
    md(r"""
## 3. Sorbent cartridge: real ION EXCHANGE

Urease converts urea to ammonium, then zirconium phosphate **exchanges** it for sodium:

$$\text{urea} + H_2O \xrightarrow{urease} 2\,NH_4^+ + CO_3^{2-} \qquad \text{ZrP–Na} + NH_4^+ \rightleftharpoons \text{ZrP–NH}_4 + Na^+$$

Column model (advection + exchange):

$$\varepsilon\frac{\partial c}{\partial t} + u\frac{\partial c}{\partial z} = -\rho_b\frac{\partial q}{\partial t}, \qquad
\frac{\partial q}{\partial t} = k\,(q_{eq}(c) - q), \qquad q_{eq} = \frac{q_{max}K c}{1+Kc}$$
"""),
    code('''
col = ks.dialysis.ion_exchange_column(c_in=15, Q=0.25, t_end_min=480)
t_bt = ks.dialysis.breakthrough_time(col)
print(f"resin capacity {col['capacity_mmol']:.0f} mmol NH4+, breakthrough after {t_bt:.0f} min")
viz2d.plot_breakthrough(col, t_bt);
'''),
    code('''
viz3d.show(viz3d.column_surface(col, which="c"))
'''),
    code('''
def experiment(blood_flow=300, dialysate_flow=500, KoA=1000, session_hours=4.0):
    tr = ks.dialysis.treatment(weeks=2, Qb=blood_flow/1000, Qd=dialysate_flow/1000,
                               KoA=KoA/1000, session_h=session_hours)
    print(f"dialyzer clearance {tr['K_D']*1000:.0f} mL/min, Kt/V = {tr['Kt_V'][0]:.2f}")
    viz2d.plot_treatment(tr)
    plt.show()

sliders(experiment, blood_flow=(100, 500, 25, 300), dialysate_flow=(300, 800, 50, 500),
        KoA=(300, 1500, 50, 1000), session_hours=(2, 8, 0.5, 4))
'''),
    md("""
## Questions
1. Why is the countercurrent arrangement better than flowing blood and dialysate in the same direction?
2. Doubling the blood flow does **not** double the clearance. Why?
3. The sorbent releases Na⁺ for every NH₄⁺ captured. Why could that be a problem for the patient?
4. Compare: what does a real kidney do that this machine cannot? (Think of Levels 2–4.)
""")
])

# =============================================================================
NB6 = "06_video.ipynb"
write(NB6, [
    md(f"""
# Video - from blood to urine
{badge(NB6)}

An animation of the whole process: **blood enters** the kidney, it is **filtered** in the glomerulus,
the tubule **cleans** it (takes back what is useful, secretes toxins) and the **urine** goes to the bladder
and out of the body.

Every number and movement in the video comes from the model:
* particles filter where the net filtration pressure (Level 1) is high,
* along the nephron each particle is reabsorbed with the probability given by the tubule model (Level 2),
* the urine flow from the model fills the bladder.

Rendering takes about 1–2 minutes in Colab (it uses `ffmpeg`, which Colab already has).
"""),
    code(SETUP),
    md("## 1. Look at one frame first (fast)"),
    code('''
from kidneysim import animation
from IPython.display import Image, Video, display

animation.preview(3, 8.0, "frame.png", lang="es")    # scene 3 (tubule) at t = 8 s; lang="en" for English
display(Image("frame.png", width=900))
'''),
    md("## 2. Render the full video (≈ 80 s long)"),
    code('''
animation.make_video("kidney_video.mp4", lang="es")   # "es" = Spanish, "en" = English
Video("kidney_video.mp4", embed=True, width=900)
'''),
    md("""
## 3. Change the model, change the video
The video is a simulation, not a cartoon. Try a person with **no ADH** (diabetes insipidus):
much more water reaches the bladder. Render only the tubule and urine scenes to save time.
"""),
    code('''
animation.make_video("no_adh.mp4", lang="es", nephron_kwargs={"ADH": 0.0}, scenes=[3, 4])
Video("no_adh.mp4", embed=True, width=900)
'''),
    code('''
# Download the video to your computer (only in Colab)
import sys
if "google.colab" in sys.modules:
    from google.colab import files
    files.download("kidney_video.mp4")
'''),
    md("""
## Questions
1. In scene 3, where do most of the blue (water) particles leave the tube? Does it match the bars?
2. Why do the pink particles (drug) *increase* in the proximal tubule?
3. With `ADH = 0`, how many hours does it take to fill the bladder with 300 mL?
4. Make a video for a person with diabetes: `nephron_kwargs={"plasma": {"glucose": 20}}`. What changes?
""")
])
