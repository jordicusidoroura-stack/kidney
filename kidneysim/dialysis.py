"""
LEVEL 5 - The artificial kidney: MEMBRANES + ION EXCHANGE
==========================================================

When the kidneys fail, a machine can do part of their job. Here the original
idea of this project is exactly right: a dialysis machine combines
membrane transport (diffusion + pressure) and, in sorbent systems,
true ion exchange.

5a) Hemodialyzer: diffusion through a membrane (countercurrent)
---------------------------------------------------------------
Blood flows inside ~10 000 hollow fibres; dialysate flows outside in the
OPPOSITE direction. Urea diffuses from blood to dialysate (Fick's law):

    Q_b dC_b/dx = - KoA (C_b - C_d)            blood, flows x: 0 -> 1
    Q_d dC_d/dx = - KoA (C_b - C_d)            dialysate, flows x: 1 -> 0
    boundary:  C_b(0) = C_in,   C_d(1) = 0     (fresh dialysate)

    KoA = mass-transfer coefficient x membrane area   [L/min]

Solving gives the dialyzer clearance (Michaels' formula):

    Z   = Q_b / Q_d          NTU = KoA / Q_b
    E   = (1 - exp(NTU (1 - Z))) / (Z - exp(NTU (1 - Z)))
    K_D = E * Q_b

Ultrafiltration (removing water) is driven by PRESSURE, like the glomerulus:
    Q_uf = K_uf * TMP          TMP = transmembrane pressure [mmHg]

5b) The patient during a session (Level 0 again, with K = K_D)
--------------------------------------------------------------
    d(V C)/dt = G - (K_D + K_residual) C
    dV/dt     = - Q_uf
    Dialysis dose:  Kt/V    and   URR = 1 - C_end / C_start

5c) Sorbent cartridge: real ION EXCHANGE
----------------------------------------
Instead of throwing away 120 L of dialysate, a sorbent system cleans and
re-uses it. Urease turns urea into ammonium:
    urea + H2O  ->  2 NH4+ + CO3(2-)        (enzyme: urease)
Then zirconium phosphate (ZrP) EXCHANGES the ammonium for sodium/hydrogen:
    ZrP-Na  +  NH4+   <->   ZrP-NH4  +  Na+

Column model (1-D advection + exchange, linear driving force):
    eps dc/dt + u dc/dz = - rho_b dq/dt
    dq/dt = k_ex (q_eq(c) - q)
    q_eq  = q_max K c / (1 + K c)          (Langmuir isotherm)

    c   NH4+ in the liquid [mmol/L],  q   NH4+ on the resin [mmol/g]
    For every NH4+ taken up, one Na+ is released into the liquid.
When the resin is full, NH4+ "breaks through" at the outlet: the cartridge
must be replaced.
"""

import numpy as np


# ---- 5a dialyzer ------------------------------------------------------------
def dialyzer_clearance(Qb=0.3, Qd=0.5, KoA=1.0):
    """Michaels' countercurrent clearance (all flows in L/min). Returns K_D (L/min)."""
    Z = Qb / Qd
    ntu = KoA / Qb
    if abs(1 - Z) < 1e-9:
        E = ntu / (1 + ntu)
    else:
        e = np.exp(ntu * (1 - Z))
        E = (1 - e) / (Z - e)
    return E * Qb


def dialyzer_profile(Qb=0.3, Qd=0.5, KoA=1.0, C_in=25.0, n=200):
    """Concentrations along the fibre: blood C_b(x) and dialysate C_d(x)."""
    x = np.linspace(0, 1, n)
    a = KoA * (1 / Qb - 1 / Qd)       # D = C_b - C_d obeys dD/dx = -a D
    # Solution: D(x) = D0 exp(-a x);  C_b(x) = C_in - (KoA/Qb) * integral D
    if abs(a) < 1e-12:
        I = x
        I1 = 1.0
    else:
        I = (1 - np.exp(-a * x)) / a
        I1 = (1 - np.exp(-a)) / a
    # C_d(1) = 0  ->  C_b(1) = D(1)  ->  C_in - (KoA/Qb) D0 I1 = D0 exp(-a)
    D0 = C_in / (np.exp(-a) + KoA / Qb * I1)
    Cb = C_in - KoA / Qb * D0 * I
    Cd = Cb - D0 * np.exp(-a * x)
    return {"x": x, "C_blood": Cb, "C_dialysate": Cd,
            "clearance": Qb * (C_in - Cb[-1]) / C_in}


def ultrafiltration(TMP=80.0, Kuf=8.0):
    """Ultrafiltration rate in L/h. Kuf in mL/h/mmHg, TMP in mmHg."""
    return Kuf * TMP / 1000.0


# ---- 5b treatment schedule ----------------------------------------------------
def treatment(weeks=2, sessions_per_week=3, session_h=4.0, Qb=0.3, Qd=0.5, KoA=1.0,
              G=0.28, V0=40.0, residual_function=0.05, C0=25.0, uf_l_per_session=2.0,
              dt_min=5.0):
    """Urea in a patient with kidney failure on hemodialysis (Mon/Wed/Fri style).

    Between sessions the patient drinks back the water removed (V returns to V0).
    Returns t (days), C (mmol/L), V (L), on (True during dialysis), Kt/V per session.
    """
    K_D = dialyzer_clearance(Qb, Qd, KoA)
    K_res = residual_function * 0.45 * 0.125         # urea clearance of what is left
    start_days = []
    gap = 7.0 / sessions_per_week
    for w in range(weeks):
        for s in range(sessions_per_week):
            start_days.append(w * 7 + s * gap)
    n = int(weeks * 7 * 1440 / dt_min) + 1
    t = np.arange(n) * dt_min
    C = np.empty(n); V = np.empty(n); on = np.zeros(n, bool)
    C[0], V[0] = C0, V0
    q_uf = uf_l_per_session / (session_h * 60)        # L/min during dialysis
    q_drink = uf_l_per_session / (gap * 1440 - session_h * 60)   # refill between sessions
    ktv = []
    for i in range(n - 1):
        day = t[i] / 1440
        dialysing = any(d <= day < d + session_h / 24 for d in start_days)
        on[i] = dialysing
        K = K_res + (K_D if dialysing else 0.0)
        dV = -q_uf if dialysing else q_drink
        # d(VC)/dt = G - K C   ->  dC/dt = (G - K C - C dV/dt) / V
        C[i + 1] = C[i] + dt_min * (G - K * C[i] - C[i] * dV) / V[i]
        V[i + 1] = V[i] + dt_min * dV
    for d in start_days:
        ktv.append(K_D * session_h * 60 / V0)
    return {"t": t / 1440, "C": C, "V": V, "on": on, "K_D": K_D,
            "Kt_V": np.array(ktv), "session_starts": np.array(start_days)}


def urea_reduction_ratio(C_start, C_end):
    return 1 - C_end / C_start


# ---- 5c sorbent ion exchange column ------------------------------------------
def ion_exchange_column(c_in=15.0, Q=0.25, t_end_min=480.0, n_z=60,
                        bed_volume=1.0, eps=0.5, rho_b=800.0,
                        q_max=1.5, K_L=0.2, k_ex=0.2, dt=None, save_every_min=5.0):
    """Ammonium uptake on a zirconium phosphate column.

    c_in    NH4+ entering (mmol/L)      Q   liquid flow (L/min)
    bed_volume (L), eps porosity, rho_b resin per bed volume (g/L)
    q_max (mmol/g), K_L Langmuir constant (L/mmol), k_ex exchange rate (1/min)
    Returns z (0..1), t (min), c[t, z], q[t, z], outlet NH4+ and outlet Na+ released.
    """
    z = np.linspace(0, 1, n_z)
    dz = z[1] - z[0]
    u = Q / (eps * bed_volume)           # 1/min, in units of column length
    if dt is None:
        stiff = (rho_b / eps) * k_ex * q_max * K_L
        dt = 0.4 * min(dz / u, 1.0 / max(stiff + k_ex, 1e-9))
    c = np.zeros(n_z)
    q = np.zeros(n_z)
    n_steps = int(t_end_min / dt)
    save_every = max(int(save_every_min / dt), 1)
    ts, cs, qs, na_out = [0.0], [c.copy()], [q.copy()], [0.0]
    for step in range(1, n_steps + 1):
        q_eq = q_max * K_L * c / (1 + K_L * c)
        dq = k_ex * (q_eq - q)
        c_up = np.concatenate(([c_in], c[:-1]))           # upwind neighbour
        dc = -u * (c - c_up) / dz - (rho_b / eps) * dq
        c = np.maximum(c + dt * dc, 0.0)
        q = q + dt * dq
        if step % save_every == 0:
            ts.append(step * dt)
            cs.append(c.copy())
            qs.append(q.copy())
            # Na+ released = NH4+ removed (1:1 exchange)
            na_out.append(c_in - c[-1])
    cs = np.array(cs)
    return {"z": z, "t": np.array(ts), "c": cs, "q": np.array(qs),
            "outlet": cs[:, -1], "na_released": np.array(na_out),
            "capacity_mmol": q_max * rho_b * bed_volume, "c_in": c_in}


def breakthrough_time(col, fraction=0.1):
    """Time (min) when outlet NH4+ first reaches `fraction` of the inlet value."""
    idx = np.where(col["outlet"] >= fraction * col["c_in"])[0]
    return float(col["t"][idx[0]]) if len(idx) else None
