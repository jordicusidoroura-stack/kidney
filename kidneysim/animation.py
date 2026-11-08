"""
Animated video: from blood entering the kidney to urine leaving the body.

Every number and every particle movement in the video comes from the model:
    * filtration happens where the net filtration pressure (Level 1) is high,
    * particles are reabsorbed along the nephron at the rate the tubule model
      (Level 2) removes each substance,
    * the urine flow and its composition fill the bladder (Level 4/5 numbers).

Usage (needs ffmpeg, already installed in Google Colab):

    from kidneysim.animation import make_video
    make_video("kidney_video.mp4", lang="es")      # or lang="en"

Change the model first and the video changes too, for example:

    make_video("no_adh.mp4", nephron_kwargs={"ADH": 0.0})

Scenes
------
1. Title and particle legend
2. Blood enters the kidney (renal artery -> glomeruli)
3. Filtration in the glomerulus (pressure-driven)
4. Cleaning along the nephron (reabsorption + secretion)
5. Urine: ureters -> bladder -> evacuation
6. Summary
"""

import os
import shutil
import subprocess
import tempfile
import textwrap

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.collections import LineCollection

from . import glomerulus as glom
from . import tubule as tub
from . import geometry as geo
from . import constants as k

W, H, FPS, DPI = 1280, 720, 30, 100
BG = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
ACCENT = "#256abf"

SPECIES = {
    #            colour     size  filtered?
    "rbc":     ("#e34948", 70, False),
    "protein": ("#6250d6", 34, False),
    "water":   ("#2a78d6", 22, True),
    "Na":      ("#eda100", 26, True),
    "glucose": ("#1baf7a", 26, True),
    "urea":    ("#eb6834", 26, True),
    "drug":    ("#e87ba4", 30, True),
}

TEXT = {
    "es": {
        "title": "Cómo el riñón fabrica la orina",
        "subtitle": "Simulación kidneysim: presión (física) + transporte de membrana (química)",
        "legend": {"rbc": "glóbulo rojo", "protein": "proteína (albúmina)", "water": "agua",
                   "Na": "sodio (sal)", "glucose": "glucosa", "urea": "urea (viene del hígado)",
                   "drug": "fármaco / toxina"},
        "note_model": "Cada número y cada movimiento sale del modelo.",
        "s2_title": "1 · La sangre entra en el riñón",
        "s2_lines": ["Arteria renal: {blood:.1f} L de sangre por minuto",
                     "≈ 20 % de todo lo que bombea el corazón",
                     "Se reparte entre ≈ 1 millón de nefronas por riñón",
                     "Cada nefrona empieza en un glomérulo"],
        "artery": "arteria renal", "vein": "vena renal", "ureter": "uréter",
        "cortex": "corteza", "medulla": "médula", "pelvis": "pelvis",
        "s3_title": "2 · Filtración en el glomérulo",
        "s3_lines": ["La PRESIÓN de la sangre empuja\nel plasma a través del filtro",
                     "P capilar {pgc:.0f} mmHg − P Bowman {pbs:.0f}\n− π proteínas {pi0:.0f}→{pi1:.0f} mmHg",
                     "Pasan: agua, sal, glucosa, urea",
                     "No pasan: células ni proteínas",
                     "Filtrado: {gfr:.0f} mL/min  =  {gfr_day:.0f} L/día"],
        "afferent": "arteriola aferente", "efferent": "arteriola eferente (más estrecha)",
        "bowman": "cápsula de Bowman", "to_tubule": "al túbulo",
        "push": "P empuja", "pull": "π tira",
        "s4_title": "3 · Limpieza en el túbulo",
        "s4_steps": [
            ("Entran {gfr:.0f} mL/min de filtrado", "Todo lo pequeño ha pasado el filtro; ahora hay que recuperarlo."),
            ("Túbulo proximal", "Recupera {pt_na:.0f} % de la sal y el agua, TODA la glucosa (SGLT)\ny secreta fármacos con un intercambiador (OAT1)."),
            ("Asa de Henle", "La rama descendente pierde agua; la ascendente bombea sal\nsin dejar salir agua: crea el gradiente salino de la médula."),
            ("Túbulo distal", "Ajuste fino de la sal (aldosterona)."),
            ("Túbulo colector", "La hormona ADH abre canales de agua (acuaporinas):\nla orina se concentra al cruzar la médula salada."),
        ],
        "remaining": "Queda en el túbulo (% de lo filtrado)",
        "back_to_blood": "vuelve a la sangre",
        "s5_title": "4 · Orina y evacuación",
        "s5_lines": ["Orina: {urine:.2f} mL/min  ≈  {urine_day:.1f} L/día",
                     "Osmolaridad: {uosm:.0f} mOsm/L (sangre: 290)",
                     "Urea en orina: {uurea:.0f} mmol/L ({ratio:.0f}× la de la sangre)",
                     "De {gfr_day:.0f} L filtrados, se recupera el {reabs:.1f} % del agua",
                     "Los uréteres llevan la orina a la vejiga",
                     "Con ≈ 300 mL aparecen las ganas de orinar"],
        "bladder": "vejiga", "urethra": "uretra", "micturition": "micción",
        "accelerated": "tiempo acelerado: {hours:.1f} h en 9 s",
        "kidney": "riñón",
        "s6_title": "Resumen",
        "s6_boxes": ["Sangre\n{blood_ml:.0f} mL/min", "Filtrado\n{gfr:.0f} mL/min", "Orina\n{urine:.1f} mL/min"],
        "s6_arrows": ["filtración\n(presión)", "reabsorción y secreción\n(transporte de membrana)"],
        "s6_lines": ["El hígado fabrica la urea; el riñón la elimina.",
                     "Prueba a cambiar el modelo y vuelve a generar el vídeo."],
    },
    "en": {
        "title": "How the kidney makes urine",
        "subtitle": "A kidneysim simulation: pressure (physics) + membrane transport (chemistry)",
        "legend": {"rbc": "red blood cell", "protein": "protein (albumin)", "water": "water",
                   "Na": "sodium (salt)", "glucose": "glucose", "urea": "urea (made by the liver)",
                   "drug": "drug / toxin"},
        "note_model": "Every number and every movement comes from the model.",
        "s2_title": "1 · Blood enters the kidney",
        "s2_lines": ["Renal artery: {blood:.1f} L of blood per minute",
                     "≈ 20 % of everything the heart pumps",
                     "Shared among ≈ 1 million nephrons per kidney",
                     "Each nephron starts at a glomerulus"],
        "artery": "renal artery", "vein": "renal vein", "ureter": "ureter",
        "cortex": "cortex", "medulla": "medulla", "pelvis": "pelvis",
        "s3_title": "2 · Filtration in the glomerulus",
        "s3_lines": ["Blood PRESSURE pushes plasma\nthrough the filter",
                     "P capillary {pgc:.0f} mmHg − P Bowman {pbs:.0f}\n− π proteins {pi0:.0f}→{pi1:.0f} mmHg",
                     "Pass: water, salt, glucose, urea",
                     "Stay: cells and proteins",
                     "Filtrate: {gfr:.0f} mL/min  =  {gfr_day:.0f} L/day"],
        "afferent": "afferent arteriole", "efferent": "efferent arteriole (narrower)",
        "bowman": "Bowman's capsule", "to_tubule": "to the tubule",
        "push": "P pushes", "pull": "π pulls",
        "s4_title": "3 · Cleaning in the tubule",
        "s4_steps": [
            ("{gfr:.0f} mL/min of filtrate enter", "Everything small passed the filter; now it must be recovered."),
            ("Proximal tubule", "Takes back {pt_na:.0f} % of salt and water, ALL the glucose (SGLT)\nand secretes drugs through an exchanger (OAT1)."),
            ("Loop of Henle", "The descending limb loses water; the ascending limb pumps salt\nbut keeps water in: it builds the salty gradient of the medulla."),
            ("Distal tubule", "Fine-tuning of salt (aldosterone)."),
            ("Collecting duct", "The hormone ADH opens water channels (aquaporins):\nurine gets concentrated crossing the salty medulla."),
        ],
        "remaining": "Left in the tubule (% of filtered)",
        "back_to_blood": "back to the blood",
        "s5_title": "4 · Urine and emptying",
        "s5_lines": ["Urine: {urine:.2f} mL/min  ≈  {urine_day:.1f} L/day",
                     "Osmolarity: {uosm:.0f} mOsm/L (blood: 290)",
                     "Urea in urine: {uurea:.0f} mmol/L ({ratio:.0f}× blood)",
                     "Of {gfr_day:.0f} L filtered, {reabs:.1f} % of the water is recovered",
                     "The ureters carry urine to the bladder",
                     "At ≈ 300 mL we feel the urge to urinate"],
        "bladder": "bladder", "urethra": "urethra", "micturition": "urination",
        "accelerated": "time sped up: {hours:.1f} h in 9 s",
        "kidney": "kidney",
        "s6_title": "Summary",
        "s6_boxes": ["Blood\n{blood_ml:.0f} mL/min", "Filtrate\n{gfr:.0f} mL/min", "Urine\n{urine:.1f} mL/min"],
        "s6_arrows": ["filtration\n(pressure)", "reabsorption & secretion\n(membrane transport)"],
        "s6_lines": ["The liver makes urea; the kidney removes it.",
                     "Change the model and render the video again."],
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class Path2D:
    """A polyline that can be sampled by fraction of its length (u in 0..1)."""

    def __init__(self, x, y):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        d = np.hypot(np.diff(self.x), np.diff(self.y))
        self.s = np.concatenate(([0], np.cumsum(d)))
        self.L = self.s[-1]

    def point(self, u):
        s = np.clip(np.asarray(u), 0, 1) * self.L
        return np.interp(s, self.s, self.x), np.interp(s, self.s, self.y)

    def normal(self, u, du=1e-3):
        x0, y0 = self.point(np.clip(u - du, 0, 1))
        x1, y1 = self.point(np.clip(u + du, 0, 1))
        tx, ty = x1 - x0, y1 - y0
        n = np.hypot(tx, ty) + 1e-12
        return -ty / n, tx / n


def _rgba(hex_colors, alpha):
    from matplotlib.colors import to_rgba_array
    c = to_rgba_array(hex_colors)
    c[:, 3] = np.clip(alpha, 0, 1)
    return c


def _ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _fade_alpha(t, duration, fade=0.5):
    """Alpha of the cover rectangle: 1 = hidden, 0 = visible."""
    return float(max(1 - t / fade, 0, 1 - (duration - t) / fade))


class Scene:
    """Base class: a figure with a drawing area (left) and a text panel (right)."""

    duration = 5.0

    def __init__(self, data, lang):
        self.d = data
        self.T = TEXT[lang]
        self.fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
        self.cover = Rectangle((0, 0), 1, 1, transform=self.fig.transFigure,
                               facecolor=BG, edgecolor="none", zorder=1000)
        self.fig.patches.append(self.cover)
        self.rng = np.random.default_rng(7)
        self.setup()

    def setup(self):
        pass

    def update(self, t, dt):
        pass

    def frame(self, t, dt):
        self.update(t, dt)
        self.cover.set_alpha(_fade_alpha(t, self.duration))

    # common layout helpers
    def left_axes(self, xlim=(0, 10), ylim=(0, 6.2)):
        ax = self.fig.add_axes([0.01, 0.03, 0.60, 0.94])
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal"); ax.axis("off")
        return ax

    def text_axes(self):
        ax = self.fig.add_axes([0.62, 0.03, 0.37, 0.94])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        return ax

    def title(self, ax, text):
        ax.text(0.0, 0.95, text, fontsize=21, fontweight="bold", color=INK, va="top", wrap=True)

    def lines(self, ax, lines, y0=0.82, dy=0.075, size=13.5):
        arts = []
        for i, line in enumerate(lines):
            arts.append(ax.text(0.0, y0 - i * dy, line, fontsize=size, color=INK2, va="top", alpha=0))
        return arts

    def reveal(self, arts, t, start, step):
        for i, a in enumerate(arts):
            a.set_alpha(_ease((t - start - i * step) / 0.5))

    def legend(self, ax, species, y=0.20, dy=0.045):
        for i, sp in enumerate(species):
            col, size, _ = SPECIES[sp]
            ax.scatter([0.02], [y - i * dy], s=size * 1.6, color=col, edgecolors="white", linewidths=1)
            ax.text(0.06, y - i * dy, self.T["legend"][sp], va="center", fontsize=11, color=INK2)


def _particles_artist(ax, zorder=5):
    art = ax.scatter(np.empty(0), np.empty(0), s=np.empty(0), edgecolors="white",
                     linewidths=0.8, zorder=zorder)
    art.set_array(None)     # use our own colours, not a colormap
    return art


def _set_particles(art, x, y, species, alpha):
    if len(x) == 0:
        art.set_offsets(np.empty((0, 2)))
        art.set_sizes([])
        return
    art.set_offsets(np.column_stack([x, y]))
    art.set_sizes([SPECIES[s][1] * 1.6 for s in species])
    art.set_facecolors(_rgba([SPECIES[s][0] for s in species], alpha))
    art.set_edgecolors(_rgba(["#ffffff"] * len(x), alpha))


def _bean(cx, cy, a, b, hilum_side=1, n=200, notch=0.33):
    """Kidney-bean outline. hilum_side=+1 hilum on the right, -1 on the left."""
    t = np.linspace(0, 2 * np.pi, n)
    ang = t if hilum_side > 0 else (t - np.pi)
    ang = (ang + np.pi) % (2 * np.pi) - np.pi
    r = 1 - notch * np.exp(-(ang / 0.55) ** 2)
    return cx + a * r * np.cos(t), cy + b * r * np.sin(t)


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------
class TitleScene(Scene):
    duration = 4.5

    def setup(self):
        ax = self.fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        self.ax = ax
        self.t1 = ax.text(0.5, 0.66, self.T["title"], ha="center", fontsize=38, fontweight="bold", color=INK)
        self.t2 = ax.text(0.5, 0.57, self.T["subtitle"], ha="center", fontsize=16, color=INK2)
        self.t3 = ax.text(0.5, 0.12, self.T["note_model"], ha="center", fontsize=12, color=INK2, style="italic")
        sp = list(SPECIES)
        self.leg = []
        for i, s in enumerate(sp):
            x = 0.12 + i * 0.125
            c = ax.scatter([x], [0.36], s=SPECIES[s][1] * 3, color=SPECIES[s][0], edgecolors="white")
            tx = ax.text(x, 0.30, self.T["legend"][s], ha="center", fontsize=10.5, color=INK2, wrap=True)
            self.leg.append((c, tx))

    def update(self, t, dt):
        for i, (c, tx) in enumerate(self.leg):
            a = _ease((t - 1.0 - 0.15 * i) / 0.4)
            c.set_alpha(a); tx.set_alpha(a)
            c.set_offsets([[0.12 + i * 0.125, 0.36 + 0.008 * np.sin(3 * t + i)]])
        self.t3.set_alpha(_ease((t - 2.5) / 0.5))


class BloodInScene(Scene):
    duration = 9.0

    def setup(self):
        ax = self.left_axes()
        self.ax = ax
        cx, cy = 3.6, 3.1
        bx, by = _bean(cx, cy, 2.3, 2.9, hilum_side=1)
        ax.add_patch(Polygon(np.column_stack([bx, by]), closed=True, fc="#f6d9d3", ec="#c9796c", lw=2))
        mx, my = _bean(cx - 0.1, cy, 1.6, 2.1, hilum_side=1, notch=0.42)
        ax.add_patch(Polygon(np.column_stack([mx, my]), closed=True, fc="#edbdb2", ec="none"))
        ax.text(cx - 2.05, cy + 2.4, self.T["cortex"], fontsize=11, color=INK2)
        ax.text(cx - 0.9, cy - 0.2, self.T["medulla"], fontsize=11, color=INK2)
        hil_x = cx + 2.3 * (1 - 0.33) + 0.05
        # renal artery, vein, ureter
        self.artery = Path2D([10, 7.5, hil_x], [3.6, 3.5, 3.35])
        ax.plot(self.artery.x, self.artery.y, color="#e34948", lw=10, solid_capstyle="round", alpha=0.85)
        ax.plot([hil_x, 7.5, 10], [2.95, 2.85, 2.8], color="#6d8fc7", lw=10, solid_capstyle="round", alpha=0.7)
        ax.plot([hil_x - 0.1, 6.0, 6.3], [2.6, 1.2, 0.1], color="#eda100", lw=6, solid_capstyle="round", alpha=0.7)
        ax.text(8.0, 3.85, self.T["artery"], fontsize=11, color="#b8322f")
        ax.text(8.0, 2.35, self.T["vein"], fontsize=11, color="#3b5f9a")
        ax.text(6.4, 0.6, self.T["ureter"], fontsize=11, color="#9a6a00")
        # branches to glomeruli in the cortex
        self.branches = []
        angs = np.linspace(0.55 * np.pi, 1.45 * np.pi, 9)
        for a in angs:
            r = 0.86
            ex, ey = cx + 2.3 * r * np.cos(a), cy + 2.9 * r * np.sin(a)
            midx, midy = cx + 0.6 * np.cos(a), cy + 0.9 * np.sin(a)
            p = Path2D([hil_x, cx + 0.9, midx, ex], [3.35, 3.3, midy, ey])
            ax.plot(p.x, p.y, color="#e34948", lw=2.2, alpha=0.6)
            ax.add_patch(Circle((ex, ey), 0.13, fc="#e34948", ec="white", lw=1.2, alpha=0.9, zorder=4))
            self.branches.append(p)
        self.parts = _particles_artist(ax)
        self.p = []   # [phase, u, branch, species]
        tx = self.text_axes()
        self.title(tx, self.T["s2_title"])
        self.arts = self.lines(tx, [l.format(**self.d["fmt"]) for l in self.T["s2_lines"]], y0=0.80, dy=0.085)
        self.legend(tx, ["rbc", "protein", "water", "Na", "glucose", "urea"], y=0.36)

    def update(self, t, dt):
        rate = 22
        for _ in range(self.rng.poisson(rate * dt)):
            sp = self.rng.choice(["rbc", "rbc", "protein", "water", "water", "Na", "glucose", "urea"])
            self.p.append([0, 0.0, int(self.rng.integers(len(self.branches))), sp])
        xs, ys, ss, al = [], [], [], []
        keep = []
        for q in self.p:
            q[1] += dt * (0.42 if q[0] == 0 else 0.5)
            if q[0] == 0 and q[1] >= 1:
                q[0], q[1] = 1, 0.0
            if q[0] == 1 and q[1] >= 1:
                continue
            path = self.artery if q[0] == 0 else self.branches[q[2]]
            x, y = path.point(q[1])
            jitter = 0.06 if q[0] == 0 else 0.02
            xs.append(x + jitter * np.sin(7 * q[1] + hash(q[3]) % 5))
            ys.append(y + jitter * np.cos(5 * q[1] + q[2]))
            ss.append(q[3]); al.append(1.0 if q[0] == 0 or q[1] < 0.85 else (1 - q[1]) / 0.15)
            keep.append(q)
        self.p = keep
        _set_particles(self.parts, xs, ys, ss, np.array(al))
        self.reveal(self.arts, t, 0.6, 1.0)


class FiltrationScene(Scene):
    duration = 15.0

    def setup(self):
        ax = self.left_axes()
        self.ax = ax
        f = self.d["filt"]
        C = (5.2, 3.3)
        self.C = C
        R_loop, R_cap = 1.05, 1.75
        # Bowman's capsule (gap at the vascular pole and at the tubular pole)
        th = np.radians(np.linspace(-150, 155, 300))
        th = th[(np.degrees(th) < -55) | (np.degrees(th) > -30)]
        cap = np.column_stack([C[0] + R_cap * np.cos(th), C[1] + R_cap * np.sin(th)])
        split = np.where(np.diff(np.degrees(th)) > 5)[0]
        for seg in np.split(cap, split + 1):
            ax.plot(seg[:, 0], seg[:, 1], color="#6250d6", lw=5, alpha=0.6, solid_capstyle="round")
        ax.add_patch(Circle(C, R_cap - 0.05, fc="#efecfb", ec="none", zorder=0))
        ax.text(C[0], C[1] - R_cap - 0.4, self.T["bowman"], fontsize=11, color="#4a3bb0", ha="center")
        # capillary loop
        ang = np.radians(np.linspace(165, -165, 300))
        loop_x = C[0] + R_loop * np.cos(ang) * (1 + 0.08 * np.sin(6 * ang))
        loop_y = C[1] + R_loop * np.sin(ang) * (1 + 0.08 * np.sin(6 * ang))
        aff = Path2D([0.0, 2.6, loop_x[0]], [C[1] + 0.32, C[1] + 0.32, loop_y[0]])
        loop = Path2D(loop_x, loop_y)
        eff = Path2D([loop_x[-1], 2.6, 0.0], [loop_y[-1], C[1] - 0.32, C[1] - 0.32])
        self.vessels = [aff, loop, eff]
        ax.plot(aff.x, aff.y, color="#e34948", lw=16, alpha=0.30, solid_capstyle="round")
        ax.plot(eff.x, eff.y, color="#e34948", lw=10, alpha=0.30, solid_capstyle="round")
        # loop coloured by net filtration pressure
        nfp = np.interp(np.linspace(0, 1, len(loop_x)), f["x"], f["net_pressure"])
        pts = np.column_stack([loop_x, loop_y]).reshape(-1, 1, 2)
        lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1),
                            cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                                "p", ["#f3d2cf", "#e34948"]), linewidths=14, alpha=0.6)
        lc.set_array(nfp[:-1]); ax.add_collection(lc)
        ax.text(0.1, C[1] + 0.62, self.T["afferent"], fontsize=10.5, color="#b8322f")
        ax.text(0.1, C[1] - 0.85, self.T["efferent"], fontsize=10.5, color="#b8322f")
        # tubule exit
        a0 = np.radians(-42)
        ex, ey = C[0] + R_cap * np.cos(a0), C[1] + R_cap * np.sin(a0)
        self.exit = (ex, ey)
        self.tubule = Path2D([ex, ex + 1.2, 9.9], [ey, ey - 0.5, ey - 0.6])
        for off in (-0.22, 0.22):
            ax.plot(self.tubule.x, self.tubule.y + off, color="#6250d6", lw=3, alpha=0.6)
        ax.text(8.2, ey - 1.2, self.T["to_tubule"] + " →", fontsize=11, color="#4a3bb0")
        # pressure arrows
        top = (C[0] + 0.15, C[1] + R_loop + 0.05)
        self.arrows = [
            FancyArrowPatch(top, (top[0], top[1] + 0.55), arrowstyle="-|>", mutation_scale=18, color="#e34948", lw=2.5),
            FancyArrowPatch((top[0] + 0.55, top[1] + 0.6), (top[0] + 0.55, top[1] + 0.08), arrowstyle="-|>",
                            mutation_scale=18, color="#6250d6", lw=2.5),
        ]
        for a in self.arrows:
            a.set_alpha(0); ax.add_patch(a)
        self.arrow_txt = [ax.text(top[0] - 0.25, top[1] + 0.7, self.T["push"], fontsize=10, color="#b8322f", ha="right", alpha=0),
                          ax.text(top[0] + 0.75, top[1] + 0.7, self.T["pull"], fontsize=10, color="#4a3bb0", alpha=0)]
        # sampling of the filtration point follows the net filtration pressure profile
        w = np.maximum(f["net_pressure"], 0)
        self.cdf = np.cumsum(w) / w.sum()
        self.fx = f["x"]
        self.ff = f["filtration_fraction"]
        self.parts = _particles_artist(ax, zorder=6)
        self.p = []
        self.n_filtered = 0
        tx = self.text_axes()
        self.title(tx, self.T["s3_title"])
        self.arts = self.lines(tx, [l.format(**self.d["fmt"]) for l in self.T["s3_lines"]], y0=0.82, dy=0.095, size=12.5)
        for i, a in enumerate(self.arts):      # two-line items need more room
            a.set_y(0.82 - [0, 0.11, 0.22, 0.29, 0.36][i])
        self.legend(tx, ["rbc", "protein", "water", "Na", "glucose", "urea"], y=0.33)

    def update(self, t, dt):
        rng = self.rng
        for _ in range(rng.poisson(16 * dt)):
            sp = rng.choice(["rbc", "rbc", "protein", "water", "water", "water", "Na", "Na", "glucose", "urea"])
            filt_at = None
            if SPECIES[sp][2] and rng.random() < self.ff * 1.6:      # x1.6 so the effect is visible
                filt_at = float(np.interp(rng.random(), self.cdf, self.fx))
            # state: 0 aff, 1 loop, 2 eff, 3 crossing, 4 drifting to exit, 5 in tubule
            self.p.append({"st": 0, "u": 0.0, "sp": sp, "f": filt_at, "j": rng.normal(0, 0.05, 2)})
        speed = [0.55, 0.30, 0.55, 1.6, 0.8, 0.35]
        xs, ys, ss, al = [], [], [], []
        keep = []
        for q in self.p:
            q["u"] += dt * speed[q["st"]]
            st = q["st"]
            if st == 1 and q["f"] is not None and q["u"] >= q["f"]:
                x, y = self.vessels[1].point(q["f"])
                vx, vy = x - self.C[0], y - self.C[1]
                n = np.hypot(vx, vy)
                q["from"] = (x, y)
                q["to"] = (self.C[0] + vx / n * 1.45, self.C[1] + vy / n * 1.45)
                q["st"], q["u"] = 3, 0.0
                self.n_filtered += 1
            elif st in (0, 1) and q["u"] >= 1:
                q["st"], q["u"] = st + 1, 0.0
            elif st == 2 and q["u"] >= 1:
                continue
            elif st == 3 and q["u"] >= 1:
                q["from"] = q["to"]; q["to"] = self.exit; q["st"], q["u"] = 4, 0.0
            elif st == 4 and q["u"] >= 1:
                q["st"], q["u"] = 5, 0.0
            elif st == 5 and q["u"] >= 1:
                continue
            st, u = q["st"], q["u"]
            if st <= 2:
                x, y = self.vessels[st].point(u)
                x, y = x + q["j"][0] * (0.5 if st == 1 else 1), y + q["j"][1] * (0.5 if st == 1 else 1)
            elif st in (3, 4):
                e = _ease(u) if st == 3 else u
                x = q["from"][0] + (q["to"][0] - q["from"][0]) * e
                y = q["from"][1] + (q["to"][1] - q["from"][1]) * e
            else:
                x, y = self.tubule.point(u)
                y = y + q["j"][1] * 1.5
            xs.append(x); ys.append(y); ss.append(q["sp"])
            al.append(1.0)
            keep.append(q)
        self.p = keep
        _set_particles(self.parts, xs, ys, ss, np.array(al))
        self.reveal(self.arts, t, 0.8, 2.0)
        a = _ease((t - 3.0) / 0.6)
        for art in self.arrows + self.arrow_txt:
            art.set_alpha(a)


class TubuleScene(Scene):
    duration = 30.0
    STEP_TIMES = [0.0, 5.0, 11.0, 17.5, 22.0]

    def setup(self):
        r = self.d["nephron"]
        p = r["profile"]
        self.r = r
        x, y, z = geo.nephron_xyz(p)
        self.pos = p["position"]
        self.path_x, self.path_z = x, z
        ax = self.left_axes(xlim=(-0.32, 1.25), ylim=(-1.12, 0.12))
        self.ax = ax
        # tissue background: cortex light, medulla getting saltier (darker)
        depth = np.linspace(0, 1, 120)
        bg = tub.interstitial_osm(depth, osm_max=p["osm_interstitium"].max())
        ax.imshow(bg[:, None], extent=[-0.32, 1.25, -1.0, 0.0], aspect="auto", origin="upper",
                  cmap=matplotlib.colors.LinearSegmentedColormap.from_list("t", ["#faf6ee", "#e8d7b8"]),
                  zorder=0, vmin=290, vmax=1250)
        ax.axhline(-geo.CORTEX_DEPTH, color=INK2, lw=0.8, ls=":", zorder=1)
        ax.text(-0.30, -0.06, self.T["cortex"], fontsize=11, color=INK2)
        ax.text(-0.30, -0.36, self.T["medulla"] + "  (" + ("más salada ↓" if self.T is TEXT["es"] else "saltier ↓") + ")",
                fontsize=11, color=INK2)
        # capillary next to the tube (where reabsorbed particles go)
        nx, nz = np.gradient(x), np.gradient(z)
        n = np.hypot(nx, nz) + 1e-12
        self.nx, self.nz = -nz / n, nx / n
        # tube coloured by osmolarity
        pts = np.column_stack([x, z]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "o", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
        self.tube = LineCollection(segs, cmap=cmap, linewidths=13, alpha=0.55, capstyle="round", zorder=2)
        self.tube.set_array(p["osm"][:-1]); self.tube.set_clim(0, 1200)
        ax.add_collection(self.tube)
        self.glow = LineCollection([], colors="#eda100", linewidths=22, alpha=0.25, capstyle="round", zorder=1.5)
        ax.add_collection(self.glow)
        self.segs = segs
        ax.add_patch(Circle((0.0, -0.05), 0.075, fc="#e34948", ec="white", lw=1.5, alpha=0.8, zorder=3))
        cbax = self.fig.add_axes([0.035, 0.08, 0.012, 0.25])
        cb = self.fig.colorbar(matplotlib.cm.ScalarMappable(norm=self.tube.norm, cmap=cmap), cax=cbax)
        cb.set_label("mOsm/L", fontsize=9, color=INK2); cb.ax.tick_params(labelsize=8, colors=INK2)
        ax.annotate("", xy=(1.0, -1.11), xytext=(1.0, -1.0),
                    arrowprops=dict(arrowstyle="-|>", color="#eda100", lw=2.5))
        ax.text(1.04, -1.09, self.T["pelvis"], fontsize=10, color="#9a6a00")
        self.parts = _particles_artist(ax, zorder=6)
        self.p = []
        # amount profiles used for survival
        self.N = {"water": p["F"], "Na": p["N_Na"], "glucose": p["N_glucose"], "urea": p["N_urea"],
                  "drug": p["N_drug"]}
        self.seg_of = p["segment"]
        # text panel
        tx = self.text_axes()
        self.tx = tx
        self.title(tx, self.T["s4_title"])
        self.step_head = tx.text(0, 0.80, "", fontsize=16, fontweight="bold", color=ACCENT, va="top")
        self.step_body = tx.text(0, 0.73, "", fontsize=12, color=INK2, va="top", linespacing=1.4)
        tx.text(0, 0.50, self.T["remaining"], fontsize=12, color=INK, fontweight="bold")
        self.bar_keys = ["water", "Na", "glucose", "urea"]
        self.bars, self.bar_txt = [], []
        for i, key in enumerate(self.bar_keys):
            yb = 0.43 - i * 0.075
            tx.add_patch(Rectangle((0.30, yb - 0.022), 0.56, 0.044, fc="#eeede9", ec="none"))
            b = Rectangle((0.30, yb - 0.022), 0.56, 0.044, fc=SPECIES[key][0], ec="none")
            tx.add_patch(b); self.bars.append(b)
            tx.text(0.0, yb, self.T["legend"][key].split(" (")[0], va="center", fontsize=11, color=INK2)
            self.bar_txt.append(tx.text(0.88, yb, "", va="center", fontsize=11, color=INK))
        self.legend(tx, ["water", "Na", "glucose", "urea", "drug"], y=0.16, dy=0.032)
        self.bar_vals = np.ones(4) * 100
        e = r["segment_end"]
        self.fmt = dict(self.d["fmt"], pt_na=100 * (1 - e["PT"]["Na_left"]))
        # target remaining (%) at the END of the segment each step explains
        prof = r["profile"]
        last = len(prof["F"]) - 1
        targets = []
        for seg in [None, 0, 2, 3, 4]:
            if seg is None:
                targets.append(np.ones(4) * 100)
                continue
            idx = min(np.where(prof["segment"] == seg)[0][-1] + 1, last)
            if seg == 4:
                idx = last
            vals = [prof["F"][idx] / r["gfr"]] + [prof["N_" + kk][idx] / r["filtered"][kk]
                                                  for kk in ["Na", "glucose", "urea"]]
            targets.append(100 * np.array(vals))
        self.targets = targets
        self.highlight = [None, 0, (1, 2), 3, 4]

    def _xy(self, pos):
        return np.interp(pos, self.pos, self.path_x), np.interp(pos, self.pos, self.path_z)

    def _normal(self, pos):
        return np.interp(pos, self.pos, self.nx), np.interp(pos, self.pos, self.nz)

    def update(self, t, dt):
        rng = self.rng
        # current explanation step
        step = max(i for i, ts in enumerate(self.STEP_TIMES) if t >= ts)
        head, body = self.T["s4_steps"][step]
        self.step_head.set_text(head.format(**self.fmt))
        body = body.format(**self.fmt).replace("\n", " ")
        self.step_body.set_text(textwrap.fill(body, 50))
        a = _ease((t - self.STEP_TIMES[step]) / 0.5)
        self.step_head.set_alpha(a); self.step_body.set_alpha(a)
        hl = self.highlight[step]
        if hl is None:
            self.glow.set_segments([])
        else:
            segs_idx = (hl,) if isinstance(hl, int) else hl
            m = np.isin(self.seg_of[:-1], segs_idx)
            self.glow.set_segments(self.segs[m])
        self.bar_vals += (self.targets[step] - self.bar_vals) * min(1.0, 3 * dt)
        for b, txt, v in zip(self.bars, self.bar_txt, self.bar_vals):
            b.set_width(0.56 * max(v, 0) / 100)
            txt.set_text(f"{v:.1f} %" if v < 10 else f"{v:.0f} %")
        # spawn filtrate at the start; drug also secreted along the proximal tubule
        speed = 0.42          # nephron units per second (5 units ~ 12 s)
        rates = {"water": 12, "Na": 7, "glucose": 3, "urea": 4, "drug": 0.3}
        for sp, rr in rates.items():
            for _ in range(rng.poisson(rr * dt)):
                self.p.append({"sp": sp, "pos": 0.0, "st": 0, "u": 0.0, "j": rng.normal(0, 0.012)})
        Nd = self.N["drug"]
        secreted_ratio = (np.interp(1.0, self.pos, Nd) - Nd[0]) / max(Nd[0], 1e-12)
        for _ in range(rng.poisson(rates["drug"] * secreted_ratio * dt)):
            ps = rng.random()
            self.p.append({"sp": "drug", "pos": ps, "st": 2, "u": 0.0, "j": rng.normal(0, 0.012)})
        xs, ys, ss, al = [], [], [], []
        keep = []
        for q in self.p:
            if q["st"] == 0:                         # flowing in the tubule
                new = q["pos"] + speed * dt
                if new >= self.pos[-1]:
                    continue                          # left as urine
                N = self.N[q["sp"]]
                n0, n1 = np.interp(q["pos"], self.pos, N), np.interp(new, self.pos, N)
                q["pos"] = new
                if n1 < n0 and rng.random() < 1 - n1 / max(n0, 1e-15):
                    q["st"], q["u"] = 1, 0.0           # reabsorbed
                x, y = self._xy(q["pos"])
                nx, ny = self._normal(q["pos"])
                x, y = x + nx * q["j"], y + ny * q["j"]
                alpha = 1.0
            elif q["st"] == 1:                       # leaving towards the blood
                q["u"] += dt / 0.8
                if q["u"] >= 1:
                    continue
                x0, y0 = self._xy(q["pos"])
                nx, ny = self._normal(q["pos"])
                d = 0.03 + 0.10 * _ease(q["u"])
                x, y = x0 + nx * d, y0 + ny * d
                alpha = 1 - q["u"]
            else:                                    # secreted drug entering the tube
                q["u"] += dt / 0.8
                x0, y0 = self._xy(q["pos"])
                nx, ny = self._normal(q["pos"])
                d = 0.13 * (1 - _ease(q["u"]))
                x, y = x0 - nx * d, y0 - ny * d
                alpha = min(1.0, q["u"] * 2)
                if q["u"] >= 1:
                    q["st"] = 0
            xs.append(x); ys.append(y); ss.append(q["sp"]); al.append(alpha)
            keep.append(q)
        self.p = keep
        _set_particles(self.parts, xs, ys, ss, np.array(al))


class UrineScene(Scene):
    duration = 15.0

    def setup(self):
        ax = self.left_axes()
        self.ax = ax
        for cx, side in ((2.4, 1), (7.6, -1)):
            bx, by = _bean(cx, 4.7, 1.0, 1.3, hilum_side=side)
            ax.add_patch(Polygon(np.column_stack([bx, by]), closed=True, fc="#f6d9d3", ec="#c9796c", lw=2))
        ax.text(2.4, 6.15, self.T["kidney"], ha="center", fontsize=11, color=INK2)
        ax.text(7.6, 6.15, self.T["kidney"], ha="center", fontsize=11, color=INK2)
        self.ureters = [Path2D([3.15, 3.6, 4.3, 4.55], [4.6, 3.6, 2.6, 2.05]),
                        Path2D([6.85, 6.4, 5.7, 5.45], [4.6, 3.6, 2.6, 2.05])]
        for u in self.ureters:
            ax.plot(u.x, u.y, color="#eda100", lw=7, alpha=0.45, solid_capstyle="round")
        ax.text(2.7, 3.0, self.T["ureter"], fontsize=11, color="#9a6a00")
        self.bc, self.bw, self.bh = (5.0, 1.35), 2.6, 1.6
        self.bladder = Ellipse(self.bc, self.bw, self.bh, fc="none", ec="#c9796c", lw=3, zorder=4)
        ax.add_patch(self.bladder)
        self.fill = Rectangle((self.bc[0] - self.bw / 2, self.bc[1] - self.bh / 2), self.bw, 0.0,
                              fc="#f2c94c", ec="none", alpha=0.8, zorder=3)
        ax.add_patch(self.fill)
        self.fill.set_clip_path(Ellipse(self.bc, self.bw - 0.06, self.bh - 0.06, transform=ax.transData))
        ax.text(6.45, 1.95, self.T["bladder"], fontsize=11, color=INK2)
        ax.plot([5.0, 5.0], [0.55, 0.0], color="#eda100", lw=6, alpha=0.45)
        ax.text(5.2, 0.15, self.T["urethra"], fontsize=11, color=INK2)
        self.vol_txt = ax.text(5.0, 1.35, "", ha="center", va="center", fontsize=14, fontweight="bold",
                               color=INK, zorder=5)
        self.mic = ax.text(3.0, 0.25, "", fontsize=13, fontweight="bold", color="#b8322f")
        self.acc = ax.text(0.1, 0.0, self.T["accelerated"].format(**self.d["fmt"]), fontsize=10,
                           color=INK2, style="italic")
        self.parts = _particles_artist(ax, zorder=6)
        self.drops = []
        tx = self.text_axes()
        self.title(tx, self.T["s5_title"])
        self.arts = self.lines(tx, [l.format(**self.d["fmt"]) for l in self.T["s5_lines"]], y0=0.80, dy=0.08, size=12.5)

    def update(self, t, dt):
        rng = self.rng
        T_fill, T_void = 9.0, 12.0
        filling = t < T_fill
        if t < T_fill:
            level = 0.08 + 0.80 * _ease(t / T_fill)
        elif t < T_void:
            level = 0.88 - 0.83 * _ease((t - T_fill) / (T_void - T_fill))
        else:
            level = 0.05 + 0.03 * _ease((t - T_void) / 3)
        self.fill.set_height(self.bh * level)
        vol = self.d["fmt"]["bladder_ml"] * (level - 0.05) / 0.83
        self.vol_txt.set_text(f"{max(vol, 0):.0f} mL")
        self.mic.set_text(self.T["micturition"] + " ↓" if T_fill <= t < T_void else "")
        # peristaltic drops in the ureters (also during emptying, the kidney never stops)
        if rng.random() < 6 * dt:
            for u in range(2):
                self.drops.append({"k": "ureter", "path": u, "u": 0.0})
        if T_fill <= t < T_void:
            for _ in range(rng.poisson(25 * dt)):
                self.drops.append({"k": "out", "x": 5.0 + rng.normal(0, 0.04), "y": 0.55, "vy": -1.5})
        xs, ys, ss, al = [], [], [], []
        keep = []
        for d in self.drops:
            if d["k"] == "ureter":
                d["u"] += dt * 0.6
                if d["u"] >= 1:
                    continue
                x, y = self.ureters[d["path"]].point(d["u"])
            else:
                d["vy"] -= 6 * dt
                d["y"] += d["vy"] * dt
                if d["y"] < -0.2:
                    continue
                x, y = d["x"], d["y"]
            xs.append(x); ys.append(y); ss.append("urea"); al.append(1.0)
            keep.append(d)
        self.drops = keep
        _set_particles(self.parts, xs, ys, ss, np.array(al))
        self.parts.set_facecolors(_rgba(["#e0a800"] * len(xs), np.array(al)) if xs else [])
        self.reveal(self.arts, t, 0.6, 1.2)


class SummaryScene(Scene):
    duration = 7.0

    def setup(self):
        ax = self.fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 16); ax.set_ylim(0, 9)
        ax.text(8, 7.9, self.T["s6_title"], ha="center", fontsize=30, fontweight="bold", color=INK)
        cols = ["#e34948", "#2a78d6", "#eda100"]
        xs = [2.6, 8.0, 13.4]
        self.items = []
        for i, (x, txt) in enumerate(zip(xs, self.T["s6_boxes"])):
            box = FancyBboxPatch((x - 1.9, 4.2), 3.8, 1.9, boxstyle="round,pad=0.1,rounding_size=0.3",
                                 fc=cols[i], ec="none", alpha=0.15)
            ax.add_patch(box)
            tt = ax.text(x, 5.15, txt.format(**self.d["fmt"]), ha="center", va="center", fontsize=19,
                         fontweight="bold", color=INK)
            self.items.append([box, tt])
        for i, txt in enumerate(self.T["s6_arrows"]):
            x0, x1 = xs[i] + 2.05, xs[i + 1] - 2.05
            arr = FancyArrowPatch((x0, 5.15), (x1, 5.15), arrowstyle="-|>", mutation_scale=24, lw=2.5, color=INK2)
            ax.add_patch(arr)
            tt = ax.text((x0 + x1) / 2, 3.55, txt, ha="center", va="top", fontsize=12.5, color=INK2)
            self.items.insert(2 * i + 1, [arr, tt])
        self.foot = [ax.text(8, 2.0 - 0.7 * i, l, ha="center", fontsize=15, color=INK2 if i else INK)
                     for i, l in enumerate(self.T["s6_lines"])]

    def update(self, t, dt):
        for i, group in enumerate(self.items):
            a = _ease((t - 0.4 - 0.5 * i) / 0.5)
            for g in group:
                g.set_alpha(a * (0.15 if isinstance(g, FancyBboxPatch) else 1))
        for i, f in enumerate(self.foot):
            f.set_alpha(_ease((t - 3.2 - 0.6 * i) / 0.5))


SCENES = [TitleScene, BloodInScene, FiltrationScene, TubuleScene, UrineScene, SummaryScene]


# ---------------------------------------------------------------------------
# model data and rendering
# ---------------------------------------------------------------------------
def model_data(nephron_kwargs=None):
    """Run Levels 1 and 2 and collect every number the video shows."""
    h = glom.hemodynamics()
    f = glom.filtration_along_capillary(P_gc=h["P_gc"], Q_plasma=h["Q_plasma"])
    kw = dict(gfr=f["gfr"])
    kw.update(nephron_kwargs or {})
    r = tub.run_nephron(**kw)
    urine = r["urine_flow_ml_min"]
    bladder_ml = 300.0
    fmt = dict(
        blood=h["Q_blood"], blood_ml=h["Q_blood"] * 1000,
        pgc=h["P_gc"], pbs=k.P_BOWMAN, pi0=f["pi_gc"][0], pi1=f["pi_gc"][-1],
        gfr=r["gfr"] * 1000, gfr_day=r["gfr"] * 1440,
        urine=urine, urine_day=urine * 1.44, uosm=r["urine_osm"],
        uurea=r["urine_conc"]["urea"], ratio=r["urine_conc"]["urea"] / r["plasma"]["urea"],
        reabs=100 * (1 - r["urine_flow"] / r["gfr"]),
        bladder_ml=bladder_ml, hours=bladder_ml / max(urine, 1e-6) / 60,
    )
    return {"filt": f, "nephron": r, "fmt": fmt}


def _render_scene(cls, data, lang, path, fps=FPS, crf=20):
    scene = cls(data, lang)
    writer = FFMpegWriter(fps=fps, codec="libx264",
                          extra_args=["-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"])
    n = int(round(scene.duration * fps))
    tmp = path[:-4] + ".part.mp4"
    with writer.saving(scene.fig, tmp, dpi=DPI):
        for i in range(n):
            scene.frame(i / fps, 1 / fps)
            writer.grab_frame(facecolor=BG)
    plt.close(scene.fig)
    os.replace(tmp, path)


def preview(scene_index, t, path, lang="es", nephron_kwargs=None, data=None):
    """Save one still frame (PNG) of a scene at time t, simulating up to t."""
    data = data or model_data(nephron_kwargs)
    scene = SCENES[scene_index](data, lang)
    dt = 1 / FPS
    for i in range(int(t * FPS) + 1):
        scene.frame(i * dt, dt)
    scene.fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(scene.fig)


def make_video(path="kidney_video.mp4", lang="es", nephron_kwargs=None, scenes=None, verbose=True):
    """Render the full video (about 80 s, 1280x720, 30 fps). Needs ffmpeg.

    nephron_kwargs: options passed to tubule.run_nephron (e.g. {"ADH": 1.0}).
    scenes: optional list of scene indices to render (default: all six).
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found (it is installed in Google Colab).")
    matplotlib.use("Agg", force=False)
    data = model_data(nephron_kwargs)
    idx = range(len(SCENES)) if scenes is None else scenes
    work = tempfile.mkdtemp(prefix="kidney_video_")
    parts = []
    for i in idx:
        p = os.path.join(work, f"scene{i}.mp4")
        if verbose:
            print(f"rendering scene {i + 1}/{len(SCENES)}: {SCENES[i].__name__} ({SCENES[i].duration:.0f} s)")
        _render_scene(SCENES[i], data, lang, p)
        parts.append(p)
    listfile = os.path.join(work, "list.txt")
    with open(listfile, "w") as fh:
        for p in parts:
            fh.write(f"file '{p}'\n")
    tmp = path[:-4] + ".part.mp4" if path.endswith(".mp4") else path + ".part.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", listfile, "-c", "copy", "-movflags", "+faststart", tmp], check=True)
    os.replace(tmp, path)
    shutil.rmtree(work, ignore_errors=True)
    if verbose:
        print("saved", path)
    return path
