"""
3D figures. Two back-ends:

    backend="plotly"      interactive (rotate, zoom, hover) - best in Colab
    backend="matplotlib"  static, works everywhere
    backend="auto"        plotly if it is installed, otherwise matplotlib

Every function returns a figure: call fig.show() for plotly; matplotlib
figures display automatically in a notebook.
"""

import numpy as np

from . import geometry as geo
from . import glomerulus as glom

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:          # pragma: no cover
    HAS_PLOTLY = False

BLUES_STOPS = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
               "#256abf", "#184f95", "#0d366b"]
PLOTLY_BLUES = [[i / (len(BLUES_STOPS) - 1), c] for i, c in enumerate(BLUES_STOPS)]
HEAT_STOPS = ["#cde2fb", "#6da7ec", "#f0efec", "#ec835a", "#d03b3b"]   # low -> high pressure
PLOTLY_HEAT = [[i / (len(HEAT_STOPS) - 1), c] for i, c in enumerate(HEAT_STOPS)]


def _mixed_columns(*cols):
    """customdata with text and numbers (keeps numbers numeric for formatting)."""
    out = np.empty((len(cols[0]), len(cols)), dtype=object)
    for j, c in enumerate(cols):
        out[:, j] = list(c)
    return out


def _backend(backend):
    if backend == "auto":
        return "plotly" if HAS_PLOTLY else "matplotlib"
    if backend == "plotly" and not HAS_PLOTLY:
        raise ImportError("plotly is not installed: pip install plotly")
    return backend


def _mpl_cmap(stops):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("c", stops)


def _plotly_layout(fig, title, xt, yt, zt, aspect=None):
    scene = dict(xaxis=dict(title=dict(text=xt)),
                 yaxis=dict(title=dict(text=yt)),
                 zaxis=dict(title=dict(text=zt)))
    if aspect:
        scene["aspectmode"] = "manual"
        scene["aspectratio"] = dict(x=aspect[0], y=aspect[1], z=aspect[2])
    fig.update_layout(title=dict(text=title), scene=scene, height=600,
                      margin=dict(l=0, r=0, t=50, b=0), paper_bgcolor="#fcfcfb")
    return fig


def _surface(X, Y, Z, title, xt, yt, zt, backend, stops=BLUES_STOPS, Z2=None, label2=None):
    """Generic surface helper (optionally a second, semi-transparent surface)."""
    b = _backend(backend)
    if b == "plotly":
        scale = [[i / (len(stops) - 1), c] for i, c in enumerate(stops)]
        data = [go.Surface(x=X, y=Y, z=Z, colorscale=scale,
                           colorbar=dict(title=dict(text=zt)), name=zt,
                           hovertemplate=f"{xt}: %{{x:.1f}}<br>{yt}: %{{y:.2f}}<br>{zt}: %{{z:.1f}}<extra></extra>")]
        if Z2 is not None:
            data.append(go.Surface(x=X, y=Y, z=Z2, opacity=0.45, showscale=False,
                                   colorscale=[[0, "#eb6834"], [1, "#eb6834"]], name=label2,
                                   hovertemplate=f"{label2}: %{{z:.1f}}<extra></extra>"))
        return _plotly_layout(go.Figure(data=data), title, xt, yt, zt)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=_mpl_cmap(stops), linewidth=0, antialiased=True, alpha=0.95,
                           rcount=min(Z.shape[0], 150), ccount=min(Z.shape[1], 150))
    if Z2 is not None:
        ax.plot_wireframe(X, Y, Z2, color="#eb6834", linewidth=0.6, rstride=3, cstride=3, label=label2)
        ax.legend(loc="upper left")
    ax.set_xlabel(xt); ax.set_ylabel(yt); ax.set_zlabel(zt)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.6, label=zt)
    return fig


# ---- Level 1 ---------------------------------------------------------------------
def gfr_surface(maps=None, kf_fractions=None, backend="auto"):
    """GFR as a function of blood pressure and filter health (fraction of K_f).
    Solid surface: with autoregulation. Orange: rigid vessels."""
    maps = np.linspace(40, 200, 33) if maps is None else maps
    kf_fractions = np.linspace(0.1, 1.0, 19) if kf_fractions is None else kf_fractions
    X, Y = np.meshgrid(maps, kf_fractions)
    Z = np.zeros_like(X); Z2 = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Kf = glom.KF_NORMAL * Y[i, j]
            Z[i, j] = glom.gfr_from_map(X[i, j], autoregulation=True, Kf=Kf) * 1000
            Z2[i, j] = glom.gfr_from_map(X[i, j], autoregulation=False, Kf=Kf) * 1000
    Z2 = np.minimum(Z2, 300)
    return _surface(X, Y, Z, "GFR vs blood pressure and filter health",
                    "MAP (mmHg)", "K_f (fraction of normal)", "GFR (mL/min)", backend,
                    Z2=Z2, label2="rigid vessels (no autoregulation)")


def glomerulus_3d(f=None, backend="auto", n_loops=8):
    """Capillary loops of the glomerulus coloured by net filtration pressure,
    inside a transparent Bowman's capsule."""
    if f is None:
        f = glom.filtration_along_capillary()
    loops = geo.glomerulus_loops(n_loops=n_loops, n=len(f["x"]))
    nfp = f["net_pressure"]
    b = _backend(backend)
    if b == "plotly":
        data = []
        for i, (x, y, z) in enumerate(loops):
            data.append(go.Scatter3d(
                x=x, y=y, z=z, mode="lines", showlegend=False,
                line=dict(color=nfp, colorscale=PLOTLY_HEAT, width=7, cmin=0, cmax=float(nfp.max()),
                          colorbar=dict(title=dict(text="net filtration<br>pressure (mmHg)")) if i == 0 else None,
                          showscale=(i == 0)),
                customdata=np.column_stack([f["x"], f["pi_gc"], nfp]),
                hovertemplate="position %{customdata[0]:.2f}<br>net pressure %{customdata[2]:.1f} mmHg"
                              "<br>oncotic %{customdata[1]:.1f} mmHg<extra></extra>"))
        sx, sy, sz = geo.sphere()
        data.append(go.Surface(x=sx, y=sy, z=sz, opacity=0.12, showscale=False,
                               colorscale=[[0, "#6250d6"], [1, "#6250d6"]], hoverinfo="skip"))
        fig = go.Figure(data=data)
        fig = _plotly_layout(fig, f"Glomerulus: filtration fades along each loop (GFR {f['gfr']*1000:.0f} mL/min)",
                             "x", "y", "z", aspect=(1, 1, 1))
        return fig
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = _mpl_cmap(HEAT_STOPS)
    for x, y, z in loops:
        pts = np.column_stack([x, y, z]).reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(segs, cmap=cmap, linewidths=3)
        lc.set_array(nfp[:-1]); lc.set_clim(0, nfp.max())
        ax.add_collection3d(lc)
    sx, sy, sz = geo.sphere()
    ax.plot_wireframe(sx, sy, sz, color="#6250d6", alpha=0.12, linewidth=0.5)
    ax.set_xlim(-0.14, 0.14); ax.set_ylim(-0.14, 0.14); ax.set_zlim(-0.19, 0.09)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Glomerulus coloured by net filtration pressure")
    fig.colorbar(lc, ax=ax, shrink=0.6, label="net filtration pressure (mmHg)")
    return fig


# ---- Level 2 ---------------------------------------------------------------------
LABELS = {"osm": "osmolarity (mOsm/L)", "F": "flow (L/min)", "C_Na": "Na (mmol/L)",
          "C_urea": "urea (mmol/L)", "C_glucose": "glucose (mmol/L)", "C_drug": "drug (mmol/L)"}


def nephron_3d(r, quantity="osm", backend="auto"):
    """The nephron in 3D, coloured by any profile quantity (osm, C_Na, C_urea, F...).
    A vertical wall behind it shows the tissue osmolarity (cortex -> papilla)."""
    from .tubule import SEGMENTS
    p = r["profile"]
    x, y, z = geo.nephron_xyz(p)
    vals = p[quantity]
    label = LABELS.get(quantity, quantity)
    seg_names = np.array(SEGMENTS)[p["segment"]]
    # tissue wall
    zz = np.linspace(0, -1, 40)
    xx = np.linspace(-0.2, 1.1, 2)
    WX, WZ = np.meshgrid(xx, zz)
    WY = np.full_like(WX, -0.3)
    WC = np.interp(-WZ, -z[np.argsort(-z)], p["osm_interstitium"][np.argsort(-z)])
    b = _backend(backend)
    if b == "plotly":
        data = [
            go.Surface(x=WX, y=WY, z=WZ, surfacecolor=WC, colorscale="Greys", opacity=0.35,
                       showscale=False, hoverinfo="skip"),
            go.Scatter3d(x=x, y=y, z=z, mode="lines",
                         line=dict(color=vals, colorscale=PLOTLY_BLUES, width=10,
                                   colorbar=dict(title=dict(text=label))),
                         customdata=_mixed_columns(seg_names, vals, p["F"] * 1000),
                         hovertemplate="%{customdata[0]}<br>" + label + ": %{customdata[1]:.3g}"
                                       "<br>flow %{customdata[2]:.2f} mL/min<extra></extra>"),
        ]
        sx, sy, sz = geo.sphere(radius=0.07)
        data.append(go.Surface(x=sx, y=sy, z=sz, opacity=0.4, showscale=False,
                               colorscale=[[0, "#e34948"], [1, "#e34948"]],
                               hovertemplate="glomerulus<extra></extra>"))
        fig = go.Figure(data=data)
        return _plotly_layout(fig, f"Nephron coloured by {label}", "x", "y", "depth (0 cortex, -1 papilla)",
                              aspect=(1.3, 0.6, 1.2))
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(WX, WY, WZ, facecolors=plt.cm.Greys(WC / 1500), alpha=0.25, linewidth=0, shade=False)
    pts = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, cmap=_mpl_cmap(BLUES_STOPS), linewidths=4)
    lc.set_array(vals[:-1])
    ax.add_collection3d(lc)
    sx, sy, sz = geo.sphere(radius=0.07)
    ax.plot_surface(sx, sy, sz, color="#e34948", alpha=0.35, linewidth=0)
    ax.set_xlim(-0.2, 1.1); ax.set_ylim(-0.3, 0.3); ax.set_zlim(-1.05, 0.05)
    ax.set_box_aspect((1.3, 0.6, 1.2))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("depth")
    ax.set_title(f"Nephron coloured by {label}")
    fig.colorbar(lc, ax=ax, shrink=0.6, label=label)
    return fig


# ---- Level 3 ---------------------------------------------------------------------
def countercurrent_surface(res, every=2, backend="auto"):
    """Osmolarity (z) vs depth and cycle: watch the gradient grow."""
    hD = res["history_D"][::every]
    cycles = np.arange(hD.shape[0]) * every
    X, Y = np.meshgrid(res["depth"], cycles)
    return _surface(X, Y, hD, "Countercurrent multiplier: gradient grows cycle by cycle",
                    "depth (0 cortex, 1 papilla)", "cycle", "osmolarity (mOsm/L)", backend)


# ---- Level 5 ---------------------------------------------------------------------
def column_surface(col, which="c", backend="auto"):
    """Ion-exchange column: NH4+ in liquid (which='c') or on resin ('q') vs z and t."""
    Z = col[which]
    X, Y = np.meshgrid(col["z"], col["t"])
    label = "NH4+ in liquid (mmol/L)" if which == "c" else "NH4+ on resin (mmol/g)"
    return _surface(X, Y, Z, "Ion-exchange column: the saturation front travels",
                    "position in column", "time (min)", label, backend)


def dialyzer_3d(prof, n_fibres=37, backend="auto"):
    """Hollow-fibre bundle, each fibre coloured by blood urea along its length."""
    yz = geo.fibre_bundle(n_fibres)
    x = prof["x"] * 4.0                       # stretch so it looks like a cartridge
    cb = prof["C_blood"]
    b = _backend(backend)
    if b == "plotly":
        data = []
        for i, (yy, zz) in enumerate(yz):
            data.append(go.Scatter3d(
                x=x, y=np.full_like(x, yy), z=np.full_like(x, zz), mode="lines", showlegend=False,
                line=dict(color=cb, colorscale=PLOTLY_BLUES, width=6, cmin=0, cmax=float(cb.max()),
                          colorbar=dict(title=dict(text="blood urea<br>(mmol/L)")) if i == 0 else None,
                          showscale=(i == 0)),
                customdata=np.column_stack([cb, prof["C_dialysate"]]),
                hovertemplate="blood urea %{customdata[0]:.1f}<br>dialysate %{customdata[1]:.1f} mmol/L"
                              "<extra></extra>"))
        # transparent shell
        th = np.linspace(0, 2 * np.pi, 40)
        XS, TH = np.meshgrid(x, th)
        data.append(go.Surface(x=XS, y=1.1 * np.cos(TH), z=1.1 * np.sin(TH), opacity=0.08,
                               showscale=False, colorscale=[[0, "#52514e"], [1, "#52514e"]],
                               hoverinfo="skip"))
        fig = go.Figure(data=data)
        fig.add_annotation(text="blood in →   ...   ← fresh dialysate in", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.02)
        return _plotly_layout(fig, f"Dialyzer fibres: urea removed along the length "
                                   f"(K = {prof['clearance']*1000:.0f} mL/min)",
                              "length", "y", "z", aspect=(2.5, 1, 1))
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    cmap = _mpl_cmap(BLUES_STOPS)
    for yy, zz in yz:
        pts = np.column_stack([x, np.full_like(x, yy), np.full_like(x, zz)]).reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(segs, cmap=cmap, linewidths=2)
        lc.set_array(cb[:-1]); lc.set_clim(0, cb.max())
        ax.add_collection3d(lc)
    ax.set_xlim(0, 4); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
    ax.set_box_aspect((2.5, 1, 1))
    ax.set_xlabel("length (blood →)")
    ax.set_title(f"Dialyzer fibres coloured by blood urea (K = {prof['clearance']*1000:.0f} mL/min)")
    fig.colorbar(lc, ax=ax, shrink=0.6, label="blood urea (mmol/L)")
    return fig


def show(fig):
    """Display a figure from either back-end."""
    if HAS_PLOTLY and isinstance(fig, go.Figure):
        fig.show()
    else:
        import matplotlib.pyplot as plt
        plt.show()
