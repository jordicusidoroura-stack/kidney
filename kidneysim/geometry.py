"""
Simple shapes used to DRAW the models (they do not change the physics).

Coordinates: x, y horizontal, z vertical. z = -depth, so the kidney surface
(cortex) is at the top (z = 0) and the papilla is at the bottom (z = -1).
The medulla starts at depth 0.3.
"""

import numpy as np

CORTEX_DEPTH = 0.3


def _segment_xyz(seg, s):
    s = np.asarray(s, dtype=float)
    if seg == "PT":        # convoluted (coiled) tube near the glomerulus
        x = 0.15 + 0.35 * s + 0.08 * np.cos(10 * np.pi * s)
        y = 0.08 * np.sin(10 * np.pi * s)
        d = 0.05 + 0.20 * s
    elif seg == "DL":      # straight down into the medulla
        x = 0.58 + 0.02 * s
        y = np.zeros_like(s)
        d = 0.25 + 0.75 * s
    elif seg == "AL":      # hairpin turn, then straight up
        turn = np.clip(s / 0.06, 0, 1)
        x = 0.60 + 0.15 * turn
        y = 0.04 * np.sin(np.pi * turn)
        d = 1.0 - 0.75 * s
    elif seg == "DCT":     # coils back to touch its own glomerulus
        x = 0.75 - 0.45 * s
        y = 0.12 * np.sin(np.pi * s) + 0.04 * np.sin(8 * np.pi * s)
        d = 0.25 - 0.15 * s
    elif seg == "CD":      # joins a collecting duct running down to the papilla
        x = 0.30 + 0.70 * np.clip(s / 0.15, 0, 1)
        y = np.zeros_like(s)
        d = 0.10 + 0.90 * s
    else:
        raise ValueError(seg)
    return x, y, -d


def nephron_xyz(profile):
    """x, y, z for every point of a tubule.run_nephron() profile."""
    from .tubule import SEGMENTS
    pos = profile["position"]
    seg_idx = np.clip(np.floor(pos).astype(int), 0, len(SEGMENTS) - 1)
    s = pos - seg_idx
    x = np.empty_like(pos); y = np.empty_like(pos); z = np.empty_like(pos)
    for j, seg in enumerate(SEGMENTS):
        m = seg_idx == j
        x[m], y[m], z[m] = _segment_xyz(seg, s[m])
    return x, y, z


def glomerulus_loops(n_loops=8, n=200, radius=0.1, center=(0.0, 0.0, -0.05), seed=1):
    """A ball of capillary loops. Returns list of (x, y, z) arrays; along each
    loop the parameter goes 0 -> 1 from the afferent to the efferent arteriole."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    cx, cy, cz = center
    loops = []
    for j in range(n_loops):
        phi0 = 2 * np.pi * j / n_loops + rng.uniform(-0.2, 0.2)
        theta = 0.15 + np.pi * 0.8 * np.sin(np.pi * t)     # out to the far side and back
        phi = phi0 + 0.9 * np.sin(2 * np.pi * t) + 0.3 * np.sin(6 * np.pi * t)
        r = radius * (0.75 + 0.2 * np.sin(5 * np.pi * t + j))
        x = cx + r * np.sin(theta) * np.cos(phi)
        y = cy + r * np.sin(theta) * np.sin(phi)
        z = cz + r * np.cos(theta)
        loops.append((x, y, z))
    return loops


def sphere(radius=0.13, center=(0.0, 0.0, -0.05), n=30):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    cx, cy, cz = center
    x = cx + radius * np.outer(np.cos(u), np.sin(v))
    y = cy + radius * np.outer(np.sin(u), np.sin(v))
    z = cz + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def fibre_bundle(n_fibres=37, radius=1.0):
    """(y, z) positions of hollow fibres packed in a round dialyzer."""
    pts = [(0.0, 0.0)]
    ring = 1
    while len(pts) < n_fibres:
        k = 6 * ring
        for i in range(k):
            a = 2 * np.pi * i / k
            pts.append((ring * np.cos(a), ring * np.sin(a)))
        ring += 1
    pts = np.array(pts[:n_fibres])
    return pts * radius / (ring - 0.5)
