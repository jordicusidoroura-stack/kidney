"""
Runs every code cell of every notebook (without Jupyter) to check that
nothing is broken. Usage:  python tools/run_notebooks.py
"""
import glob, json, os, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import IPython.display  # noqa: F401
    HAVE_IPYTHON = True
except ImportError:
    HAVE_IPYTHON = False


def _fake_ipython():
    """Outside Jupyter, the display helpers become no-ops (only while a cell runs)."""
    import types
    ipy = types.ModuleType("IPython")
    disp = types.ModuleType("IPython.display")
    disp.display = lambda *a, **k: None
    disp.Image = disp.Video = lambda *a, **k: None
    ipy.display = disp
    sys.modules["IPython"] = ipy
    sys.modules["IPython.display"] = disp


root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(os.path.join(root, "notebooks"))
ok = True
for nb in sorted(glob.glob("*.ipynb")):
    cells = json.load(open(nb))["cells"]
    ns = {}
    t0 = time.time()
    try:
        for c in cells:
            if c["cell_type"] == "code":
                src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
                fake = "IPython" in src and not HAVE_IPYTHON
                if fake:
                    _fake_ipython()
                try:
                    exec(compile(src, nb, "exec"), ns)
                finally:
                    if fake:
                        sys.modules.pop("IPython", None)
                        sys.modules.pop("IPython.display", None)
                plt.close("all")
        print(f"OK    {nb}  ({time.time()-t0:.1f} s)")
    except Exception as e:
        ok = False
        print(f"FAIL  {nb}: {type(e).__name__}: {e}")
    os.chdir(os.path.join(root, "notebooks"))
sys.exit(0 if ok else 1)
