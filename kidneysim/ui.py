"""
Tiny helper for interactive sliders.

In Google Colab / Jupyter, `sliders(func, a=(0, 1, 0.1), ...)` shows sliders
(ipywidgets). If ipywidgets is not available, the function just runs once
with the middle/default values, so notebooks never break.
"""


def sliders(func, **ranges):
    """ranges: name=(min, max, step) or name=(min, max, step, default)
    or name=[list of options]."""
    try:
        from ipywidgets import interact, FloatSlider, Dropdown
    except ImportError:
        defaults = {}
        for name, r in ranges.items():
            if isinstance(r, list):
                defaults[name] = r[0]
            else:
                defaults[name] = r[3] if len(r) > 3 else r[0] + (r[1] - r[0]) / 2
        return func(**defaults)

    widgets = {}
    for name, r in ranges.items():
        if isinstance(r, list):
            widgets[name] = Dropdown(options=r, description=name)
        else:
            lo, hi, step = r[:3]
            value = r[3] if len(r) > 3 else lo + (hi - lo) / 2
            widgets[name] = FloatSlider(min=lo, max=hi, step=step, value=value,
                                        description=name, continuous_update=False,
                                        style={"description_width": "initial"})
    return interact(func, **widgets)
