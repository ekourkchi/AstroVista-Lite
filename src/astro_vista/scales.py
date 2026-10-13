"""Display scales: limits, stretches, colour maps, contrast and bias.

An image is shown in three steps:

1. **Limits**: the pixel values mapped to the darkest and brightest colours (min-max,
   ZScale, percentiles, or typed).
2. **Stretch**: values between the limits, scaled to 0..1, are stretched (linear, log,
   power, square root, squared, asinh, sinh, or histogram equalisation).
3. **Colour map**, adjusted by **contrast and bias** (as in DS9: dragging with the right
   mouse button moves them).  Only the colour table changes, so this step is instant.

The stretch formulas are the usual ones of astronomical display programs, written out
here; display changes never change pixel values.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pyqtgraph as pg

LIMITS = (
    "99.5%",
    "99%",
    "98%",
    "95%",
    "90%",
    "ZScale",
    "ZMax",
    "Min-max",
    "Manual",
)
STRETCHES = ("Linear", "Log", "Power", "Square root", "Squared", "Asinh", "Sinh", "Histogram")
COLOURMAPS = (
    "Grey",
    "Heat",
    "Cool",
    "Rainbow",
    "Red",
    "Green",
    "Blue",
    "inferno",
    "magma",
    "plasma",
    "viridis",
    "cividis",
)
SAMPLE = 2_000_000  # limits and histograms of larger images come from a regular sample


def sample(data: np.ndarray) -> np.ndarray:
    """The finite pixels of ``data`` (a regular sample of them for large images)."""
    flat = data.ravel()
    if flat.size > SAMPLE:
        flat = flat[:: flat.size // SAMPLE + 1]
    return flat[np.isfinite(flat)]


def limits(data: np.ndarray, mode: str) -> Tuple[float, float]:
    """The display limits of ``data`` in the mode ``mode`` (not Manual)."""
    values = sample(data)
    if not values.size:
        return 0.0, 1.0
    if mode.endswith("%"):
        tail = (100.0 - float(mode[:-1])) / 2.0
        lo, hi = np.percentile(values, [tail, 100.0 - tail])
    elif mode in ("ZScale", "ZMax"):
        from astropy.visualization import ZScaleInterval

        lo, hi = ZScaleInterval().get_limits(values)
        if mode == "ZMax":
            hi = values.max()
    else:
        lo, hi = values.min(), values.max()
    lo, hi = float(lo), float(hi)
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def stretch(t: np.ndarray, name: str, exponent: float = 1000.0, data=None) -> np.ndarray:
    """Stretch ``t`` (0..1).  ``exponent`` is the constant of the log and power
    stretches; histogram equalisation needs the scaled pixels themselves (``t``)."""
    t = np.clip(t, 0.0, 1.0)
    a = max(float(exponent), 1.0001)
    with np.errstate(all="ignore"):
        if name == "Log":
            return np.log10(a * t + 1.0) / np.log10(a + 1.0)
        if name == "Power":
            return (np.power(a, t) - 1.0) / (a - 1.0)
        if name == "Square root":
            return np.sqrt(t)
        if name == "Squared":
            return t * t
        if name == "Asinh":
            return np.arcsinh(10.0 * t) / np.arcsinh(10.0)
        if name == "Sinh":
            return np.sinh(3.0 * t) / np.sinh(3.0)
        if name == "Histogram":
            values = np.sort(sample(t))
            if not values.size:
                return t
            # each value goes to the fraction of the pixels below it: 0 for the
            # faintest, 1 for the brightest
            ranks = np.searchsorted(values, t, side="left") / max(values.size - 1, 1)
            return np.clip(ranks, 0.0, 1.0)
    return t


def scaled(data: np.ndarray, lo: float, hi: float, name: str, exponent: float) -> np.ndarray:
    """``data`` between ``lo`` and ``hi``, stretched to 0..1 (NaN shows as 0)."""
    finite = np.isfinite(data)
    t = (np.where(finite, data, lo) - lo) / (hi - lo)
    return np.asarray(stretch(t, name, exponent), dtype=np.float32)


def _ramp(points) -> np.ndarray:
    """A 256-entry colour table through ``points``: [(position 0..1, (r, g, b)), ...]."""
    x = np.linspace(0.0, 1.0, 256)
    pos = [p for p, _ in points]
    rgb = np.array([c for _, c in points], float)
    return np.stack([np.interp(x, pos, rgb[:, k]) for k in range(3)], axis=1)


RAMPS: Dict[str, list] = {
    "Grey": [(0, (0, 0, 0)), (1, (255, 255, 255))],
    "Heat": [(0, (0, 0, 0)), (0.34, (255, 0, 0)), (0.67, (255, 255, 0)), (1, (255, 255, 255))],
    "Cool": [(0, (0, 0, 0)), (0.3, (0, 0, 160)), (0.65, (0, 200, 255)), (1, (255, 255, 255))],
    "Rainbow": [
        (0, (40, 0, 90)),
        (0.2, (0, 0, 255)),
        (0.4, (0, 210, 255)),
        (0.55, (0, 220, 60)),
        (0.75, (255, 230, 0)),
        (0.9, (255, 90, 0)),
        (1, (255, 0, 0)),
    ],
    "Red": [(0, (0, 0, 0)), (1, (255, 0, 0))],
    "Green": [(0, (0, 0, 0)), (1, (0, 255, 0))],
    "Blue": [(0, (0, 0, 0)), (1, (0, 0, 255))],
}


def colour_table(name: str, invert=False, contrast=1.0, bias=0.5) -> np.ndarray:
    """The 256 x 3 colour table, with contrast (slope) and bias (centre) applied."""
    if name in RAMPS:
        base = _ramp(RAMPS[name])
    else:
        base = pg.colormap.get(name).getLookupTable(nPts=256).astype(float)
    if invert:
        base = base[::-1]
    # entry k of the table shows the colour the base table has at (k - bias) * contrast
    k = np.linspace(0.0, 1.0, 256)
    source = np.clip((k - bias) * contrast + 0.5, 0.0, 1.0)
    return base[np.round(source * 255).astype(int)].astype(np.uint8)


def colour_bar_ticks(lo: float, hi: float, name: str, exponent: float, count: int = 5):
    """Values at evenly spaced positions along the colour bar (the stretch inverted)."""
    values = np.linspace(lo, hi, 2001)
    positions = stretch(
        (values - lo) / (hi - lo), name if name != "Histogram" else "Linear", exponent
    )
    marks = np.linspace(0.0, 1.0, count)
    return [(float(p), float(np.interp(p, positions, values))) for p in marks]
