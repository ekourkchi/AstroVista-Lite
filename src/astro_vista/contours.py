"""Display-only contour overlays.

Contours are traced with a vectorised marching-squares algorithm (the
standard method: classify each 2×2 cell of pixel centres by which corners
are at or above the level, then join the linearly interpolated edge
crossings).  It is written here with NumPy so that no Python loop runs over
pixels.  The result is a set of independent line segments, drawn as pairs.

Large buffers are sampled on every k-th pixel (no averaging, so no
smoothing is applied); the stride is reported so the overlay can say so.
Contours never change pixel values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Largest sampled grid traced per contour set (pixels on a side).
MAX_TRACE_SIZE = 2048
AUTO_PERCENTILES = (50.0, 99.5)

# Cell corners: a = (row i, col j), b = (i, j+1), c = (i+1, j+1), d = (i+1, j).
# Case index bits: a -> 1, b -> 2, c -> 4, d -> 8 (set when the corner >= level).
# Edges: 0 = a-b (lower), 1 = b-c (right), 2 = d-c (upper), 3 = a-d (left).
# Each case lists the edge pairs joined by a segment.
SEGMENTS: Dict[int, List[Tuple[int, int]]] = {
    1: [(3, 0)],
    2: [(0, 1)],
    3: [(3, 1)],
    4: [(1, 2)],
    6: [(0, 2)],
    7: [(3, 2)],
    8: [(2, 3)],
    9: [(0, 2)],
    11: [(1, 2)],
    12: [(1, 3)],
    13: [(0, 1)],
    14: [(3, 0)],
}
# Saddle cells (diagonal corners high) are resolved with the cell-centre mean:
# a high centre joins the two high corners, so the segments cut off the low ones.
SADDLE_SEGMENTS = {
    5: {True: [(0, 1), (2, 3)], False: [(3, 0), (1, 2)]},
    10: {True: [(3, 0), (1, 2)], False: [(0, 1), (2, 3)]},
}


@dataclass
class ContourGeometry:
    """Line segments in the scene frame: points 2k and 2k+1 form segment k."""

    x: np.ndarray
    y: np.ndarray
    stride: int


def auto_levels(data: np.ndarray, count: int) -> List[float]:
    """``count`` linearly spaced levels between the 50th and 99.5th percentiles of the finite pixels."""
    finite = data[np.isfinite(data)]
    if finite.size == 0 or count < 1:
        return []
    low, high = np.percentile(finite, AUTO_PERCENTILES)
    if high <= low:
        return [float(low)]
    return [float(v) for v in np.linspace(low, high, count + 2)[1:-1]]


def _edge_points(edge: int, corners, level: float, rows, cols):
    """Interpolated crossing on ``edge`` for the selected cells (array coordinates x=col, y=row)."""
    a, b, c, d = corners

    def fraction(v_from, v_to):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (level - v_from) / (v_to - v_from)
        return np.clip(np.nan_to_num(t, nan=0.5), 0.0, 1.0)

    if edge == 0:
        return cols + fraction(a, b), rows.astype(float)
    if edge == 1:
        return cols + 1.0, rows + fraction(b, c)
    if edge == 2:
        return cols + fraction(d, c), rows + 1.0
    return cols.astype(float), rows + fraction(a, d)


def _trace_level(grid: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray]:
    """Segments of one level on a finite 2-D grid, as interleaved endpoint arrays."""
    a, b = grid[:-1, :-1], grid[:-1, 1:]
    d, c = grid[1:, :-1], grid[1:, 1:]
    case = (
        (a >= level).astype(np.uint8)
        | ((b >= level).astype(np.uint8) << 1)
        | ((c >= level).astype(np.uint8) << 2)
        | ((d >= level).astype(np.uint8) << 3)
    )
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    def emit(selected: np.ndarray, pairs: List[Tuple[int, int]]) -> None:
        rows, cols = np.nonzero(selected)
        if rows.size == 0:
            return
        corners = (a[rows, cols], b[rows, cols], c[rows, cols], d[rows, cols])
        for first, second in pairs:
            x0, y0 = _edge_points(first, corners, level, rows, cols)
            x1, y1 = _edge_points(second, corners, level, rows, cols)
            xs.append(np.column_stack([x0, x1]).ravel())
            ys.append(np.column_stack([y0, y1]).ravel())

    for index, pairs in SEGMENTS.items():
        emit(case == index, pairs)
    centre_high = (a + b + c + d) / 4.0 >= level
    for index, choice in SADDLE_SEGMENTS.items():
        saddle = case == index
        emit(saddle & centre_high, choice[True])
        emit(saddle & ~centre_high, choice[False])
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def trace(data: np.ndarray, levels: List[float]) -> ContourGeometry:
    """Contour segments of ``data`` at ``levels``.

    Pixel (row i, column j) has its centre at scene (j + 0.5, i + 0.5).  NaN
    and infinite pixels are treated as lower than every level, so contours
    close around them instead of crossing.
    """
    stride = max(1, int(np.ceil(max(data.shape) / MAX_TRACE_SIZE)))
    sampled = data[::stride, ::stride].astype(np.float64)
    finite = np.isfinite(sampled)
    if not finite.any() or not levels or min(sampled.shape) < 2:
        return ContourGeometry(np.array([]), np.array([]), stride)
    # A finite stand-in below every level, so interpolation stays finite.
    below = min(float(sampled[finite].min()), min(levels)) - (max(levels) - min(levels) + 1.0)
    sampled = np.where(finite, sampled, below)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for level in levels:
        x, y = _trace_level(sampled, float(level))
        xs.append(x)
        ys.append(y)
    x = np.concatenate(xs) * stride + 0.5
    y = np.concatenate(ys) * stride + 0.5
    return ContourGeometry(x, y, stride)


def parse_levels(text: str) -> Tuple[List[float], int]:
    """Levels typed by the user: 'auto N' or numbers separated by spaces/commas.

    Returns (explicit levels, automatic count); exactly one is non-empty/non-zero.
    Raises ValueError for anything else, so a typo is reported, not ignored.
    """
    words = text.replace(",", " ").split()
    if not words:
        raise ValueError("no contour levels given")
    if words[0].lower() == "auto":
        count = int(words[1]) if len(words) > 1 else 5
        if not 1 <= count <= 50:
            raise ValueError("automatic level count must be 1 to 50")
        return [], count
    return sorted(float(w) for w in words), 0


# ---- levels and smoothing, with the parameters of DS9's contour dialog

METHODS = ("Block", "Smooth")  # block averaging, or a Gaussian of the smoothness as sigma


def generate_levels(lo: float, hi: float, count: int, scale: str, exponent: float = 1000.0):
    """``count`` levels from ``lo`` to ``hi`` (both included), evenly spaced in the
    stretched values of ``scale`` (Linear, Log, Power, ...): so a Log scale puts more
    levels at the faint end, as DS9 does."""
    from .scales import stretch

    if count < 1 or not hi > lo:
        return []
    if count == 1:
        return [float(lo)]
    values = np.linspace(lo, hi, 4001)
    name = "Linear" if scale == "Histogram" else scale
    positions = stretch((values - lo) / (hi - lo), name, exponent)
    return [float(np.interp(p, positions, values)) for p in np.linspace(0.0, 1.0, count)]


def histogram_levels(data: np.ndarray, lo: float, hi: float, count: int):
    """``count`` levels at evenly spaced quantiles of the pixels between lo and hi."""
    from .scales import sample

    values = sample(data)
    values = np.sort(values[(values >= lo) & (values <= hi)])
    if not values.size or count < 1:
        return []
    return [float(v) for v in np.quantile(values, np.linspace(0.0, 1.0, count))]


def smoothed(data: np.ndarray, method: str, size: int):
    """The image the contours are traced on, and the size of its pixels in image pixels:
    Block averages ``size`` x ``size`` pixels; Smooth convolves with a Gaussian of sigma
    ``size`` (NaN pixels are left out of both)."""
    size = max(int(size), 1)
    if size == 1:
        return data, 1
    finite = np.isfinite(data)
    values = np.where(finite, data, 0.0).astype(np.float64)
    if method == "Block":
        ny, nx = (data.shape[0] // size) * size, (data.shape[1] // size) * size
        shape = (ny // size, size, nx // size, size)
        total = values[:ny, :nx].reshape(shape).sum(axis=(1, 3))
        count = finite[:ny, :nx].reshape(shape).sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / count, np.nan), size
    from scipy import ndimage

    total = ndimage.gaussian_filter(values, size)
    weight = ndimage.gaussian_filter(finite.astype(np.float64), size)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight > 0.05, total / weight, np.nan), 1


def trace_smoothed(data: np.ndarray, levels, method: str = "Smooth", size: int = 1):
    """Contour segments of ``data`` after smoothing, in the viewer's coordinates."""
    image, block = smoothed(data, method, size)
    geometry = trace(image, list(levels))
    if block > 1:  # block pixels back to image pixels: block j spans j*k .. (j+1)*k
        geometry.x = (geometry.x - 0.5) * block + block / 2.0
        geometry.y = (geometry.y - 0.5) * block + block / 2.0
    return geometry
