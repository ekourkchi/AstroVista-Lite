"""Mask layers: named pixel masks drawn over the image and used by the isophote fit.

A mask is True where a pixel is masked (left out).  On disk, masks follow elliprof's
convention: 0 is masked and any other value is good, in FITS files and in legacy BITPIX = 1
``.dmask`` bitmaps alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ROLES = {
    "fit": "fit-exclusion",
    "sky": "sky-exclusion",
    "candidate": "fit-exclusion candidate",
    "": "no role",
}
# colours given to new masks in turn
COLOURS = ("#ff5050", "#ffb000", "#c060ff", "#40c0ff", "#60e060", "#ff70c0")


@dataclass
class Mask:
    name: str
    masked: np.ndarray  # bool, True = masked
    colour: str = COLOURS[0]
    visible: bool = True
    opacity: float = 0.45
    source: str = ""  # how it was made (shown as a tooltip)
    # what the layer is for: "fit" (left out of fits), "sky" (left out of the sky only),
    # "candidate" (proposed, not used until promoted to "fit"), or "" (not said: used
    # wherever it is named)
    role: str = ""

    @property
    def shape(self):
        return self.masked.shape

    def count(self) -> int:
        return int(self.masked.sum())


def circle(masked: np.ndarray, x: float, y: float, r: float, value: bool = True) -> None:
    """Set the pixels whose centres lie within ``r`` of (x, y), viewer coordinates."""
    ny, nx = masked.shape
    i0, i1 = max(int(np.floor(x - r)), 0), min(int(np.ceil(x + r)) + 1, nx)
    j0, j1 = max(int(np.floor(y - r)), 0), min(int(np.ceil(y + r)) + 1, ny)
    if i0 >= i1 or j0 >= j1:
        return
    jj, ii = np.mgrid[j0:j1, i0:i1]
    inside = (ii + 0.5 - x) ** 2 + (jj + 0.5 - y) ** 2 <= r * r
    masked[j0:j1, i0:i1][inside] = value


def box(masked: np.ndarray, x0: float, y0: float, x1: float, y1: float, value=True) -> None:
    """Set the pixels whose centres lie inside the box, viewer coordinates."""
    ny, nx = masked.shape
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    i0, i1 = max(int(np.ceil(xa - 0.5)), 0), min(int(np.floor(xb - 0.5)) + 1, nx)
    j0, j1 = max(int(np.ceil(ya - 0.5)), 0), min(int(np.floor(yb - 0.5)) + 1, ny)
    if i0 < i1 and j0 < j1:
        masked[j0:j1, i0:i1] = value


def load(path, shape=None) -> np.ndarray:
    """Read a mask file (FITS or .dmask): True where it masks a pixel."""
    from elliprof import load_mask

    values = np.asarray(load_mask(str(Path(path).expanduser())))
    if shape is not None and values.shape != tuple(shape):
        raise ValueError(
            f"the mask {Path(path).name} is {values.shape[1]} x {values.shape[0]} pixels, "
            f"the image {shape[1]} x {shape[0]}"
        )
    return (values == 0) | ~np.isfinite(values)


def save(mask: Mask, path) -> Path:
    """Write a mask: a legacy bitmap for ``.dmask``, otherwise FITS (1 = good, 0 = masked)."""
    path = Path(path).expanduser()
    good = ~mask.masked
    if path.suffix.lower() == ".dmask":
        from elliprof.masks import write_bitmap_mask

        write_bitmap_mask(str(path), good.astype(np.uint8))
    else:
        from astropy.io import fits

        header = fits.Header()
        header["MASKNAME"] = mask.name
        header["COMMENT"] = "1 = good, 0 = masked"
        fits.PrimaryHDU(good.astype(np.uint8), header=header).writeto(path, overwrite=True)
    return path


def union(masks) -> Optional[np.ndarray]:
    """The pixels masked by any of ``masks`` (None when there are none)."""
    out = None
    for m in masks:
        out = m.masked.copy() if out is None else out | m.masked
    return out


def ellipse(masked, x, y, a, b, angle, value=True) -> None:
    """Set the pixels whose centres lie inside an ellipse (semi-axes a, b; the a axis at
    ``angle`` degrees counter-clockwise from +x), viewer coordinates."""
    ny, nx = masked.shape
    r = max(a, b)
    i0, i1 = max(int(np.floor(x - r)), 0), min(int(np.ceil(x + r)) + 1, nx)
    j0, j1 = max(int(np.floor(y - r)), 0), min(int(np.ceil(y + r)) + 1, ny)
    if i0 >= i1 or j0 >= j1 or a <= 0 or b <= 0:
        return
    jj, ii = np.mgrid[j0:j1, i0:i1]
    dx, dy = ii + 0.5 - x, jj + 0.5 - y
    t = np.radians(angle)
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    masked[j0:j1, i0:i1][(u / a) ** 2 + (v / b) ** 2 <= 1.0] = value


def _inside_polygon(px, py, xs, ys) -> np.ndarray:
    """Even-odd rule: True for the points (px, py) inside the polygon (xs, ys)."""
    inside = np.zeros(px.shape, bool)
    n = len(xs)
    for k in range(n):
        x1, y1 = xs[k], ys[k]
        x2, y2 = xs[(k + 1) % n], ys[(k + 1) % n]
        crosses = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xcross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (px < xcross)
    return inside


def polygon(masked, points, value=True) -> None:
    """Set the pixels whose centres lie inside the polygon ``points`` [(x, y), ...]."""
    if len(points) < 3:
        return
    xs = np.array([p[0] for p in points], float)
    ys = np.array([p[1] for p in points], float)
    ny, nx = masked.shape
    i0, i1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())) + 1, nx)
    j0, j1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())) + 1, ny)
    if i0 >= i1 or j0 >= j1:
        return
    jj, ii = np.mgrid[j0:j1, i0:i1]
    inside = _inside_polygon(ii + 0.5, jj + 0.5, xs, ys)
    masked[j0:j1, i0:i1][inside] = value


def brush(masked, points, r, value=True) -> None:
    """Paint circles of radius ``r`` along the path ``points``, close enough together to
    leave no gaps."""
    if not points:
        return
    path = [points[0]]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        steps = max(int(np.hypot(x1 - x0, y1 - y0) / max(r / 2.0, 0.5)), 1)
        for k in range(1, steps + 1):
            path.append((x0 + (x1 - x0) * k / steps, y0 + (y1 - y0) * k / steps))
    for x, y in path:
        circle(masked, x, y, r, value)


def grow(masked: np.ndarray, r: float) -> np.ndarray:
    """``masked`` grown by ``r`` pixels (every pixel within r of a masked one)."""
    from scipy import ndimage

    if r <= 0 or not masked.any():
        return masked.copy()
    return ndimage.distance_transform_edt(~masked) <= r


def shrink(masked: np.ndarray, r: float) -> np.ndarray:
    """``masked`` shrunk by ``r`` pixels."""
    return ~grow(~masked, r)


def patch_at(masked: np.ndarray, x: float, y: float) -> Optional[np.ndarray]:
    """The connected masked area (8-connected) under the point, or None."""
    from scipy import ndimage

    i, j = int(np.floor(x)), int(np.floor(y))
    ny, nx = masked.shape
    if not (0 <= i < nx and 0 <= j < ny) or not masked[j, i]:
        return None
    labels, _count = ndimage.label(masked, structure=np.ones((3, 3), bool))
    return labels == labels[j, i]


def where(columns, text: str) -> np.ndarray:
    """The rows of a catalogue that satisfy ``text``: comparisons of a column with a
    number joined by 'and' / 'or', e.g. ``MAG_AUTO < 24 and FWHM_IMAGE < 3``."""
    import re

    n = len(next(iter(columns.values())))
    result, joiner = None, "and"
    for part in re.split(r"\s+(and|or)\s+", text.strip(), flags=re.I):
        if part.lower() in ("and", "or"):
            joiner = part.lower()
            continue
        m = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|==|!=|<|>|=)\s*(\S+)\s*", part)
        if not m:
            raise ValueError(f"cannot read the condition {part!r} (e.g. MAG_AUTO < 24)")
        name, op, value = m.group(1).upper(), m.group(2), float(m.group(3))
        if name not in columns:
            raise ValueError(f"the catalogue has no column {name}")
        c = np.asarray(columns[name], float)
        ops = {
            "<": c < value,
            ">": c > value,
            "<=": c <= value,
            ">=": c >= value,
            "==": c == value,
            "=": c == value,
            "!=": c != value,
        }
        rows = ops[op]
        result = rows if result is None else (result & rows if joiner == "and" else result | rows)
    return result if result is not None else np.ones(n, bool)
