"""Geometry of the WCS display aids: coordinate grid, compass and scale bar.

These functions only compute line coordinates and labels; the canvas draws
them.  They work in the viewer's coordinates, where array column j spans
x = j .. j + 1 and row i spans y = i .. i + 1.  The 0-based WCS pixel of a
point is therefore (x - 0.5 + isc, y - 0.5 + isr), where isc and isr are an
image origin (0 for images read from FITS files).

Nothing here invents a WCS: every function takes the buffer's real
celestial WCS, and the caller does not draw the aids when there is none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# Grid spacings considered, smallest first.  Right ascension uses time
# seconds (15 arcsec of RA each); declination uses arcseconds.
RA_STEPS_TIME_SECONDS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800]
DEC_STEPS_ARCSEC = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 18000, 36000]

# Scale-bar lengths considered, in arcseconds.
SCALE_BAR_ARCSEC = [1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200, 18000]

MAX_GRID_LINES = 7  # per coordinate
SAMPLES_PER_LINE = 120


@dataclass
class GridGeometry:
    """Grid polylines (NaN-separated) and their labels, in scene coordinates."""

    x: np.ndarray
    y: np.ndarray
    labels: List[Tuple[float, float, str, str]] = field(
        default_factory=list
    )  # x, y, text, "ra"|"dec"


def scene_to_world(wcs, x, y, isc: int, isr: int) -> Tuple[np.ndarray, np.ndarray]:
    """RA, Dec (degrees) of scene points."""
    pix = np.column_stack([np.ravel(x) - 0.5 + isc, np.ravel(y) - 0.5 + isr])
    ra, dec = wcs.all_pix2world(pix, 0).T
    return ra, dec


def world_to_scene(wcs, ra, dec, isc: int, isr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Scene coordinates of RA, Dec (degrees); NaN where the WCS cannot invert."""
    world = np.column_stack([np.ravel(ra), np.ravel(dec)])
    try:
        px, py = wcs.all_world2pix(world, 0, quiet=True).T
    except Exception:
        # Points far outside the image may not converge; fall back to the
        # non-iterative transformation, which is exact for linear WCS.
        px, py = wcs.wcs_world2pix(world, 0).T
    return px + 0.5 - isc, py + 0.5 - isr


def _wrap_ra(ra: np.ndarray, centre: float) -> np.ndarray:
    """RA unwrapped around ``centre`` into (centre - 180, centre + 180]."""
    return centre + (ra - centre + 180.0) % 360.0 - 180.0


def _choose_step(span_deg: float, steps_deg: List[float]) -> float:
    """The smallest step that gives at most MAX_GRID_LINES lines over the span."""
    for step in steps_deg:
        if span_deg / step <= MAX_GRID_LINES:
            return step
    return steps_deg[-1]


def format_ra(ra_deg: float, step_deg: float) -> str:
    """RA as h m s, with as many fields as the grid step needs."""
    total_seconds = (ra_deg % 360.0) / 15.0 * 3600.0
    total_seconds = round(total_seconds)
    hours, rest = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    hours %= 24
    if step_deg * 240.0 >= 3600:  # whole hours
        return f"{hours:02d}h"
    if step_deg * 240.0 >= 60:  # whole minutes of time
        return f"{hours:02d}h{minutes:02d}m"
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"


def format_dec(dec_deg: float, step_deg: float) -> str:
    """Dec as ±d ′ ″, with as many fields as the grid step needs."""
    sign = "-" if dec_deg < 0 else "+"
    total_arcsec = round(abs(dec_deg) * 3600.0)
    degrees, rest = divmod(total_arcsec, 3600)
    arcmin, arcsec = divmod(rest, 60)
    if step_deg * 3600.0 >= 3600:
        return f"{sign}{degrees:02d}°"
    if step_deg * 3600.0 >= 60:
        return f"{sign}{degrees:02d}°{arcmin:02d}′"
    return f"{sign}{degrees:02d}°{arcmin:02d}′{arcsec:02d}″"


def grid_geometry(wcs, shape: Tuple[int, int], isc: int, isr: int) -> GridGeometry:
    """Lines of constant RA and Dec across an image of ``shape`` (rows, columns).

    The sky range is found by sampling the image footprint; line points
    outside the image are dropped (NaN), so lines end at the image edge.
    Each line is labelled at its first point inside the image.
    """
    rows, cols = shape
    gx, gy = np.meshgrid(np.linspace(0, cols, 25), np.linspace(0, rows, 25))
    ra, dec = scene_to_world(wcs, gx, gy, isc, isr)
    ok = np.isfinite(ra) & np.isfinite(dec)
    if not ok.any():
        return GridGeometry(np.array([]), np.array([]))
    ra_centre = float(np.median(ra[ok]))
    ra = _wrap_ra(ra[ok], ra_centre)
    dec = dec[ok]
    ra_lo, ra_hi = float(ra.min()), float(ra.max())
    dec_lo, dec_hi = float(dec.min()), float(dec.max())
    # A footprint containing a celestial pole spans all right ascensions;
    # the grid then still draws the declination circles correctly.
    ra_step = _choose_step(ra_hi - ra_lo, [s * 15.0 / 3600.0 for s in RA_STEPS_TIME_SECONDS])
    dec_step = _choose_step(dec_hi - dec_lo, [s / 3600.0 for s in DEC_STEPS_ARCSEC])

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    labels: List[Tuple[float, float, str, str]] = []

    def add_line(line_ra: np.ndarray, line_dec: np.ndarray, text: str, axis: str) -> None:
        x, y = world_to_scene(wcs, line_ra, line_dec, isc, isr)
        inside = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= cols) & (y >= 0) & (y <= rows)
        if not inside.any():
            return
        x = np.where(inside, x, np.nan)
        y = np.where(inside, y, np.nan)
        xs.append(np.append(x, np.nan))
        ys.append(np.append(y, np.nan))
        first = int(np.flatnonzero(inside)[0])
        labels.append((float(x[first]), float(y[first]), text, axis))

    dec_samples = np.linspace(dec_lo, dec_hi, SAMPLES_PER_LINE)
    for value in np.arange(math.ceil(ra_lo / ra_step) * ra_step, ra_hi + 1e-9, ra_step):
        add_line(np.full_like(dec_samples, value), dec_samples, format_ra(value, ra_step), "ra")
    ra_samples = np.linspace(ra_lo, ra_hi, SAMPLES_PER_LINE)
    for value in np.arange(math.ceil(dec_lo / dec_step) * dec_step, dec_hi + 1e-9, dec_step):
        add_line(ra_samples, np.full_like(ra_samples, value), format_dec(value, dec_step), "dec")

    if not xs:
        return GridGeometry(np.array([]), np.array([]))
    return GridGeometry(np.concatenate(xs), np.concatenate(ys), labels)


def sky_axes(wcs, x: float, y: float, isc: int, isr: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Unit vectors of north and east at a scene point, and the pixel scale (arcsec/pixel).

    The directions come from the WCS itself (a 1″ step in Dec and in RA), so
    flipped or rotated images give the correct compass.  The scale is the
    geometric mean of the local x and y pixel sizes.
    """
    ra, dec = scene_to_world(wcs, [x], [y], isc, isr)
    ra, dec = float(ra[0]), float(dec[0])
    step = 1.0 / 3600.0
    nx, ny = world_to_scene(wcs, [ra], [dec + step], isc, isr)
    ex, ey = world_to_scene(
        wcs, [ra + step / max(math.cos(math.radians(dec)), 1e-6)], [dec], isc, isr
    )
    north = np.array([nx[0] - x, ny[0] - y])
    east = np.array([ex[0] - x, ey[0] - y])
    pixels_per_arcsec = math.sqrt(abs(north[0] * east[1] - north[1] * east[0]))
    scale = 1.0 / pixels_per_arcsec if pixels_per_arcsec > 0 else float("nan")
    return north / np.hypot(*north), east / np.hypot(*east), scale


def north_up(north, east) -> Tuple[float, bool]:
    """Display rotation (degrees, counter-clockwise) and mirror that put North up and
    East to the left, from the North and East directions on the image (sky_axes).

    The mirror (left-right flip, applied before the rotation) is needed only when the
    image has the other handedness, as DS9 does for "North up, East left"."""
    for flip in (False, True):
        sign = -1.0 if flip else 1.0
        nx, ny = sign * north[0], north[1]
        ex, ey = sign * east[0], east[1]
        theta = 90.0 - math.degrees(math.atan2(ny, nx))
        t = math.radians(theta)
        east_x = ex * math.cos(t) - ey * math.sin(t)
        if east_x < 0:
            return (theta + 180.0) % 360.0 - 180.0, flip
    return 0.0, False  # degenerate axes: leave the display as it is


def scale_bar_length(scale_arcsec: float, max_pixels: float) -> Tuple[float, str]:
    """The longest standard bar no longer than ``max_pixels``: (length in pixels, label)."""
    best = SCALE_BAR_ARCSEC[0]
    for arcsec in SCALE_BAR_ARCSEC:
        if arcsec / scale_arcsec <= max_pixels:
            best = arcsec
    if best >= 3600:
        label = f"{best // 3600}°"
    elif best >= 60:
        label = f"{best // 60}′"
    else:
        label = f"{best}″"
    return best / scale_arcsec, label
