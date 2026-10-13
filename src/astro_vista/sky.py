"""The sky level of an image, measured where there are no sources.

Every source is found down to a low threshold (0.5 sigma by default) and masked, with its
pixels grown a little; the sky is measured on what is left (the "support"):

- the level: the median of the support pixels, with a clipped mean and sigma;
- cells: the frame cut into CELLS x CELLS boxes, the median of each box with enough
  support;
- the gradient: a plane fitted to the cell medians;
- two uncertainties, kept apart: an approximate sampling error of the level, and the
  spatial variation of the cells;
- a verdict on the support (``no_estimate``, ``candidate_inspect_support``,
  ``usable_support``).  Enough support is necessary, not sufficient: no verdict proves that
  a smooth halo, too faint to be detected, is absent.  The verdict advises; the level is
  kept whenever there are support pixels, and the user decides whether to use it.

After a fit, :func:`residual_level` measures the residual (image - sky - models) on the same
support; its median m gives the next sky factor, f' = f + m / base (the sky is f x base), as
in the manual practice of refining the sky by its residual histogram.

The detection follows one path: a level and a noise (iterated, see measure), then
SEP on the image minus that level, with that sigma as the noise (so the threshold is
THRESH x sigma above the level), the default 3 x 3 kernel, 32 deblending thresholds,
contrast 0.005, cleaning on.  All of it is recorded in the result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import masks as maskops

DEFAULTS = {"THRESH": 0.5, "MINAREA": 12, "GROW": 2.0, "CELLS": 8, "FWHM": 0.0}
# what makes a cell usable: this fraction of its pixels in the support, and this many
CELL_FRACTION = 0.30
CELL_PIXELS = 200
STATES = ("no_estimate", "candidate_inspect_support", "usable_support")


@dataclass
class Cell:
    i: int  # column of the cell (0 .. CELLS-1)
    j: int  # row
    x: float  # centre, native coordinates
    y: float
    x0: int  # pixel edges: data[y0:y1, x0:x1]
    x1: int
    y0: int
    y1: int
    support: int  # support pixels in it
    fraction: float
    median: float = float("nan")
    usable: bool = False
    rejected: bool = False  # left out of the plane fit (beyond 3 MAD)


@dataclass
class SkyResult:
    image_name: str
    image_id: str  # checksum of the pixels measured
    settings: Dict[str, object]
    detection: Dict[str, object]  # everything the detection used
    exclusion: np.ndarray  # True: left out of the sky (sources grown, invalid, layers)
    support: np.ndarray  # True: used for the sky
    valid: np.ndarray  # True: a finite pixel that is not padding
    cells: List[Cell]
    level: float = float("nan")  # median of the support
    mean: float = float("nan")  # 3-sigma clipped mean
    sigma: float = float("nan")  # clipped standard deviation (MAD based)
    first_level: float = float("nan")
    first_sigma: float = float("nan")
    sources: int = 0
    plane: Optional[Dict[str, float]] = None  # s0, gx, gy (per pixel), xc, yc
    sampling_error: float = float("nan")
    cell_range: float = float("nan")
    cell_scatter: float = float("nan")
    plane_change: float = float("nan")  # peak to peak over the frame (corners)
    state: str = "no_estimate"
    reasons: List[str] = field(default_factory=list)
    id: str = ""

    @property
    def support_fraction(self) -> float:
        n = int(self.valid.sum())
        return float(self.support.sum()) / n if n else 0.0

    @property
    def usable_cells(self) -> List[Cell]:
        return [c for c in self.cells if c.usable]

    def gradient(self) -> Optional[Dict[str, float]]:
        """|g| and gx, gy in counts per 1000 pixels, and its direction as an image-axis
        angle (degrees from +x towards +y; not a position angle on the sky)."""
        if self.plane is None:
            return None
        gx, gy = self.plane["gx"] * 1000, self.plane["gy"] * 1000
        return {
            "gx": gx,
            "gy": gy,
            "size": float(np.hypot(gx, gy)),
            "angle": float(np.degrees(np.arctan2(gy, gx)) % 360),
        }

    def summary(self) -> str:
        """The report, as the terminal shows it."""
        lines = []
        if not np.isfinite(self.level):
            lines.append(f"{self.image_name}: no sky estimate ({self.state})")
        else:
            lines.append(
                f"{self.image_name}: sky {self.level:.6g} (median; clipped mean {self.mean:.6g}, "
                f"sigma {self.sigma:.4g}) [{self.state}]"
            )
        if self.state != "no_estimate":
            lines.append(
                f"  uncertainty: sampling ±{self.sampling_error:.3g} (approximate: assumes "
                f"independent cells, leaves out any halo bias); spatial: cell range "
                f"{self.cell_range:.3g}, scatter {self.cell_scatter:.3g}"
            )
            g = self.gradient()
            if g is not None:
                text = (
                    f"  gradient {g['size']:.3g} per 1000 px toward {g['angle']:.0f} deg "
                    f"(image-axis angle from +x); {self.plane_change:.3g} across the frame "
                    f"= {self.plane_change / self.sigma:.2f} sigma"
                    if self.sigma > 0
                    else f"  gradient {g['size']:.3g} per 1000 px toward {g['angle']:.0f} deg"
                )
                if abs(self.plane["s0"]) > 10 * self.sigma > 0:
                    text += f" = {100 * self.plane_change / abs(self.plane['s0']):.2f} % of the sky"
                lines.append(text)
        usable = len(self.usable_cells)
        lines.append(
            f"  {100 * self.support_fraction:.1f} % of the valid pixels used; {usable} of "
            f"{len(self.cells)} cells usable; {self.sources} sources masked "
            f"({self.settings['THRESH']:g} sigma, grow {self.settings['GROW']:g}); "
            f"measurement {self.id}"
        )
        for reason in self.reasons:
            lines.append(f"  - {reason}")
        return "\n".join(lines)

    def report(self) -> dict:
        """Everything but the pixel masks, for export (sky.json)."""
        out = {
            "id": self.id,
            "image": self.image_name,
            "image_id": self.image_id,
            "state": self.state,
            "reasons": self.reasons,
            "level_median": self.level,
            "clipped_mean": self.mean,
            "clipped_sigma": self.sigma,
            "first_level": self.first_level,
            "first_sigma": self.first_sigma,
            "sampling_error_approximate": self.sampling_error,
            "cell_range": self.cell_range,
            "cell_scatter": self.cell_scatter,
            "plane": self.plane,
            "gradient": self.gradient(),
            "plane_change_peak_to_peak": self.plane_change,
            "support_pixels": int(self.support.sum()),
            "valid_pixels": int(self.valid.sum()),
            "sources": self.sources,
            "settings": self.settings,
            "detection": self.detection,
            "cells": [{k: v for k, v in c.__dict__.items()} for c in self.cells],
        }
        return json.loads(json.dumps(out, default=float))


def checksum(data: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(data).tobytes()).hexdigest()[:12]


def clipped(values: np.ndarray):
    """A 3-sigma clipped median, mean and (MAD-based) sigma."""
    from astropy.stats import sigma_clipped_stats

    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean, median, std = sigma_clipped_stats(values, sigma=3.0, maxiters=10, stdfunc="mad_std")
    return float(median), float(mean), float(std)


def local_noise(data: np.ndarray, good: np.ndarray, mesh: int = 32) -> float:
    """The pixel noise: the MAD of the image minus a SEP background mesh of ``mesh`` pixels,
    on the ``good`` pixels.  The mesh takes out smooth light (a halo, a gradient) and keeps
    noise that is correlated between neighbouring pixels (as in drizzled images)."""
    import sep

    if good.sum() < 50:
        return float("nan")
    filled = np.where(good, data, np.median(data[good])).astype(np.float64)
    back = sep.Background(np.ascontiguousarray(filled), mask=~good, bw=mesh, bh=mesh, fw=3, fh=3)
    return mad((filled - back.back())[good])


def mad(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if values.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def make_cells(shape, n: int) -> List[Cell]:
    ny, nx = shape
    xs = np.linspace(0, nx, n + 1).round().astype(int)
    ys = np.linspace(0, ny, n + 1).round().astype(int)
    cells = []
    for j in range(n):
        for i in range(n):
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            if x1 <= x0 or y1 <= y0:
                continue
            cells.append(Cell(i, j, (x0 + x1) / 2, (y0 + y1) / 2, x0, x1, y0, y1, 0, 0.0))
    return cells


def cell_statistics(data, support, cells: List[Cell]) -> None:
    """The median of each cell's support pixels (in place); usable cells marked."""
    for c in cells:
        box = support[c.y0 : c.y1, c.x0 : c.x1]
        c.support = int(box.sum())
        c.fraction = c.support / box.size
        c.usable = c.fraction >= CELL_FRACTION and c.support >= CELL_PIXELS
        c.median = (
            float(np.median(data[c.y0 : c.y1, c.x0 : c.x1][box])) if c.support else float("nan")
        )
        c.rejected = False


def fit_plane(cells: List[Cell], shape) -> Optional[Dict[str, float]]:
    """A plane through the usable cells' medians, by least squares, with one pass leaving
    out cells beyond 3 MAD of it; None with fewer than 3 cells or a degenerate layout."""
    ny, nx = shape
    xc, yc = nx / 2.0, ny / 2.0
    use = [c for c in cells if c.usable]
    if len(use) < 3:
        return None

    def solve(group):
        a = np.array([[1.0, c.x - xc, c.y - yc] for c in group])
        m = np.array([c.median for c in group])
        if np.linalg.matrix_rank(a) < 3:
            return None
        coef, *_ = np.linalg.lstsq(a, m, rcond=None)
        return coef

    coef = solve(use)
    if coef is None:
        return None
    resid = np.array(
        [c.median - (coef[0] + coef[1] * (c.x - xc) + coef[2] * (c.y - yc)) for c in use]
    )
    spread = mad(resid)
    if spread > 0:
        keep = [c for c, r in zip(use, resid) if abs(r) <= 3 * spread]
        for c, r in zip(use, resid):
            c.rejected = abs(r) > 3 * spread
        if len(keep) >= 3 and len(keep) < len(use):
            again = solve(keep)
            if again is not None:
                coef = again
    return {"s0": float(coef[0]), "gx": float(coef[1]), "gy": float(coef[2]), "xc": xc, "yc": yc}


def plane_at(plane, x, y):
    return (
        plane["s0"]
        + plane["gx"] * (np.asarray(x) - plane["xc"])
        + plane["gy"] * (np.asarray(y) - plane["yc"])
    )


def plane_image(plane, shape) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    return plane_at(plane, xx + 0.5, yy + 0.5)


def measure(
    data: np.ndarray,
    name: str = "",
    exclude: Optional[np.ndarray] = None,
    thresh: float = DEFAULTS["THRESH"],
    minarea: int = DEFAULTS["MINAREA"],
    grow: float = DEFAULTS["GROW"],
    cells: int = DEFAULTS["CELLS"],
    pad: Optional[float] = None,
    layers: str = "",
    back_size: int = 0,
    fwhm: float = 0.0,
) -> SkyResult:
    """Measure the sky of ``data`` (see the module's description).  ``exclude`` (True =
    left out) adds the user's own exclusions; ``pad`` is a value that marks pixels with no
    data (such as 0 around a mosaic).  ``back_size`` > 0 detects instead on the image minus
    a SEP background mesh of that size, with the mesh's global RMS as the noise, in one
    pass (as SExtractor does with BACK_SIZE): a mesh takes part of a large halo for
    background, so fewer of its pixels are masked.  ``fwhm`` > 0 filters the detection with
    a Gaussian of that FWHM (pixels) instead of SEP's default 3 x 3 kernel."""
    import sep

    if thresh <= 0 or minarea < 1 or grow < 0 or cells < 2 or fwhm < 0:
        raise ValueError("THRESH > 0, MINAREA >= 1, GROW >= 0, CELLS >= 2 and FWHM >= 0 are needed")
    data = np.asarray(data)
    shape = data.shape
    valid = np.isfinite(data)
    if pad is not None:
        valid &= data != pad
    settings = {
        "THRESH": float(thresh),
        "MINAREA": int(minarea),
        "GROW": float(grow),
        "CELLS": int(cells),
        "PAD": pad,
        "MASK": layers,
        "BACK_SIZE": int(back_size),
        "FWHM": float(fwhm),
    }
    first_level, _mean, _sigma = clipped(data[valid])
    first_sigma = local_noise(data, valid)
    detection = {
        "engine": f"SEP {sep.__version__}",
        "background": "scalar, iterated: first the 3-sigma clipped median of all valid "
        "pixels, then the median of the support left by the previous detection, until it "
        "changes by less than 0.02 sigma (at most 6 passes)",
        "noise": "scalar: the MAD of the image minus a 32-pixel SEP background mesh (smooth "
        "light taken out, correlated noise kept), of all valid pixels, once (on the support it "
        "would be biased low: the support has lost the upper tail of the noise)",
        "threshold": f"{thresh:g} x sigma above the level (err = sigma)",
        "minarea": int(minarea),
        "filter": (
            f"Gaussian, FWHM {fwhm:g} px, {gaussian_kernel(fwhm).shape[0]} x "
            f"{gaussian_kernel(fwhm).shape[0]}, matched"
            if fwhm > 0
            else "SEP default 3x3 kernel, matched"
        ),
        "deblending": "none (deblend_cont=1): only the union of the segments is used",
        "clean": False,
        "grow": f"a disc of {grow:g} px around every segment (0: none)",
    }
    result = SkyResult(
        image_name=name,
        image_id=checksum(data),
        settings=settings,
        detection=detection,
        exclusion=np.ones(shape, bool),
        support=np.zeros(shape, bool),
        valid=valid,
        cells=make_cells(shape, cells),
        first_level=first_level,
        first_sigma=first_sigma,
    )
    if not np.isfinite(first_sigma) or first_sigma <= 0 or valid.sum() < 100:
        result.reasons.append("too few valid pixels, or no noise, to detect sources")
        result.id = _identity(result)
        return result
    blocked = ~valid
    if exclude is not None:
        blocked = blocked | exclude
    sep.set_extract_pixstack(max(300000, data.size))
    sep.set_sub_object_limit(4096)
    level, noise = first_level, first_sigma
    passes = []
    converged = False
    background = None
    if back_size > 0:
        filled = np.ascontiguousarray(np.where(valid, data, first_level), dtype=np.float64)
        mesh = sep.Background(
            filled, mask=blocked, bw=int(back_size), bh=int(back_size), fw=3, fh=3
        )
        background = mesh.back()
        noise = float(mesh.globalrms)
        detection["background"] = f"SEP mesh {int(back_size)} px, filter 3 x 3 (one pass)"
        detection["noise"] = "scalar: the mesh's global RMS"
    for _ in range(1 if back_size > 0 else 6):
        base = background if background is not None else level
        work = np.ascontiguousarray(np.where(valid, data - base, 0.0), dtype=np.float64)
        try:
            objects, segmentation = sep.extract(
                work,
                float(thresh),
                err=float(noise),
                mask=blocked,
                minarea=int(minarea),
                filter_kernel=gaussian_kernel(fwhm) if fwhm > 0 else sep.default_kernel,
                filter_type="matched",
                deblend_nthresh=1,
                deblend_cont=1.0,
                clean=False,
                segmentation_map=True,
            )
        except Exception as e:  # SEP's limits (too many pixels in one object, ...)
            raise RuntimeError(f"the source detection failed: {e}")
        support = valid & ~blocked & (segmentation == 0)
        passes.append({"level": float(level), "sigma": float(noise), "sources": len(objects)})
        if support.sum() < 100 or background is not None:
            converged = background is not None
            break
        new_level = float(np.median(data[support]))
        done = abs(new_level - level) < 0.02 * noise
        level = new_level
        if done:
            converged = True
            break
    detection["passes"] = passes
    detection["converged"] = converged
    sources = segmentation > 0
    sources = maskops.grow(sources, grow) if grow > 0 else sources.copy()
    exclusion = sources | ~valid
    if exclude is not None:
        exclusion |= exclude
    support = valid & ~exclusion
    result.exclusion, result.support, result.sources = exclusion, support, int(len(objects))
    values = data[support]
    if values.size:
        result.level, result.mean, result.sigma = clipped(values)
        result.level = float(np.median(values))
    cell_statistics(data, support, result.cells)
    result.plane = fit_plane(result.cells, shape)
    _judge(result, shape)
    result.id = _identity(result)
    return result


def gaussian_kernel(fwhm: float) -> np.ndarray:
    """A normalised Gaussian detection kernel of ``fwhm`` pixels, 2 ceil(fwhm) + 1 wide (5 x 5
    for FWHM 2, as SExtractor's gauss_2.0_5x5)."""
    half = max(1, int(np.ceil(fwhm)))
    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    s = fwhm / 2.3548
    k = np.exp(-(x * x + y * y) / (2 * s * s))
    return np.ascontiguousarray(k / k.sum(), dtype=np.float32)


def residual_level(
    residual: np.ndarray, support: np.ndarray, base: float = float("nan"), factor: float = 1.0
) -> Dict[str, float]:
    """The residual (image - sky - models) on the sky support: its 3-sigma clipped median,
    mean and sigma, and, with the base level and the factor of the sky used (sky = factor x
    base), the next factor f + median / base and the sky it gives."""
    use = support & np.isfinite(residual)
    values = residual[use]
    out = {"pixels": int(values.size)}
    if values.size < 10:
        return out
    median, mean, sigma = clipped(values)
    out.update(median=median, mean=mean, sigma=sigma, factor=float(factor))
    if np.isfinite(base) and base != 0:
        out["base"] = float(base)
        out["next_factor"] = float(factor + median / base)
        out["next_sky"] = float(factor * base + median)
    return out


def _identity(result: SkyResult) -> str:
    h = hashlib.sha1()
    h.update(result.image_id.encode())
    h.update(np.packbits(result.exclusion).tobytes())
    h.update(json.dumps(result.settings, sort_keys=True, default=str).encode())
    return h.hexdigest()[:10]


def _judge(result: SkyResult, shape) -> None:
    """The uncertainties and the verdict on the support."""
    ny, nx = shape
    usable = result.usable_cells
    n_valid = int(result.valid.sum())
    fraction = result.support.sum() / n_valid if n_valid else 0.0
    reasons = result.reasons
    if fraction < 0.005 or len(usable) < 4:
        result.state = "no_estimate"
        reasons.append(
            f"{100 * fraction:.2f} % of the valid pixels and {len(usable)} usable cells "
            "left (at least 0.5 % and 4 are needed)"
        )
        if result.support.sum() < 100:
            result.level = result.mean = float("nan")
        return
    medians = np.array([c.median for c in usable])
    if result.plane is not None:
        fitted = plane_at(result.plane, [c.x for c in usable], [c.y for c in usable])
        resid = medians - fitted
        corners = plane_at(result.plane, [0, nx, 0, nx], [0, 0, ny, ny])
        result.plane_change = float(np.ptp(corners))
    else:
        resid = medians - np.median(medians)
        result.plane_change = float("nan")
    result.sampling_error = float(1.2533 * mad(resid) / np.sqrt(len(usable)))
    result.cell_range = float(np.ptp(medians))
    result.cell_scatter = mad(medians)
    problems = []
    passes = result.detection.get("passes", [])
    if not result.detection.get("converged", True) and len(passes) >= 2:
        step = (passes[-2]["level"] - passes[-1]["level"]) / passes[-1]["sigma"]
        problems.append(
            f"the level was still changing by {step:+.2f} sigma per pass after {len(passes)} "
            "passes: diffuse light (a halo) probably remains in the support"
        )
    if fraction < 0.05:
        problems.append(f"only {100 * fraction:.1f} % of the valid pixels are sky support")
    if len(usable) < 12:
        problems.append(f"only {len(usable)} usable cells (12 wanted)")
    x_extent = (max(c.x1 for c in usable) - min(c.x0 for c in usable)) / nx
    y_extent = (max(c.y1 for c in usable) - min(c.y0 for c in usable)) / ny
    if x_extent < 0.5 or y_extent < 0.5:
        problems.append(
            f"the usable cells span {100 * x_extent:.0f} % of the width and "
            f"{100 * y_extent:.0f} % of the height"
        )
    xy = np.array([[c.x, c.y] for c in usable])
    eig = np.linalg.eigvalsh(np.cov(xy.T)) if len(usable) > 2 else np.array([0.0, 1.0])
    if eig[-1] <= 0 or eig[0] < 0.05 * eig[-1]:
        problems.append("the usable cells lie nearly on a line: the plane is poorly constrained")
    if result.plane is None:
        problems.append("no plane could be fitted to the cells")
    # cells that disagree with the plane by more than their own noise allows
    expected = np.array([3 * 1.2533 * result.sigma / np.sqrt(max(c.support, 1)) for c in usable])
    disagree = np.abs(resid) > expected
    if disagree.mean() > 0.20:
        problems.append(
            f"{int(disagree.sum())} of {len(usable)} cells differ from the plane by more "
            "than 3 times their own noise"
        )
    if np.isfinite(result.plane_change) and result.plane_change > result.sigma:
        problems.append(
            f"the plane changes by {result.plane_change:.3g} across the frame, more than "
            f"1 sigma ({result.sigma:.3g})"
        )
    reasons.extend(problems)
    result.state = "candidate_inspect_support" if problems else "usable_support"


def check(result: SkyResult, residual: np.ndarray) -> Dict[str, float]:
    """The same statistics on ``residual`` (an image with the galaxies taken out), on the
    pixels accepted by the frozen sky mask and with the same cells."""
    support = result.support & np.isfinite(residual)
    values = residual[support]
    out = {"pixels": int(values.size)}
    if not values.size:
        return out
    out["median"] = float(np.median(values))
    cells = [Cell(**{k: v for k, v in c.__dict__.items()}) for c in result.cells]
    cell_statistics(residual, support, cells)
    usable = [c for c in cells if c.usable]
    if usable:
        m = np.array([c.median for c in usable])
        out["cells"] = len(usable)
        out["sampling_error"] = float(1.2533 * mad(m) / np.sqrt(len(usable)))
        out["cell_range"] = float(np.ptp(m))
    return out
