"""Source detection with SEP (K. Barbary and contributors): the detection and photometry
algorithms of SExtractor as a Python library, installed with this package.

A run gives the catalogue columns SExtractor gives (as far as SEP measures them; there is
no star/galaxy classifier) and a segmentation map.  The settings keep SExtractor's
keyword names (DETECT_THRESH, DETECT_MINAREA, ...), as astronomers know them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .images import Image

# the catalogue columns asked for
COLUMNS = [
    "NUMBER",
    "X_IMAGE",
    "Y_IMAGE",
    "A_IMAGE",
    "B_IMAGE",
    "THETA_IMAGE",
    "ELLIPTICITY",
    "KRON_RADIUS",
    "FLUX_RADIUS",
    "FWHM_IMAGE",
    "ISOAREA_IMAGE",
    "FLUX_AUTO",
    "FLUXERR_AUTO",
    "MAG_AUTO",
    "MAGERR_AUTO",
    "BACKGROUND",
    "FLAGS",
    "ALPHA_J2000",
    "DELTA_J2000",
]

# the settings shown in the panel: keyword, label, default, type, explanation
# The settings of a run, in SExtractor's keyword names: (keyword, label, default, kind,
# explanation, group).  kind: float, int, a list of choices, or str (free text).
SETTINGS = [
    (
        "THRESH_TYPE",
        "Threshold type",
        "RELATIVE",
        ["RELATIVE", "ABSOLUTE"],
        "RELATIVE: in standard deviations of the background; ABSOLUTE: in the image's units",
        "Detection",
    ),
    (
        "DETECT_THRESH",
        "Detection threshold",
        3.0,
        float,
        "how far above the background a pixel must be to belong to a source",
        "Detection",
    ),
    ("DETECT_MINAREA", "Minimum area", 5, int, "connected pixels above the threshold", "Detection"),
    (
        "FILTER",
        "Filter",
        "Y",
        ["Y", "N"],
        "smooth the image before detection (finds faint sources better)",
        "Filtering",
    ),
    (
        "FILTER_FWHM",
        "Filter FWHM",
        0.0,
        float,
        "pixels: a Gaussian of this FWHM; 0 for the default 3 x 3 kernel",
        "Filtering",
    ),
    (
        "FILTER_TYPE",
        "Filter type",
        "matched",
        ["matched", "conv"],
        "matched: a matched filter, which accounts for the noise; conv: a plain convolution",
        "Filtering",
    ),
    (
        "DEBLEND_NTHRESH",
        "Deblending levels",
        32,
        int,
        "sub-thresholds used to split blended objects",
        "Deblending",
    ),
    (
        "DEBLEND_MINCONT",
        "Deblending contrast",
        0.005,
        float,
        "the faintest part, as a fraction of the total, kept as a source (0 splits the most)",
        "Deblending",
    ),
    (
        "CLEAN",
        "Clean",
        "Y",
        ["Y", "N"],
        "remove spurious detections in the wings of bright objects",
        "Deblending",
    ),
    ("CLEAN_PARAM", "Cleaning efficiency", 1.0, float, "larger cleans more", "Deblending"),
    (
        "BACK_TYPE",
        "Background",
        "AUTO",
        ["AUTO", "MANUAL"],
        "AUTO: mapped from the image; MANUAL: the constant BACK_VALUE",
        "Background",
    ),
    (
        "BACK_VALUE",
        "Background value",
        0.0,
        float,
        "the constant background (MANUAL)",
        "Background",
    ),
    (
        "BACK_SIZE",
        "Background mesh",
        64,
        int,
        "pixels on a side of each background cell",
        "Background",
    ),
    (
        "BACK_FILTERSIZE",
        "Background filter",
        3,
        int,
        "cells in the median filter of the background map",
        "Background",
    ),
    (
        "BACK_FILTTHRESH",
        "Filter threshold",
        0.0,
        float,
        "only cells differing from their neighbours by more than this are filtered",
        "Background",
    ),
    (
        "PHOT_APERTURES",
        "Apertures",
        "5",
        str,
        "aperture diameters in pixels, comma separated: MAG_APER, MAG_APER_2, ...",
        "Photometry",
    ),
    (
        "PHOT_AUTOPARAMS",
        "Kron factor, min. radius",
        "2.5,3.5",
        str,
        "MAG_AUTO's ellipse: the Kron factor and the smallest radius (in A, B)",
        "Photometry",
    ),
    (
        "PHOT_FLUXFRAC",
        "Flux fraction",
        0.5,
        float,
        "the fraction of the light within FLUX_RADIUS (0.5: the half-light radius)",
        "Photometry",
    ),
    ("MAG_ZEROPOINT", "Zero point", 0.0, float, "the magnitude of one count", "Photometry"),
    ("GAIN", "Gain", 0.0, float, "e-/ADU, for the flux errors; 0 means infinite", "Photometry"),
]
DEFAULTS = {row[0]: row[2] for row in SETTINGS}


@dataclass
class SourceRun:
    number: int  # runs are numbered 1, 2, ... in a session
    image_name: str
    columns: Dict[str, np.ndarray]
    settings: Dict[str, object]
    segmentation: Optional[np.ndarray] = None
    command: List[str] = field(default_factory=list)
    version: str = ""
    seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    image_key: str = ""  # the image (and its models and residuals) the sources belong to
    engine: str = "SEP"

    def __len__(self) -> int:
        return len(self.columns.get("NUMBER", []))

    def ellipses(self, scale: float = 1.0):
        """(x, y, a, b, angle) of each source in the viewer's coordinates: the Kron
        ellipse (KRON_RADIUS x A, B) times ``scale``, at least 1.5 pixels."""
        c = self.columns
        kron = np.where(c["KRON_RADIUS"] > 0, c["KRON_RADIUS"], 2.5)
        a = np.maximum(scale * kron * c["A_IMAGE"], 1.5)
        b = np.maximum(scale * kron * c["B_IMAGE"], 1.5)
        x, y = c["X_IMAGE"] - 0.5, c["Y_IMAGE"] - 0.5
        return list(zip(x, y, a, b, c["THETA_IMAGE"]))

    def nearest(self, x: float, y: float) -> Optional[int]:
        """The row of the source nearest the point (x, y) in the viewer's coordinates."""
        if not len(self):
            return None
        d = np.hypot(self.columns["X_IMAGE"] - 0.5 - x, self.columns["Y_IMAGE"] - 0.5 - y)
        return int(np.argmin(d))


def source_mask(
    run: SourceRun, shape, keep: Optional[int] = None, scale: float = 1.5, chosen=None
) -> np.ndarray:
    """True inside the (scaled) Kron ellipse of every source except row ``keep`` (the
    galaxy being fitted), for masking other objects in an ELLIPROF fit.  ``chosen``
    (booleans, one per source) limits the mask to some sources."""
    ny, nx = shape
    masked = np.zeros(shape, bool)
    for row, (x, y, a, b, theta) in enumerate(run.ellipses(scale)):
        if row == keep or (chosen is not None and not chosen[row]):
            continue
        r = int(np.ceil(a)) + 1
        i0, i1 = max(int(x) - r, 0), min(int(x) + r + 1, nx)
        j0, j1 = max(int(y) - r, 0), min(int(y) + r + 1, ny)
        if i0 >= i1 or j0 >= j1:
            continue
        jj, ii = np.mgrid[j0:j1, i0:i1]
        dx, dy = ii + 0.5 - x, jj + 0.5 - y  # pixel centres in the viewer's coordinates
        t = np.radians(theta)
        u = dx * np.cos(t) + dy * np.sin(t)
        v = -dx * np.sin(t) + dy * np.cos(t)
        masked[j0:j1, i0:i1] |= (u / a) ** 2 + (v / b) ** 2 <= 1.0
    return masked


def galaxy(run: SourceRun, shape) -> Optional[int]:
    """The row of the source most likely to be the galaxy to fit: a large source near the
    middle of the image (its isophotal area, weighted down with the distance from the
    middle so that a big object at the edge does not win)."""
    if not len(run):
        return None
    ny, nx = shape
    c = run.columns
    d = np.hypot(c["X_IMAGE"] - 0.5 - nx / 2, c["Y_IMAGE"] - 0.5 - ny / 2) / np.hypot(nx, ny)
    score = c["ISOAREA_IMAGE"] * np.exp(-((d / 0.25) ** 2))
    return int(np.argmax(score))


def detect(
    image: Image,
    settings: Dict[str, object],
    number: int = 1,
    masked=None,
    local_rms: bool = False,
) -> SourceRun:
    """Find the sources in ``image``; ``settings`` uses SExtractor's keyword names.
    ``masked`` (True = masked) leaves pixels out of the detection and the background.
    ``local_rms``: the threshold is relative to the background mesh's local RMS map (as
    SExtractor's), not its global RMS: needed in a residual, where the SBF and the noise
    near a galaxy's centre are larger than in the sky."""
    import sep

    unknown = sorted(k for k in settings if k not in DEFAULTS)
    if unknown:
        raise RuntimeError(f"unknown setting {', '.join(unknown)} (help sources lists them)")
    start = time.perf_counter()

    def get(key):
        return settings.get(key, DEFAULTS[key])

    def yes(key):
        return str(get(key)).upper().startswith("Y")

    data = np.ascontiguousarray(image.data, dtype=np.float64)
    mask = ~np.isfinite(data)
    if masked is not None:
        mask |= masked
    data = np.where(mask, 0.0, data)
    size, filt = int(get("BACK_SIZE")), int(get("BACK_FILTERSIZE"))
    background = sep.Background(
        data, mask=mask, bw=size, bh=size, fw=filt, fh=filt, fthresh=float(get("BACK_FILTTHRESH"))
    )
    if str(get("BACK_TYPE")).upper() == "MANUAL":
        level = np.full(data.shape, float(get("BACK_VALUE")))
    else:
        level = background.back()
    sub = data - level
    sep.set_extract_pixstack(max(300000, data.size))
    sep.set_sub_object_limit(4096)
    kernel = None
    if yes("FILTER"):
        fwhm = float(get("FILTER_FWHM"))
        if fwhm > 0:  # a Gaussian kernel, as SExtractor's gauss_*.conv files
            r = max(int(np.ceil(1.5 * fwhm)), 1)
            yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
            kernel = np.exp(-(xx**2 + yy**2) / (2 * (fwhm / 2.3548) ** 2))
        else:
            kernel = sep.default_kernel
    absolute = str(get("THRESH_TYPE")).upper() == "ABSOLUTE"
    objects, segmentation = sep.extract(
        sub,
        float(get("DETECT_THRESH")),
        err=None if absolute else (background.rms() if local_rms else background.globalrms),
        mask=mask,
        minarea=int(get("DETECT_MINAREA")),
        filter_kernel=kernel,
        filter_type=str(get("FILTER_TYPE")) if kernel is not None else "matched",
        deblend_nthresh=int(get("DEBLEND_NTHRESH")),
        deblend_cont=float(get("DEBLEND_MINCONT")),
        clean=yes("CLEAN"),
        clean_param=float(get("CLEAN_PARAM")),
        segmentation_map=True,
    )
    x, y = objects["x"], objects["y"]
    a, b = objects["a"], objects["b"]
    # theta comes as float32, and pi/2 in float32 (a source along the columns, such as a
    # one-pixel streak) is a little more than pi/2 in float64: the aperture sums refuse it
    # ("invalid aperture parameters"), so it is brought back into [-pi/2, pi/2]
    theta = np.clip(objects["theta"].astype(np.float64), -np.pi / 2, np.pi / 2)
    # MAG_AUTO as SExtractor measures it: the flux in an ellipse of KRON_RADIUS times A
    # and B, where KRON_RADIUS is the Kron factor (2.5) times the first-moment radius, and
    # at least the minimum radius (3.5): PHOT_AUTOPARAMS
    try:
        k, r_min = (float(v) for v in str(get("PHOT_AUTOPARAMS")).split(","))
    except ValueError:
        raise RuntimeError("PHOT_AUTOPARAMS must be two numbers, e.g. 2.5,3.5")
    first_moment, _f = sep.kron_radius(sub, x, y, a, b, theta, 6.0, mask=mask)
    kron = np.maximum(k * first_moment, r_min)
    gain = float(get("GAIN")) or None
    flux, fluxerr, flag = sep.sum_ellipse(
        sub, x, y, a, b, theta, kron, err=background.globalrms, gain=gain, subpix=1, mask=mask
    )
    fraction = float(get("PHOT_FLUXFRAC"))
    if not 0 < fraction < 1:
        raise RuntimeError("PHOT_FLUXFRAC must be between 0 and 1")
    radius, _f = sep.flux_radius(sub, x, y, 6.0 * a, fraction, normflux=flux, subpix=5, mask=mask)
    zero = float(get("MAG_ZEROPOINT"))

    def magnitudes(f, ferr):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                np.where(f > 0, zero - 2.5 * np.log10(f), 99.0),
                np.where(f > 0, 1.0857 * ferr / f, 99.0),
            )

    mag, magerr = magnitudes(flux, fluxerr)
    # circular apertures (PHOT_APERTURES: diameters in pixels)
    try:
        diameters = [float(d) for d in str(get("PHOT_APERTURES")).replace(" ", "").split(",") if d]
    except ValueError:
        raise RuntimeError("PHOT_APERTURES must be diameters in pixels, e.g. 5,10")
    apertures = {}
    for k_, d in enumerate(diameters):
        f, ferr, _flag = sep.sum_circle(
            sub, x, y, d / 2.0, err=background.globalrms, gain=gain, subpix=5, mask=mask
        )
        m, merr = magnitudes(f, ferr)
        tag = "" if k_ == 0 else f"_{k_ + 1}"
        apertures.update({f"FLUX_APER{tag}": f, f"MAG_APER{tag}": m, f"MAGERR_APER{tag}": merr})
    iy = np.clip(np.round(y).astype(int), 0, data.shape[0] - 1)
    ix = np.clip(np.round(x).astype(int), 0, data.shape[1] - 1)
    cols = {
        "NUMBER": np.arange(1, len(x) + 1),
        "X_IMAGE": x + 1.0,  # SExtractor's convention: the first pixel's centre is 1
        "Y_IMAGE": y + 1.0,
        "A_IMAGE": a,
        "B_IMAGE": b,
        "THETA_IMAGE": np.degrees(theta),
        "ELLIPTICITY": 1.0 - b / np.where(a > 0, a, 1.0),
        "KRON_RADIUS": kron,
        "FLUX_RADIUS": radius,
        # the FWHM of a Gaussian with the object's second moments (an estimate)
        "FWHM_IMAGE": 2.3548 * np.sqrt((a * a + b * b) / 2.0),
        "ISOAREA_IMAGE": objects["npix"].astype(float),
        "FLUX_AUTO": flux,
        "FLUXERR_AUTO": fluxerr,
        "MAG_AUTO": mag,
        "MAGERR_AUTO": magerr,
        **apertures,
        "BACKGROUND": level[iy, ix],
        "FLAGS": objects["flag"].astype(int) | flag.astype(int),
    }
    w = image.wcs()
    if w is not None and len(x):
        sky = w.pixel_to_world(x, y)
        cols["ALPHA_J2000"] = sky.icrs.ra.deg
        cols["DELTA_J2000"] = sky.icrs.dec.deg
    used = dict(settings)
    return SourceRun(
        number=number,
        image_name=image.name,
        columns=cols,
        settings=dict(settings),
        segmentation=segmentation.astype(np.int32),
        command=["sep"] + [f"{k}={v}" for k, v in used.items()],
        version=f"SEP {sep.__version__}",
        seconds=time.perf_counter() - start,
        image_key=image.key,
        engine="SEP",
    )


# Faint sources in a residual (RESIDMASK): what the first source finding could not separate
# from the galaxies' light (globular clusters, faint stars and background galaxies on them)
RESID_THRESH = 3.0  # in units of the residual's local RMS
RESID_MINAREA = 5
RESID_GROW = 2  # pixels added around each one in the mask
RESID_BACK = 32  # the background mesh: what a model leaves at larger scales is not a source
RESID_CORE = 10.0  # left alone around each galaxy's centre: a model's error, not a source


def residual_sources(
    image: Image,
    residual: np.ndarray,
    masked: np.ndarray,
    centres,
    number: int = 1,
    thresh: float = RESID_THRESH,
    minarea: int = RESID_MINAREA,
    grow: int = RESID_GROW,
    core: float = RESID_CORE,
):
    """The sources in a fit's residual (not in ``masked``, not within ``core`` px of a
    galaxy's ``centres``), as a source run, and the pixels to add to the fitting mask: their
    segments grown by ``grow`` px."""
    from scipy import ndimage

    ny, nx = residual.shape
    yy, xx = np.mgrid[0:ny, 0:nx] + 0.5
    near = np.zeros(residual.shape, bool)
    for x, y in centres:
        near |= np.hypot(xx - x, yy - y) <= core
    settings = dict(
        DETECT_THRESH=float(thresh),
        DETECT_MINAREA=int(minarea),
        BACK_SIZE=RESID_BACK,
        BACK_FILTERSIZE=3,
    )
    data = np.where(np.isfinite(residual), residual, np.nan)
    run = detect(
        image.derived(f"{image.name} residual", data),
        settings,
        number,
        masked=masked | near,
        local_rms=True,
    )
    run.image_key = image.key
    seg = run.segmentation > 0 if run.segmentation is not None else np.zeros_like(masked)
    add = (ndimage.binary_dilation(seg, iterations=grow) if grow else seg) & ~masked & ~near
    return run, add
