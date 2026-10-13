"""ELLIPROF isophote fits, through the ``elliprof`` package.

The image (and the mask, when there is one) is written to a scratch folder, elliprof runs
there, and its profile, model and residual are read back.  ELLIPROF's centre (X0, Y0) is in
the viewer's coordinates (see images.py).
"""

from __future__ import annotations

import shlex
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits

from . import masks
from .images import Image, write_fits

Ellipse = Tuple[float, float, float, float, float]  # x, y, a, b, angle (degrees from +x)


@dataclass
class FitSettings:
    x0: float
    y0: float
    r0: float = 5.0  # first semi-major axis, pixels
    r1: float = 100.0  # last semi-major axis, pixels
    nr: int = 20  # number of isophotes
    niter: int = 10  # iterations of the fit
    sky: Optional[float] = None  # a constant sky subtracted first (None: none)
    rmstar: bool = False  # reject star-like bright samples along each ellipse
    mask_file: Optional[Path] = None  # a mask file elliprof reads (.dmask or FITS)
    extra_mask: Optional[np.ndarray] = None  # True where mask layers mask a pixel
    extra_mask_label: str = ""  # the names of those layers


@dataclass
class FitResult:
    profile: object  # pandas.DataFrame: Rmaj, x0, y0, I0, alpha, ellip, I3, A3, I4, A4, slope
    model: np.ndarray
    residual: np.ndarray  # masked pixels are NaN
    masked: np.ndarray  # True where the fit ignored the pixel
    ellipses: List[Ellipse] = field(default_factory=list)  # in the viewer's coordinates
    command: str = ""
    version: str = ""
    seconds: float = 0.0


def fit(image: Image, s: FitSettings) -> FitResult:
    """Fit isophotes to ``image``.  Raises RuntimeError with elliprof's own message."""
    try:
        import elliprof
        from elliprof import run_elliprof
        from elliprof.regions import profile_ellipses
    except ImportError as e:  # pragma: no cover - elliprof is a dependency
        raise RuntimeError("the elliprof package is not installed (pip install elliprof)") from e

    if not (0 < s.r0 < s.r1):
        raise ValueError("the radii must satisfy 0 < R0 < R1")
    if s.nr < 2:
        raise ValueError("NR must be at least 2")
    start = time.perf_counter()
    masked = np.zeros(image.shape, bool)
    if s.mask_file is not None:
        masked |= masks.load(s.mask_file, image.shape)
    if s.extra_mask is not None:
        masked |= s.extra_mask
    with tempfile.TemporaryDirectory(prefix="astro-vista-") as tmp:
        tmp = Path(tmp)
        source = image.derived(image.name, image.data)
        for key in ("CNPIX1", "CNPIX2"):  # elliprof would add an image origin to X0, Y0
            source.header.remove(key, ignore_missing=True)
        path = write_fits(source, tmp / "image.fits")
        mask_path = None
        if masked.any():
            mask_path = tmp / "mask.fits"  # 1 = good, 0 = masked
            fits.PrimaryHDU((~masked).astype(np.float32)).writeto(mask_path)
        result = run_elliprof(
            str(path),
            float(s.x0),
            float(s.y0),
            mask=str(mask_path) if mask_path else None,
            sky=s.sky,
            r0=float(s.r0),
            r1=float(s.r1),
            nr=int(s.nr),
            niter=int(s.niter),
            rmstar=s.rmstar,
            output_dir=str(tmp),
            model_path=str(tmp / "model.fits"),
            residual_path=str(tmp / "residual.fits"),
            csv_path=str(tmp / "profile.csv"),
            check=False,
        )
        if result.returncode != 0 or result.profile is None:
            lines = (result.stderr or result.stdout or "").strip().splitlines()
            raise RuntimeError(
                "ELLIPROF failed: " + (lines[-1] if lines else f"exit {result.returncode}")
            )
        model = fits.getdata(tmp / "model.fits").astype(np.float32)
        residual = fits.getdata(tmp / "residual.fits").astype(np.float32)
    residual[masked] = np.nan
    profile = result.profile
    # elliprof's DS9 ellipses put the first pixel's centre at 1; the viewer at 0.5
    ellipses = [(x - 0.5, y - 0.5, a, b, t) for x, y, a, b, t in profile_ellipses(profile)]
    return FitResult(
        profile=profile,
        model=model,
        residual=residual,
        masked=masked,
        ellipses=ellipses,
        command=command_line(image, s),
        version=elliprof.__version__,
        seconds=time.perf_counter() - start,
    )


def command_line(image: Image, s: FitSettings) -> str:
    """The equivalent elliprof command line (for the log and for reproducing a fit)."""
    parts = ["elliprof", str(image.path or image.name)]
    if s.mask_file is not None:
        parts += ["--mask", str(s.mask_file)]
    if s.sky is not None:
        parts += ["--sky", f"{s.sky:g}"]
    parts += [f"X0={s.x0:g}", f"Y0={s.y0:g}", f"R0={s.r0:g}", f"R1={s.r1:g}", f"NR={s.nr}"]
    parts += [f"NITER={s.niter}"] + (["RMSTAR"] if s.rmstar else [])
    line = " ".join(shlex.quote(p) if " " in p else p for p in parts)
    if s.extra_mask is not None:
        line += f"   # plus the mask layers {s.extra_mask_label}"
    return line
