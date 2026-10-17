"""ELLIPROF isophote fits, through the ``elliprof`` package.

The image (and the mask, when there is one) is written to a scratch folder, elliprof runs
there, and its profile, model and residual are read back.  ELLIPROF's centre (X0, Y0) is in
the viewer's coordinates (see images.py).
"""

from __future__ import annotations

import shlex
import tempfile
import time
from dataclasses import dataclass, field, replace
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
    # passed to elliprof only when set (None: elliprof's own default)
    fixctr: Optional[int] = None  # 0 free centres, 1 fixed at X0, Y0, 2 their median
    cos3x: Optional[int] = None  # 3rd (or, negative, 6th) order harmonics
    cos4x: Optional[int] = None  # 4th order harmonics
    rlaw: Optional[int] = None  # spacing of the radii
    gain: Optional[float] = None
    tie: Optional[int] = None


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
    notes: List[str] = field(default_factory=list)  # what was changed to fit (R0 raised)


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
    notes = []
    # a masked centre: R0 raised until elliprof can fit the isophote there
    r0, note = clear_centre(
        np.asarray(image.data, np.float64),
        s.x0,
        s.y0,
        s.r0,
        s.r1,
        masked | ~np.isfinite(image.data),
        sky=s.sky,
        fixctr=s.fixctr if s.fixctr is not None else 0,
    )
    if note:
        notes.append(note)
    if r0 != s.r0:
        s = replace(s, r0=r0)
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
            **native_options(s),
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
        notes=notes,
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
    parts += [f"{k.upper()}={v:g}" for k, v in native_options(s).items()]
    line = " ".join(shlex.quote(p) if " " in p else p for p in parts)
    if s.extra_mask is not None:
        line += f"   # plus the mask layers {s.extra_mask_label}"
    return line


# harmonic modes by name: (COS3X, COS4X); "m" takes the median coefficients over all
# isophotes, and a negative COS3X fits the 6th order instead of the 3rd
MODES = {
    "none": (0, 0),
    "3": (2, 0),
    "4": (0, 2),
    "3,4": (2, 2),
    "4,6": (-2, 2),
    "3m": (1, 0),
    "4m": (0, 1),
    "6m,4m": (-1, 1),
}
# what elliprof says when a 6th-order model is wrong beyond a PA wrap
SIXTH_ORDER_WARNING = "the fitted PA wraps across 0/180"


def native_options(s) -> dict:
    """The optional ELLIPROF keywords set in ``s`` (FitSettings or alike)."""
    out = {}
    for key in ("fixctr", "cos3x", "cos4x", "rlaw", "gain", "tie"):
        value = getattr(s, key, None)
        if value is not None:
            out[key] = value
    return out


@dataclass
class ArrayFit:
    """One elliprof call on an array, as the pair fit needs it: the model at elliprof's
    own precision, promoted to float64, and everything said."""

    profile: object  # pandas.DataFrame, or None
    model: Optional[np.ndarray]  # float64
    returncode: int
    stdout: str
    stderr: str
    arguments: List[str]
    seconds: float
    precision: Optional[str] = None
    message: str = ""  # why it failed, when it did

    @property
    def output(self) -> str:
        return (self.stdout or "") + "\n" + (self.stderr or "")


def fit_array(
    data: np.ndarray,
    x0: float,
    y0: float,
    r0: float,
    r1: float,
    nr: int,
    *,
    niter: int = 10,
    sky: Optional[float] = None,
    masked: Optional[np.ndarray] = None,
    rmstar: bool = False,
    timeout: Optional[float] = 300.0,
    header=None,
    **options,
) -> ArrayFit:
    """Fit isophotes to an array (written as float32 FITS, as the single fit does).
    ``masked``: True where a pixel is left out.  ``options``: fixctr, cos3x, cos4x, rlaw,
    gain, tie.  Never raises for a failed fit: the result says why."""
    from elliprof import run_elliprof

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="astro-vista-pair-") as tmp:
        tmp = Path(tmp)
        hdr = fits.Header() if header is None else header.copy()
        for key in ("CNPIX1", "CNPIX2", "BSCALE", "BZERO", "BLANK"):
            hdr.remove(key, ignore_missing=True)
        fits.PrimaryHDU(np.asarray(data, np.float32), header=hdr).writeto(
            tmp / "image.fits", output_verify="silentfix"
        )
        mask_path = None
        if masked is not None and masked.any():
            mask_path = tmp / "mask.fits"  # 1 = good, 0 = masked
            fits.PrimaryHDU((~masked).astype(np.float32)).writeto(mask_path)
        kwargs = dict(
            mask=str(mask_path) if mask_path else None,
            sky=sky,
            r0=float(r0),
            r1=float(r1),
            nr=int(nr),
            niter=int(niter),
            rmstar=rmstar,
            output_dir=str(tmp),
            model_path=str(tmp / "model.fits"),
            csv_path=str(tmp / "profile.csv"),
            timeout=timeout,
            check=False,
            **{k: v for k, v in options.items() if v is not None},
        )
        try:
            result = run_elliprof(str(tmp / "image.fits"), float(x0), float(y0), **kwargs)
        except Exception as e:  # a timeout, a bad value
            return ArrayFit(
                None,
                None,
                -1,
                "",
                str(e),
                [],
                time.perf_counter() - start,
                message=f"{type(e).__name__}: {e}",
            )
        model = None
        if (tmp / "model.fits").exists():
            model = np.asarray(fits.getdata(tmp / "model.fits")).astype(np.float64)
        message = ""
        if result.returncode != 0 or result.profile is None:
            lines = (result.stderr or result.stdout or "").strip().splitlines()
            message = lines[-1] if lines else f"exit {result.returncode}"
        return ArrayFit(
            result.profile,
            model,
            result.returncode,
            result.stdout or "",
            result.stderr or "",
            [str(a) for a in result.command],
            time.perf_counter() - start,
            result.precision,
            message,
        )


# a galaxy whose centre is masked: R0 is raised by this factor at a time until elliprof can
# fit the isophote at R0 (clear_centre)
R0_GROWTH = 1.05


def centre_masked(masked: Optional[np.ndarray], x0: float, y0: float) -> bool:
    """Whether the pixel at the centre (ELLIPROF's coordinates) is masked."""
    if masked is None:
        return False
    i, j = int(np.floor(x0)), int(np.floor(y0))
    ny, nx = masked.shape
    return 0 <= i < nx and 0 <= j < ny and bool(masked[j, i])


def first_isophote_ok(profile) -> bool:
    """Whether elliprof fitted the innermost isophote: every value finite, an ellipticity of
    0 to 0.95 and a positive intensity (a masked one comes back NaN, or as its start)."""
    if profile is None or not len(profile):
        return False
    row = profile.iloc[0]
    try:
        values = [float(row[k]) for k in ("Rmaj", "x0", "y0", "I0", "alpha", "ellip")]
    except (KeyError, TypeError, ValueError):
        return False
    if not all(np.isfinite(values)):
        return False
    return 0 <= values[5] < 0.95 and values[3] > 0


def clear_centre(
    data: np.ndarray,
    x0: float,
    y0: float,
    r0: float,
    r1: float,
    masked: Optional[np.ndarray],
    sky: Optional[float] = None,
    fitter=None,
    **options,
) -> Tuple[float, str]:
    """R0 for a galaxy whose centre is masked: raised by 5 % at a time (R0_GROWTH) until
    elliprof fits the isophote at R0 (a short call, four isophotes out to 1.6 R0, on a
    cut-out around the centre), never past half R1.  (R0, what was done); R0 as given and
    "" when the centre is not masked; R0 as given and why when no R0 would do."""
    if not centre_masked(masked, x0, y0):
        return r0, ""
    fitter = fitter or fit_array
    ny, nx = data.shape
    r, tries = float(r0), 0
    while r < 0.5 * r1:
        tries += 1
        outer = min(1.6 * r, r1)
        half = int(np.ceil(outer * 1.3)) + 8  # the cut-out: the probe's isophotes and more
        i0, j0 = max(int(x0) - half, 0), max(int(y0) - half, 0)
        i1, j1 = min(int(x0) + half + 1, nx), min(int(y0) + half + 1, ny)
        probe = fitter(
            data[j0:j1, i0:i1],
            x0 - i0,
            y0 - j0,
            r,
            outer,
            4,
            sky=sky,
            masked=masked[j0:j1, i0:i1],
            **options,
        )
        if first_isophote_ok(probe.profile):
            if tries == 1:
                return r, ""  # masked, but elliprof fits it at R0 as it is
            return r, (
                f"the centre is masked: R0 raised from {r0:g} to {r:.3g} px (5 % at a time) "
                "until elliprof could fit the isophote there"
            )
        r *= R0_GROWTH
    return r0, (
        f"the centre is masked and elliprof could fit no isophote between R0 {r0:g} and half "
        f"R1 ({0.5 * r1:g} px): R0 kept at {r0:g}"
    )
