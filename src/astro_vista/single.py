"""Iterative fits of one galaxy: the pair fit's options for a single ELLIPROF fit.

A single galaxy has no companion to take turns with, so R1 is not grown: the galaxy is
fitted out to R1 at once, and to a smaller R1 when the isophotes fail the pair fit's checks;
the outermost isophotes are checked against the trend inside them (SHAPETOL); the sky factor
is refined from the residual sky (SKYITER) and a tilt of the sky taken out (SKYPLANE); a
search over centre fitting keeps the fit with the smallest residual (TRYFIX), and with
SEARCH=FULL over R0 and TIE as well, then the fixed centre to a fraction of a pixel and,
after the harmonics (HARMONICS=AUTO, TRYCOS3X, TRYCOS4X), the number of isophotes; the faint
sources of the residual are masked and the galaxy fitted again (RESIDMASK); and N fits, the
fit kept and N - 1 with R0 and R1 varied, are averaged (PARALLEL), those with a residual
much larger than the kept's left out.  Every model is listed with its residual (Rings) and
the lowest kept.  Every elliprof call goes through fitting.fit_array, as the pair fit's.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import pair
from .fitting import centre_masked, fit_array


@dataclass
class SingleSettings:
    x0: float
    y0: float
    r0: float = 3.0
    r1: float = 100.0
    nr: int = 0  # 0: automatic (isophotes about 30 % apart)
    niter: int = 10
    sky: float = 0.0
    sky_base: Optional[float] = None
    sky_factor: float = 1.0
    sky_iter: int = 0  # SKYITER: refine the sky factor from the residual sky
    sky_plane: bool = False  # SKYPLANE: take out a tilt of the sky the residual shows
    rmstar: bool = False
    tie: Optional[int] = None
    rlaw: Optional[int] = None
    gain: Optional[float] = None
    fixctr: Optional[int] = None
    cos3x: Optional[int] = None
    cos4x: Optional[int] = None
    tail: str = "elliprof"  # the model beyond R1 (pair.tail_model)
    try_fix: List[int] = field(default_factory=list)
    try_cos3x: List[int] = field(default_factory=list)
    try_cos4x: List[int] = field(default_factory=list)
    harmonic_search: str = ""  # HARMONICS=COMMON or AUTO (pair.harmonic_values)
    wide: bool = False  # SEARCH=FULL: R0, TIE, the centre and NR searched as well
    shape_tol: float = 0.0  # the outer-shape guard (0: off)
    parallel: int = 1  # N fits at the same time, all averaged
    runs: int = 1  # another number of fits averaged than PARALLEL's
    jitter: float = 0.15
    seed: Optional[int] = None
    resid_rounds: int = 0
    resid_thresh: float = 8.0
    resid_grow: int = 2
    resid_core: float = 10.0
    timeout: float = 300.0

    def component(self, **changes) -> pair.Component:
        values = dict(
            id="A",
            x0=self.x0,
            y0=self.y0,
            r0=self.r0,
            r1=self.r1,
            nr=self.nr,
            fixctr=self.fixctr if self.fixctr is not None else 0,
            cos3x=self.cos3x,
            cos4x=self.cos4x,
        )
        values.update(changes)
        return pair.Component(**values)


@dataclass
class SingleResult:
    status: str  # "fitted", "incomplete" or "cancelled"
    settings: SingleSettings
    profile: object = None
    model: Optional[np.ndarray] = None  # with the model beyond R1 (TAIL)
    residual: Optional[np.ndarray] = None  # image - sky - model, every pixel
    sky_map: Optional[np.ndarray] = None  # the sky subtracted when it has a tilt
    r1: float = 0.0  # the R1 the fit reached
    nr: int = 0
    calls: int = 0
    seconds: float = 0.0
    steps: List[str] = field(default_factory=list)
    reason: str = ""
    score: Optional[float] = None
    core: Optional[float] = None  # the residual of the innermost rings (Rings.measure)
    sky_iterations: List[dict] = field(default_factory=list)
    search: List[dict] = field(default_factory=list)
    ensemble: dict = field(default_factory=dict)
    resid_runs: List[object] = field(default_factory=list)
    resid_mask: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None  # the fitting mask used
    harmonics: List[dict] = field(default_factory=list)  # the harmonic analysis, every try
    models: List[dict] = field(default_factory=list)  # every model made: settings, residual
    # every complete model, by its number: its model (float32), sky, sky tilt and mask, so
    # that any of them can be shown or several averaged afterwards (Session.single_models)
    model_store: Dict[int, dict] = field(default_factory=dict)
    harmonic_choice: Tuple[int, int] = (2, 2)  # COS3X, COS4X of the fit kept
    rings: Optional["Rings"] = None  # where every model's residual was measured

    @property
    def complete(self) -> bool:
        return self.profile is not None and self.model is not None


Progress = Optional[Callable[[dict], None]]
Cancelled = Optional[Callable[[], bool]]


HARMONIC_ROUNDS = 3  # the harmonics and the search, alternated at most this many times
# every model's residual is measured between SCORE_INNER x R0 (the core is often
# disturbed) and SCORE_OUTER x R1, R0 and R1 as asked: the same annulus for every model
# (search, centre, harmonics, isophotes, sky tilt), and the lowest residual kept
SCORE_INNER = 1.15
SCORE_OUTER = 1.1
# the annulus is cut into rings, each RING_RATIO times wider than the one inside it, and
# each ring counts the same: the residual's RMS over the ring's own noise, a pixel at most
# SCORE_CLIP times it (the stars left); the score is the RMS of the rings' values, so a
# residual the bright inner rings hold is not hidden by the large faint outer ones
RING_RATIO = 1.2
SCORE_CLIP = 10.0
RING_PIXELS = 30  # a ring with fewer usable pixels is left out
# the core: the rings out to CORE_SPAN x the annulus' inner radius; a model whose core
# residual is over CORE_LIMIT x the cleanest core of the models compared (and over
# CORE_FLOOR) has a high-contrast core: it is never kept nor averaged
CORE_SPAN = 2.0
CORE_LIMIT = 2.0
CORE_FLOOR = 3.0
# the wide search (SEARCH=FULL): R0 also R0_OUT x the R0 asked, or out of a masked core
# (R0_MARGIN x its radius); the centre moved by CENTRE_STEP px, the step halved down to
# CENTRE_LAST, at most CENTRE_FITS fits; NR x each of NR_FACTORS (with NR=AUTO)
R0_OUT = 1.5
R0_MARGIN = 1.25
CENTRE_STEP = 0.5
CENTRE_LAST = 0.125
CENTRE_FITS = 40
NR_FACTORS = (0.75, 1.5, 2.0)
# the averaged fits: one whose residual is over (1 + AVERAGE_TOL) x the kept's, whose
# isophotes stopped short of AVERAGE_REACH x the kept's R1 or whose core is high-contrast is
# left out of the mean
AVERAGE_TOL = 0.05
AVERAGE_REACH = 0.8


class _Stop(Exception):
    pass


def _options(s: SingleSettings, comp: pair.Component) -> dict:
    return dict(
        niter=s.niter,
        sky=s.sky,
        rmstar=s.rmstar,
        timeout=s.timeout,
        cos3x=comp.cos3x,
        cos4x=comp.cos4x,
        rlaw=s.rlaw,
        gain=s.gain,
        tie=s.tie,
        fixctr=comp.fixctr,
    )


def fit_once(
    image: np.ndarray,
    mask: np.ndarray,
    s: SingleSettings,
    header=None,
    progress: Progress = None,
    cancelled: Cancelled = None,
    fitter=fit_array,
    counter: Optional[Dict[str, int]] = None,
) -> SingleResult:
    """One fit of the galaxy out to R1 (a smaller R1 when it fails the checks), then the
    outer-shape guard."""
    counter = counter if counter is not None else {"n": 0}
    start = time.perf_counter()
    finite = np.isfinite(image)
    out = SingleResult("incomplete", s)
    comp = s.component()

    def stop():
        if cancelled is not None and cancelled():
            raise _Stop()

    def call(r1: float, nr: Optional[int] = None, floor: float = 0.0):
        """A fit out to r1 that passes the checks, retried from a larger R0 when the
        innermost isophotes fail and inside the radius where it failed: (fit, r1, nr) or
        (None, reason, None)."""
        r0, inner, reason = comp.r0, pair.INNER_TRIES, ""
        for _attempt in range(6):
            stop()
            trial = replace(comp, r0=r0, r1=r1, nr=comp.nr if nr is None else nr)
            n = pair.nr_for(trial, r1)
            counter["n"] += 1
            number = counter["n"]
            if progress is not None:
                progress({"phase": "fit", "r1": r1, "call": number})
            fit = fitter(
                image,
                comp.x0,
                comp.y0,
                r0,
                r1,
                n,
                masked=mask,
                header=header,
                **_options(s, trial),
            )
            ok, reason, _kind, radius = pair.check(fit, trial, r1, finite)
            if progress is not None:
                progress(
                    {
                        "phase": "call",
                        "call": number,
                        "r0": r0,
                        "r1": r1,
                        "nr": n,
                        "passed": ok,
                        "reason": reason,
                        "ellipses": pair.profile_ellipses(fit.profile) if ok else None,
                    }
                )
            if ok:
                return fit, r1, n
            if radius is not None and radius <= pair.INNER_R0 and inner and 1.5 * r0 < 0.25 * r1:
                inner -= 1
                r0 = round(1.5 * r0, 1)
                out.steps.append(f"the innermost isophotes failed: R0 {r0:g}")
                continue
            shrink = 0.8 * r1
            if radius is not None and radius > 1.5 * r0:
                shrink = min(shrink, 0.95 * radius)
            if shrink < max(pair.JOE_MIN_R1, floor + 0.5):
                break
            r1 = shrink
        return None, reason, None

    try:
        got = call(s.r1)
        if got[0] is None:
            out.reason = f"no fit passed: {got[1]}"
            out.seconds = time.perf_counter() - start
            out.calls = counter["n"]
            return out
        if got[1] < s.r1 - 0.5:
            out.steps.append(f"R1 {s.r1:.1f}: fitted to {got[1]:.1f}")
        fit, r1, nr = got
        if s.shape_tol > 0:
            fit, r1, nr = _guard(fit, r1, nr, comp, s, image, mask, call, out)
        usable = finite & ~mask
        out.model = pair.tail_model(fit.model, fit.profile, s.tail, image - s.sky, usable)
        out.profile = fit.profile
        out.residual = image - s.sky - out.model
        out.r1, out.nr = r1, nr
        out.status = "fitted"
    except _Stop:
        out.status = "cancelled"
    out.calls = counter["n"]
    out.seconds = time.perf_counter() - start
    return out


def _guard(fit, r1, nr, comp, s, image, mask, call, out):
    """SHAPETOL: the outermost isophotes checked against the trend inside them; when one
    is off (or the model is anomalous beyond R1), coarser isophotes and an R1 inside it are
    tried and the largest clean fit kept (pair.outer_shape, pair.model_anomaly)."""
    finite = np.isfinite(image)

    def flaw(f):
        bad = pair.outer_shape(f.profile, s.shape_tol)
        if bad is not None:
            return bad
        usable = finite & ~mask
        model = pair.tail_model(f.model, f.profile, s.tail, image - s.sky, usable)
        why = pair.model_anomaly(model, f.profile, image - s.sky, usable)
        return {"r": None, "good": None, "d": 0.0, "model": why} if why else None

    bad = flaw(fit)
    if bad is None:
        return fit, r1, nr
    what = (
        f"its ellipse at {bad['r']:.1f} is off the trend (by {bad['d']:.2f})"
        if bad["r"] is not None
        else bad["model"]
    )
    tries = []
    if comp.nr == 0 and nr > 5:
        tries.append((r1, max(4, int(round(pair.SHAPE_COARSE * nr)))))
    trim = bad["good"] if bad["good"] is not None else 0.85 * r1
    if trim >= pair.JOE_MIN_R1:
        tries.append((trim, None))
        n_trim = pair.nr_for(replace(comp, r1=trim), trim)
        if comp.nr == 0 and n_trim > 5:
            tries.append((trim, max(4, int(round(pair.SHAPE_COARSE * n_trim)))))
    clean, shaped = None, None
    for r_try, n_try in tries:
        f2, r2, n2 = call(r_try, nr=n_try)
        if f2 is None:
            continue
        b2 = flaw(f2)
        if b2 is None and (clean is None or r2 > clean[1] + 0.5):
            clean = (f2, r2, n2)
        if (b2 is None or b2["r"] is None) and (shaped is None or r2 > shaped[1] + 0.5):
            shaped = (f2, r2, n2)
    chosen = clean or (shaped if bad["r"] is not None else None)
    if chosen is None:
        out.steps.append(f"outer isophotes: {what}; no other radius or NR did better, kept")
        return fit, r1, nr
    out.steps.append(f"outer isophotes: {what}; refitted with R1 {chosen[1]:.1f}, NR {chosen[2]}")
    return chosen


def _key(v: "SingleSettings") -> tuple:
    return (
        v.fixctr,
        v.cos3x,
        v.cos4x,
        round(v.x0, 4),
        round(v.y0, 4),
        v.r0,
        v.r1,
        v.nr,
        v.tie,
        round(v.sky, 6),
        v.sky_iter,
    )


class Rings:
    """The annulus where every model's residual is measured (inner..outer px from
    ``centre``), cut into rings RING_RATIO apart.  Each ring's noise is measured once, on
    the image, from its second differences along x: the photon noise of the galaxy and the
    sky, the smooth light cancelled; so every model is measured against the same noise."""

    def __init__(self, image: np.ndarray, mask: np.ndarray, centre, inner: float, outer: float):
        self.centre, self.inner, self.outer = (float(centre[0]), float(centre[1])), inner, outer
        edges = [inner]
        while edges[-1] < outer * (1 - 1e-9):
            edges.append(min(edges[-1] * RING_RATIO, outer))
        self.edges = np.array(edges)
        ny, nx = image.shape
        yy, xx = np.mgrid[0:ny, 0:nx] + 0.5
        r = np.hypot(xx - centre[0], yy - centre[1])
        self.index = np.full(image.shape, -1, np.int32)
        inside = (r >= inner) & (r < outer)
        self.index[inside] = np.searchsorted(self.edges, r[inside], side="right") - 1
        ok = ~mask & np.isfinite(image)
        both = np.zeros(image.shape, bool)
        both[:, 1:-1] = ok[:, :-2] & ok[:, 1:-1] & ok[:, 2:]
        both &= self.index >= 0
        d2 = image[:, :-2] - 2 * image[:, 1:-1] + image[:, 2:]
        k = self.index[:, 1:-1][both[:, 1:-1]]
        v = d2[both[:, 1:-1]]
        self.sigma = np.full(len(self.edges) - 1, np.nan)
        for ring in range(len(self.sigma)):
            vr = v[k == ring]
            if vr.size >= RING_PIXELS:
                self.sigma[ring] = 1.4826 * np.median(np.abs(vr - np.median(vr))) / math.sqrt(6)

    @property
    def core_rings(self) -> int:
        """How many rings the core has (at least one)."""
        return max(1, int(np.sum(self.edges[1:] <= CORE_SPAN * self.inner * (1 + 1e-9))))

    def measure(self, residual: np.ndarray, mask: Optional[np.ndarray]) -> Optional[dict]:
        """The residual's value in each ring (its RMS over the ring's noise), the score (the
        RMS of the rings') and the core's (of the innermost rings'); None when fewer than
        two rings can be measured."""
        sel = (self.index >= 0) & np.isfinite(residual)
        if mask is not None:
            sel &= ~mask
        k = self.index[sel]
        s = self.sigma[k]
        good = np.isfinite(s) & (s > 0)
        k = k[good]
        v = np.clip(residual[sel][good] / s[good], -SCORE_CLIP, SCORE_CLIP)
        n = len(self.sigma)
        counts = np.bincount(k, minlength=n)
        sums = np.bincount(k, v * v, minlength=n)
        chi = np.where(counts >= RING_PIXELS, np.sqrt(sums / np.maximum(counts, 1)), np.nan)
        use = np.isfinite(chi)
        if use.sum() < 2:
            return None
        core = chi[: self.core_rings]
        core = core[np.isfinite(core)]
        if not core.size:  # rings too small to measure: the innermost one that is
            core = chi[use][:1]
        return {
            "score": float(np.sqrt(np.mean(chi[use] ** 2))),
            "core": float(np.sqrt(np.mean(core**2))) if core.size else None,
            "rings": chi,
        }


def masked_core(mask: np.ndarray, centre, limit: float) -> float:
    """How far out from ``centre`` the mask covers the core: the radius out to which every
    1-px ring is at least half masked (0 when the centre is not masked)."""
    cx, cy = centre
    ny, nx = mask.shape
    reach = int(math.ceil(limit)) + 2
    j0, j1 = max(int(cy) - reach, 0), min(int(cy) + reach + 1, ny)
    i0, i1 = max(int(cx) - reach, 0), min(int(cx) + reach + 1, nx)
    if j0 >= j1 or i0 >= i1:
        return 0.0
    yy, xx = np.mgrid[j0:j1, i0:i1] + 0.5
    r = np.hypot(xx - cx, yy - cy)
    m = mask[j0:j1, i0:i1]
    for k in range(int(limit) + 1):
        ring = (r >= k) & (r < k + 1)
        if ring.any() and m[ring].mean() < 0.5:
            return float(k)
    return float(int(limit) + 1)


def _model_row(number, stage, settings, r, tilt, chooses, rings) -> dict:
    """A model's parameters and its residual (the models table)."""
    return {
        "n": number,
        "stage": stage,
        "status": r.status,
        "x0": settings.x0,
        "y0": settings.y0,
        "fixctr": settings.fixctr if settings.fixctr is not None else 0,
        "cos3x": settings.cos3x if settings.cos3x is not None else 2,
        "cos4x": settings.cos4x if settings.cos4x is not None else 2,
        "tie": settings.tie,
        "niter": settings.niter,
        "r0": settings.r0,
        "r1": settings.r1,
        "r1_reached": r.r1,
        "nr": r.nr,
        "sky": r.settings.sky,
        "tilt": tilt,
        "inner": rings.inner if rings is not None else SCORE_INNER * settings.r0,
        "outer": rings.outer if rings is not None else SCORE_OUTER * settings.r1,
        "score": r.score,
        "core": r.core,
        "flag": "",
        "chooses": chooses,
    }


def score(result: SingleResult, mask: np.ndarray, rings: "Rings") -> Optional[float]:
    """The model's residual: the RMS, over the rings of the annulus, of the residual's RMS
    over each ring's noise (Rings.measure); None when it cannot be measured."""
    if result.residual is None:
        return None
    got = rings.measure(result.residual, mask)
    return None if got is None else got["score"]


def with_sky(image, mask, s, support, header, progress, cancelled, fitter, counter):
    """SKYITER: the galaxy fitted up to SKYITER times, the sky factor updated each time from
    the residual on the sky support (f + median / base), as the pair fit does; the fit with
    the smallest |median| is kept.  Stops when the residual does not follow the sky (a halo
    that fills the frame)."""
    from . import sky as skies

    if s.sky_iter <= 0 or support is None:
        return fit_once(image, mask, s, header, progress, cancelled, fitter, counter)
    base = s.sky_base if s.sky_base is not None else s.sky
    use = support & ~mask
    factor = s.sky_factor
    history, best, previous, first = [], None, None, None
    for k in range(s.sky_iter):
        trial = replace(s, sky=factor * base, sky_factor=factor)
        tagged = progress
        if progress is not None:
            tagged = lambda info, k=k, f=factor: progress(  # noqa: E731
                {**info, "sky_iteration": k + 1, "sky_iterations": s.sky_iter, "sky_factor": f}
            )
        result = fit_once(image, mask, trial, header, tagged, cancelled, fitter, counter)
        entry = {"iteration": k + 1, "factor": factor, "sky": trial.sky, "status": result.status}
        history.append(entry)
        if not result.complete:
            break
        level = skies.residual_level(result.residual, use, base, factor)
        entry.update({key: level.get(key) for key in ("median", "sigma", "next_factor")})
        if "median" not in level:
            best = best or (0.0, result, k)
            break
        if first is None:
            first = (abs(level["median"]), result, k)
        step = trial.sky - previous[0] if previous is not None else 0.0
        if step and abs(step) >= 0.5 * level["sigma"]:
            response = (previous[1] - level["median"]) / step
            if response < pair.SKY_RESPONSE:
                entry["note"] = "the sky and the galaxy's outer light are degenerate: stopped"
                best = first
                break
        previous = (trial.sky, level["median"])
        if best is None or abs(level["median"]) < best[0]:
            best = (abs(level["median"]), result, k)
        if abs(level["median"]) < 0.01 * max(level["sigma"], 1e-300):
            break
        nxt = level.get("next_factor")
        if nxt is None or not 0.5 <= nxt <= 1.5:
            break
        largest = pair.SKY_STEP * level["sigma"] / base if base else abs(nxt - factor)
        if abs(nxt - factor) > largest:
            nxt = factor + math.copysign(largest, nxt - factor)
        factor = nxt
    chosen = best[1] if best is not None else result
    if best is not None:
        history[best[2]]["chosen"] = True
    chosen.sky_iterations = history
    return chosen


def fit_single(
    image: np.ndarray,
    mask: np.ndarray,
    s: SingleSettings,
    header=None,
    support: Optional[np.ndarray] = None,
    image_obj=None,
    progress: Progress = None,
    cancelled: Cancelled = None,
    fitter=fit_array,
) -> SingleResult:
    """The whole iterative fit (see the module's description)."""
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.perf_counter()
    counter = {"n": 0}
    stopped = lambda: cancelled is not None and cancelled()  # noqa: E731
    centre = (s.x0, s.y0)
    # every model measured on the same rings: SCORE_INNER x R0 to SCORE_OUTER x R1 as asked
    rings = Rings(image, mask, centre, SCORE_INNER * s.r0, SCORE_OUTER * s.r1)

    models: List[dict] = []  # every model made
    order = itertools.count(1)  # each model's number, given when it is started
    fitted: Dict[tuple, SingleResult] = {}  # the fits with a fixed sky, by their settings

    def say(info):
        if progress is not None:
            progress(info)

    def one(settings, m=mask, img=image, stage="fit", plane=None, chooses=True, number=None):
        """One fit (with its sky iterations), measured on the rings and listed in
        ``models`` (``chooses``: one of the models the lowest residual is chosen from);
        every call reported, by ``stage``."""
        number = number if number is not None else next(order)
        tagged = None
        if progress is not None:
            tagged = lambda info: progress({**info, "stage": stage})  # noqa: E731
        r = with_sky(img, m, settings, support, header, tagged, cancelled, fitter, counter)
        r.mask = m
        if r.complete:
            got = rings.measure(r.residual, m)
            if got is not None:
                r.score, r.core = got["score"], got["core"]
        row = _model_row(number, stage, settings, r, plane is not None, chooses, rings)
        models.append({**row, "settings": settings, "result": r, "plane": plane})
        say(
            {
                "phase": "fitted",
                **row,
                "call": counter["n"],
                "ellipses": pair.profile_ellipses(r.profile) if r.complete else None,
            }
        )
        return r

    def each(items, workers, stages, chooses=True, img=None, plane=None):
        """``img``: the image fitted (the image as given by default), ``plane`` the sky
        tilt taken out of it."""
        img = image if img is None else img

        def run(j):
            return one(j[0], img=img, stage=j[1], number=j[2], chooses=chooses, plane=plane)

        jobs = [(x, lbl, next(order)) for x, lbl in zip(items, stages)]
        if workers <= 1 or len(jobs) <= 1:
            return [run(j) for j in jobs]
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            return list(pool.map(run, jobs))

    def cached(trials, stage):
        """Fits with a fixed sky, each made once (the same settings and sky: reused)."""
        new, seen = [], set()
        for v in trials:
            if _key(v) not in fitted and _key(v) not in seen:
                new.append(v)
                seen.add(_key(v))
        for v, r in zip(new, each(new, s.parallel, [stage(v) for v in new])):
            fitted[_key(v)] = r
        return [fitted[_key(v)] for v in trials], len(trials) - len(new)

    def lower(r, than) -> bool:
        return r.complete and r.score is not None and (than.score is None or r.score < than.score)

    def search(base: SingleSettings, tag: str = "", wide: bool = False):
        """Every TRYFIX on ``base`` (and, ``wide``, TIE off and a larger R0): the fit with
        the lowest residual, its settings, the rows."""
        variants = []
        r0s = [(base.r0, "")]
        if wide:
            core_r = masked_core(mask, (base.x0, base.y0), 0.25 * base.r1)
            r0_out = round(max(R0_OUT * base.r0, R0_MARGIN * core_r), 1)
            if r0_out < 0.25 * base.r1 and r0_out > base.r0 + 0.5:
                why = f"out of the masked core ({core_r:g} px)" if core_r > base.r0 else ""
                r0s.append((r0_out, f"R0 {r0_out:g}" + (f" {why}" if why else "")))
        ties = [(base.tie, "")] + ([(None, "TIE off")] if wide and base.tie is not None else [])
        for fix in base.try_fix or [None]:
            for tie, tie_label in ties:
                for r0, r0_label in r0s:
                    v, words = replace(base, tie=tie, r0=r0), []
                    if fix is not None:
                        v = replace(v, fixctr=fix)
                        words.append("centres " + {0: "free", 1: "fixed", 2: "median"}[fix])
                    words += [w for w in (tie_label, r0_label) if w]
                    variants.append((", ".join(words) or "as asked", v))
        say({"phase": "stage", "text": f"search{tag}: {len(variants)} fit(s)"})
        results = each([v for _l, v in variants], s.parallel, ["search" + tag] * len(variants))
        rows_ = [
            {"variant": lbl, "status": r.status, "score": r.score, "r1": r.r1}
            for (lbl, _v), r in zip(variants, results)
        ]
        done = [(r.score, i) for i, r in enumerate(results) if r.complete and r.score is not None]
        if not done:
            return results[0], variants[0][1], rows_
        best_i = min(done)[1]  # the lowest residual (a tie: the first asked)
        if len(rows_) > 1:
            rows_[best_i]["chosen"] = True
        return results[best_i], variants[best_i][1], rows_

    def fixed_sky(v: SingleSettings, r: SingleResult) -> SingleSettings:
        return replace(v, sky=r.settings.sky, sky_factor=r.settings.sky_factor, sky_iter=0)

    kept, kept_s, rows = search(s, wide=s.wide)
    if not kept.complete:
        kept.seconds = time.perf_counter() - t0
        kept.calls = counter["n"]
        return kept
    steps = list(kept.steps)
    chosen = next((x["variant"] for x in rows if x.get("chosen")), None)
    if chosen and chosen != "as asked":
        steps.append(f"search: {chosen} has the lowest residual ({kept.score:.3f})")

    # SEARCH=FULL: the fixed centre, moved by a fraction of a pixel (the light's centre is
    # rarely where it was clicked): the 8 positions around it, the lowest residual taken,
    # the step halved when none is lower
    if s.wide and kept_s.fixctr == 1 and not stopped():
        base = fixed_sky(kept_s, kept)
        fitted[_key(base)] = kept
        here, step, made = (kept_s.x0, kept_s.y0), CENTRE_STEP, 0
        while step >= CENTRE_LAST - 1e-9 and made < CENTRE_FITS and not stopped():
            trials = [
                replace(base, x0=round(here[0] + dx * step, 4), y0=round(here[1] + dy * step, 4))
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                if dx or dy
            ]
            say({"phase": "stage", "text": f"centre: 8 positions {step:g} px around it"})
            got, reused = cached(trials, lambda v: f"centre {v.x0:.3f}, {v.y0:.3f}")
            made += len(trials) - reused
            best = min((r for r in got if lower(r, kept)), key=lambda r: r.score, default=None)
            if best is None:
                step /= 2
                continue
            i = got.index(best)
            here = (trials[i].x0, trials[i].y0)
            kept, kept_s = best, replace(kept_s, x0=here[0], y0=here[1])
        if (kept_s.x0, kept_s.y0) != (s.x0, s.y0):
            steps.append(
                f"centre: X0={kept_s.x0:.3f} Y0={kept_s.y0:.3f} has the lowest residual "
                f"({kept.score:.3f})"
            )

    harmonic_rows: List[dict] = []
    # the harmonics, iteratively: every COS3X and COS4X asked (all with HARMONICS=AUTO) on
    # the fit kept, its sky kept; when another combination lowers the residual, the search
    # is made again with it, until the harmonics stay (at most HARMONIC_ROUNDS times).  A
    # combination whose ellipses cross is fitted only out to where they do not (the checks).
    found3, found4 = pair.harmonic_values(s.harmonic_search)
    c3s = s.try_cos3x or found3
    c4s = s.try_cos4x or found4
    for rnd in range(1, HARMONIC_ROUNDS + 1):
        if not (c3s or c4s) or stopped():
            break
        own = (
            kept_s.cos3x if kept_s.cos3x is not None else 2,
            kept_s.cos4x if kept_s.cos4x is not None else 2,
        )
        combos = [(a, b) for a in (c3s or [own[0]]) for b in (c4s or [own[1]]) if (a, b) != own]
        base = fixed_sky(kept_s, kept)
        trials = [replace(base, cos3x=a, cos4x=b) for a, b in combos]
        new = sum(_key(v) not in fitted for v in trials)
        say(
            {
                "phase": "stage",
                "text": f"harmonics round {rnd}: {new} combinations"
                + (f" ({len(trials) - new} fitted before)" if new < len(trials) else ""),
            }
        )
        tried, _reused = cached(trials, lambda v, rnd=rnd: f"harmonics {rnd}")
        harmonic_rows.append(
            {
                "round": rnd,
                "cos3x": own[0],
                "cos4x": own[1],
                "score": kept.score,
                "r1": kept.r1,
                "status": kept.status,
                "kept": True,
            }
        )
        for (a, b), r in zip(combos, tried):
            harmonic_rows.append(
                {
                    "round": rnd,
                    "cos3x": a,
                    "cos4x": b,
                    "score": r.score,
                    "r1": r.r1,
                    "status": r.status,
                    "short": r.r1 < kept.r1 - 0.5,
                    "kept": False,
                }
            )
        # the smallest residual; on a tie, the combination closest to the one it has
        good = [
            (r.score, abs(combos[i][0] - own[0]) + abs(combos[i][1] - own[1]), i)
            for i, r in enumerate(tried)
            if r.complete and r.score is not None
        ]
        if not good or min(good)[0] >= kept.score:
            steps.append(
                f"harmonics (round {rnd}): COS3X={own[0]} COS4X={own[1]} kept; no other of "
                f"{len(combos)} has a lower residual"
            )
            break
        i = min(good)[2]
        a, b = combos[i]
        for row in harmonic_rows[-len(combos) - 1 :]:
            row["kept"] = row["cos3x"] == a and row["cos4x"] == b
        steps.append(
            f"harmonics (round {rnd}): COS3X={a} COS4X={b} lowered the residual "
            f"({kept.score:.3f} -> {tried[i].score:.3f}); the search again with them"
        )
        again, again_s, again_rows = search(replace(kept_s, cos3x=a, cos4x=b), " again")
        rows += again_rows
        if again.complete and again.score is not None and again.score <= tried[i].score:
            kept, kept_s = again, again_s
        else:
            kept, kept_s = tried[i], replace(kept_s, cos3x=a, cos4x=b)
    # SEARCH=FULL with NR=AUTO: fewer and more isophotes
    if s.wide and s.nr == 0 and kept.nr and not stopped():
        counts = sorted({max(4, int(round(kept.nr * f))) for f in NR_FACTORS} - {kept.nr})
        base = fixed_sky(kept_s, kept)
        say({"phase": "stage", "text": f"isophotes: NR {', '.join(map(str, counts))}"})
        got, _reused = cached(
            [replace(base, nr=k) for k in counts], lambda v: f"isophotes NR {v.nr}"
        )
        best = min((r for r in got if lower(r, kept)), key=lambda r: r.score, default=None)
        if best is not None:
            k = counts[got.index(best)]
            steps.append(f"isophotes: NR {k} has the lowest residual ({best.score:.3f})")
            kept, kept_s = best, replace(kept_s, nr=k)
    # SKYPLANE: a tilt of the sky in the residual, taken out, and the fit again (kept if
    # the residual is lower)
    sky_map = kept_plane = None
    if s.sky_plane and kept.complete and not stopped():
        tilt = _tilt(kept.residual, mask)
        if tilt is not None:
            plane, words = tilt
            say({"phase": "stage", "text": "the sky tilt taken out: fitted again"})
            one(kept_s, mask, image - plane, stage="sky tilt out", plane=plane)
            steps.append(words + ": taken out of the image and fitted again")
    # the cores: one far worse than the cleanest of the models compared is high-contrast
    compared = [
        m for m in models if m["chooses"] and m["result"].complete and m["result"].score is not None
    ]
    cores = [m["result"].core for m in compared if m["result"].core is not None]
    cleanest = min(cores) if cores else None

    def flag(m) -> str:
        c = m["result"].core
        if cleanest is not None and c is not None and c > max(CORE_FLOOR, CORE_LIMIT * cleanest):
            return "core"
        return ""

    for m in compared:
        m["flag"] = flag(m)
    # the model with the lowest residual of all those compared (search, centre, harmonics,
    # isophotes, tilt), a high-contrast core never
    pool_ = [m for m in compared if not m["flag"]] or compared
    if pool_:
        # on a tie, the model the search has
        best = min(pool_, key=lambda m: (m["result"].score, m["result"] is not kept, m["n"]))
        if best["result"] is not kept:
            steps.append(
                f"the lowest residual of {len(pool_)} models: #{best['n']} {best['stage']}"
            )
        kept = best["result"]
        # its settings, with the sky as asked (the averaged fits refine their own)
        kept_s = replace(best["settings"], sky=s.sky, sky_factor=s.sky_factor, sky_iter=s.sky_iter)
        if best["plane"] is not None:
            # the averaged fits are made on the same image, the tilt taken out
            kept_plane = best["plane"]
            image = image - kept_plane
            sky_map = best["settings"].sky + kept_plane
            steps.append("the fit on the image with the sky tilt taken out kept")
    flagged = sum(bool(m["flag"]) for m in compared)
    if flagged:
        steps.append(
            f"{flagged} model(s) with a high-contrast core (its residual over {CORE_LIMIT:g} x "
            f"the cleanest, {cleanest:.2f}): never kept nor averaged"
        )
    # RESIDMASK: the faint sources of the residual, masked, and the fit kept again
    if s.resid_rounds and kept.complete and image_obj is not None:
        from . import detection

        added = np.zeros(mask.shape, bool)
        for k in range(1, s.resid_rounds + 1):
            if stopped():
                break
            run, add = detection.residual_sources(
                image_obj,
                kept.residual,
                kept.mask,
                [centre],
                number=k,
                thresh=s.resid_thresh,
                grow=s.resid_grow,
                core=s.resid_core,
            )
            kept.resid_runs = kept.resid_runs + [run]
            runs_so_far = kept.resid_runs
            if not add.any():
                break
            added |= add
            say({"phase": "stage", "text": f"residual sources round {k}: fitted again"})
            again = one(
                replace(kept_s, sky=kept.settings.sky, sky_iter=0),
                kept.mask | add,
                image,
                stage=f"residual sources {k}",
                chooses=False,
            )
            if not again.complete:
                steps.append(f"residual sources round {k}: the refit was incomplete; not used")
                break
            steps.append(
                f"residual sources round {k}: {len(run)} found, {int(add.sum())} px masked"
            )
            again.resid_runs = runs_so_far
            kept = again
        kept.resid_mask = added if added.any() else None
    # PARALLEL / RUNS: N fits averaged, the fit kept and N - 1 with R0 and R1 varied (R0
    # never into a masked core); one much worse than the fit kept is left out
    n_avg = s.runs if s.runs > 1 else s.parallel
    if n_avg > 1 and kept.complete and not stopped():
        rng = np.random.default_rng(s.seed if s.seed is not None else 0)
        floor = R0_MARGIN * masked_core(mask, (kept_s.x0, kept_s.y0), 0.25 * kept_s.r1)
        if centre_masked(mask, s.x0, s.y0):  # the R0 elliprof could fit there (clear_centre)
            floor = max(floor, s.r0)
        jittered = []
        for k in range(2, n_avg + 1):
            f0, f1 = (
                1 + rng.choice((-1.0, 1.0)) * rng.uniform(2 * s.jitter / 3, s.jitter)
                for _ in range(2)
            )
            r1 = kept_s.r1 * f1
            r0 = kept_s.r0 * f0
            if r0 < floor:
                r0 = kept_s.r0 * (2 - f0)  # outward instead
            jittered.append(replace(kept_s, r0=round(min(r0, 0.5 * r1), 2), r1=round(r1, 2)))
        say({"phase": "stage", "text": f"averaging: {n_avg - 1} more fits, R0 and R1 varied"})
        more = each(
            jittered,
            s.parallel,
            [f"average {k}/{n_avg}" for k in range(2, n_avg + 1)],
            chooses=False,
            img=image,
            plane=kept_plane,
        )
        members, left = [("the fit kept", kept)], []
        limit = (1 + AVERAGE_TOL) * kept.score if kept.score is not None else None
        for j, r in zip(jittered, more):
            m = next(x for x in models if x["result"] is r)
            if not r.complete:
                continue
            m["flag"] = flag(m)
            if m["flag"]:
                why = "a high-contrast core"
            elif r.r1 < AVERAGE_REACH * kept.r1:
                why = f"fitted only to {r.r1:.1f}"
            elif limit is not None and (r.score is None or r.score > limit):
                why = "residual " + _number(r.score)
            else:
                m["averaged"] = True
                members.append((f"R0 {j.r0:g} R1 {j.r1:g}", r))
                continue
            m["flag"] = m["flag"] or "poor"
            left.append(f"#{m['n']} ({why})")
        if left:
            steps.append(
                f"averaging: {', '.join(left)} left out (a residual over {1 + AVERAGE_TOL:g} x "
                f"the kept's, R1 short of {AVERAGE_REACH:g} x its, or a high-contrast core)"
            )
        if len(members) > 1:
            kept.ensemble = _mean(members, image, n_avg)
            kept.ensemble["plane"] = kept_plane  # the sky tilt taken out, or None
            for f, (_lbl, r) in zip(kept.ensemble["fits"], members):
                f["n"] = next(x["n"] for x in models if x["result"] is r)
    final = next((m for m in models if m["result"] is kept), None)
    if final is not None and kept.ensemble:
        final["averaged"] = True
    kept.search = rows if len(rows) > 1 else []
    kept.harmonics = harmonic_rows
    kept.models = sorted(
        ({k: v for k, v in m.items() if k not in ("settings", "result", "plane")} for m in models),
        key=lambda m: m["n"],
    )
    kept.model_store = {
        m["n"]: {
            "model": m["result"].model.astype(np.float32),
            "sky": float(m["result"].settings.sky),
            "plane": m["plane"],
            "mask": m["result"].mask,
        }
        for m in models
        if m["result"].complete
    }
    for m in kept.models:
        m["kept"] = final is not None and m["n"] == final["n"]
        m.setdefault("averaged", False)
    if final is not None:
        say({"phase": "chosen", "n": final["n"], "rows": kept.models})
    kept.harmonic_choice = (
        kept_s.cos3x if kept_s.cos3x is not None else 2,
        kept_s.cos4x if kept_s.cos4x is not None else 2,
    )
    kept.steps = steps
    kept.sky_map = sky_map
    kept.rings = rings
    kept.calls = counter["n"]
    kept.seconds = time.perf_counter() - t0
    return kept


def _tilt(residual, mask) -> Optional[Tuple[np.ndarray, str]]:
    """A plane through the residual's boxes, when its gradient is 3 times its error."""
    from . import sky as skies

    good = ~mask & np.isfinite(residual)
    values = residual[good]
    if values.size < 100:
        return None
    sigma = skies.mad(values)
    good &= np.abs(residual - np.median(values)) < 5 * sigma
    fitted = skies.plane_by_boxes(residual, good, sigma, pair.PLANE_BOX)
    if fitted is None:
        return None
    plane, errors = fitted[0], fitted[1]
    g = math.hypot(plane["gx"], plane["gy"])
    err = math.hypot(errors["gx"], errors["gy"]) / math.sqrt(2) or float("inf")
    words = (
        f"sky tilt {1000 * plane['gx']:+.3g}, {1000 * plane['gy']:+.3g} per 1000 px "
        f"(± {1000 * errors['gx']:.2g}, {1000 * errors['gy']:.2g})"
    )
    if g < 3 * err:
        return None
    return skies.plane_image(dict(plane, s0=0.0), residual.shape), words


def _mean(members, image, of: int) -> dict:
    """The mean of the fits ``members`` [(label, result)]: the mean model and sky, and the
    mean residual image - mean sky - mean model; each fit's own model and residual."""
    models = np.stack([r.model for _lbl, r in members])
    skies_ = np.array([r.settings.sky for _lbl, r in members])
    model = models.mean(axis=0)
    return {
        "members": len(members),
        "of": of,
        "model": model,
        "sky": float(skies_.mean()),
        "sky_spread": float(skies_.std(ddof=1)),
        "residual": image - skies_.mean() - model,
        "spread": models.std(axis=0, ddof=1),
        "fits": [
            {
                "label": lbl,
                "sky": float(r.settings.sky),
                "model": r.model,
                "residual": image - r.settings.sky - r.model,
                "score": r.score,
            }
            for lbl, r in members
        ],
    }


def largest_crossing(profile) -> Optional[float]:
    """How far, at most, an ellipse reaches out of the next one (1: touching)."""
    ell = pair.profile_ellipses(profile)
    if len(ell) < 2:
        return None
    return max(pair.crossing(ell[k], ell[k + 1]) for k in range(len(ell) - 1))


CATEGORIES = [  # (key, the steps' names start with, label)
    ("search", "search", "Search"),
    ("centre", "centre", "Centre"),
    ("harmonics", "harmonics", "Harmonics"),
    ("isophotes", "isophotes", "Isophotes (NR)"),
    ("tilt", "sky tilt", "Sky tilt"),
    ("residual", "residual sources", "Residual sources"),
    ("average", "average", "Averaged"),
]


def category(stage: str) -> str:
    """The category of a model's step (CATEGORIES' key)."""
    for key, prefix, _label in CATEGORIES:
        if stage.startswith(prefix):
            return key
    return "other"


def average_models(image, store: Dict[int, dict], numbers, rings: Rings) -> dict:
    """The mean of the models ``numbers``: mean model, mean sky (with any tilt), residual =
    image - mean sky - mean model (NaN where any of them was masked), and that residual's
    score and core on ``rings``."""
    picked = [store[k] for k in numbers]
    model = np.mean([p["model"] for p in picked], axis=0, dtype=np.float64)
    sky = np.mean(
        [p["sky"] + (p["plane"] if p["plane"] is not None else 0.0) for p in picked], axis=0
    )
    masked = np.zeros(image.shape, bool)
    for p in picked:
        if p["mask"] is not None:
            masked |= p["mask"]
    residual = image - sky - model
    got = rings.measure(residual, masked) or {}
    return {
        "model": model,
        "sky": sky,
        "residual": np.where(masked, np.nan, residual),
        "score": got.get("score"),
        "core": got.get("core"),
    }


def _number(v, fmt=".3f") -> str:
    return "-" if v is None else format(v, fmt)


def _coord(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def command_for(row: dict, buffer: int = 1) -> str:
    """The elliprof command that makes the model of ``row`` again (one call, its sky)."""
    words = [f"elliprof {buffer}"]
    if row.get("x0") is not None and row.get("y0") is not None:
        words += [f"X0={_coord(row['x0'])}", f"Y0={_coord(row['y0'])}"]
    words += [
        f"R0={row['r0']:g}",
        f"R1={row['r1_reached']:.1f}",
        f"NR={row['nr']}",
        f"FIXCTR={row['fixctr']}",
        f"COS3X={row['cos3x']}",
        f"COS4X={row['cos4x']}",
    ]
    if row.get("tie") is not None:
        words.append(f"TIE={row['tie']}")
    if row.get("niter"):
        words.append(f"NITER={row['niter']}")
    words.append(f"SKY={row['sky']:.10g}")
    return " ".join(words)


def describe(row: dict, buffer: int = 1) -> str:
    """A model's parameters and its residual, in words (the models table's right click)."""
    if row.get("kept"):
        status = "kept (the lowest residual)"
    else:
        status = "compared" if row.get("chooses", True) else "not compared (listed only)"
    flag = {
        "core": "a high-contrast core: never kept nor averaged",
        "poor": "left out of the mean",
    }
    inner, outer = row.get("inner"), row.get("outer")
    where = (
        f" (RMS over the noise, rings {inner:.1f} to {outer:.1f} px; the core: out to "
        f"{CORE_SPAN:g} x {inner:.1f} px)"
        if inner is not None and outer is not None
        else ""
    )
    centre = (
        f"X0={row['x0']:.3f} Y0={row['y0']:.3f}, "
        if row.get("x0") is not None and row.get("y0") is not None
        else ""
    )
    lines = [
        f"model #{row['n']}: {row['stage']}",
        f"  {status}"
        + (f"; {flag[row['flag']]}" if row.get("flag") in flag else "")
        + ("; in the mean" if row.get("averaged") else ""),
        f"  residual {_number(row.get('score'))}, core {_number(row.get('core'))}{where}",
        f"  centre {centre}FIXCTR={row['fixctr']} ("
        + {0: "free centres", 1: "fixed", 2: "their median"}.get(row["fixctr"], "?")
        + ")",
        f"  R0 {row['r0']:g}, R1 {row['r1']:.1f} asked, {row['r1_reached']:.1f} reached; "
        f"NR {row['nr']}",
        f"  COS3X={row['cos3x']} COS4X={row['cos4x']}, TIE="
        + ("off" if row.get("tie") is None else str(row["tie"]))
        + (f", NITER={row['niter']}" if row.get("niter") else ""),
        f"  sky {row['sky']:.8g}" + (", its tilt taken out" if row.get("tilt") else ""),
        f"  status: {row['status']}",
        "",
        "the elliprof command for this model (one call, its sky):",
        "  " + command_for(row, buffer),
    ]
    return "\n".join(lines)


def models_table(models: List[dict]) -> List[str]:
    """Every model made: its parameters and its residual, the lowest kept."""
    if not models:
        return []
    out = [
        f"models (residual: RMS over the noise in rings from {models[0]['inner']:.1f} to "
        f"{models[0]['outer']:.1f} px, each ring counted the same; core: the rings out to "
        f"{CORE_SPAN:g} x {models[0]['inner']:.1f} px; the lowest of those compared is kept, "
        "a high-contrast core never; the averaged fits and refits after it are listed only):",
        f"  {'#':>3} {'step':<30} {'X0':>8} {'Y0':>8} {'FIX':>3} {'C3':>2} {'C4':>2} "
        f"{'TIE':>3} {'R0':>5} {'R1':>6} {'reached':>7} {'NR':>3} {'sky':>9} {'tilt':>4} "
        f"{'core':>6} {'residual':>8}",
    ]
    for m in models:
        sc = _number(m["score"])
        if m["status"] != "fitted":
            sc = m["status"]
        if m.get("kept"):
            mark = "  <- kept"
        elif m.get("flag") == "core":
            mark = "  (high-contrast core)"
        elif m.get("flag") == "poor":
            mark = "  (left out of the mean)"
        elif not m["chooses"]:
            mark = "  (in the mean)" if m.get("averaged") else "  (not compared)"
        else:
            mark = ""
        tie = "off" if m.get("tie") is None else str(m["tie"])
        out.append(
            f"  {m['n']:>3} {m['stage'][:30]:<30} {m['x0']:>8.3f} {m['y0']:>8.3f} "
            f"{m['fixctr']:>3} {m['cos3x']:>2} {m['cos4x']:>2} {tie:>3} {m['r0']:>5.1f} "
            f"{m['r1']:>6.1f} {m['r1_reached']:>7.1f} {m['nr']:>3} {m['sky']:>9.6g} "
            f"{'yes' if m['tilt'] else 'no':>4} {_number(m.get('core'), '.2f'):>6} {sc:>8}{mark}"
        )
    return out


def summary(result: SingleResult) -> str:
    s = result.settings
    lines = [
        f"{result.status}: R1 {result.r1:.1f} (asked {s.r1:g}), NR {result.nr}, sky "
        f"{s.sky:.6g}"
        + (f" (factor {s.sky_factor:.5f})" if result.sky_iterations else "")
        + (f"; residual {result.score:.3f}" if result.score is not None else "")
        + (f", core {result.core:.2f}" if result.core is not None else "")
    ]
    if result.reason:
        lines.append(result.reason)
    for e in result.sky_iterations:
        med = e.get("median")
        lines.append(
            f"  sky {e['iteration']}: factor {e['factor']:.5f}"
            + (f", residual sky {med:+.4g}" if med is not None else "")
            + ("  <- kept" if e.get("chosen") else "")
            + (f"; {e['note']}" if e.get("note") else "")
        )
    lines += [f"  {x}" for x in result.steps]
    lines += models_table(result.models)
    if result.harmonics:
        lines.append(
            f"harmonics kept: COS3X={result.harmonic_choice[0]} COS4X={result.harmonic_choice[1]}"
        )
    if result.profile is not None:
        worst = largest_crossing(result.profile)
        if worst is not None:
            lines.append(
                f"ellipses: none cross (an inner one reaches at most {worst:.3f} of the next, "
                f"{pair.CROSSING:g} allowed)"
            )
    e = result.ensemble
    if e:
        lines.append(
            f"mean of {e['members']} fits (the fit kept and {e['members'] - 1} of {e['of'] - 1} "
            f"with R0 and R1 varied, each with its own sky): mean sky {e['sky']:.6g} ± "
            f"{e['sky_spread']:.2g}; mean residual = image - mean sky - mean model"
        )
    lines.append(f"{result.calls} elliprof calls in {result.seconds:.1f} s")
    return "\n".join(lines)
