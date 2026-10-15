"""Iterative fits of one galaxy: the pair fit's options for a single ELLIPROF fit.

A single galaxy has no companion to take turns with, so its iterations are these: R1 grows
from a first radius (START, by FACTOR) and the fit keeps the largest R1 at which the
isophotes pass the pair fit's checks (a step that fails is tried again halfway); the
outermost isophotes are checked against the trend inside them (SHAPETOL); the sky factor is
refined from the residual sky (SKYITER) and a tilt of the sky taken out (SKYPLANE); a search
over centre fitting and growth factor keeps the fit with the smallest residual (TRYFIX,
TRYFACTOR) and then the harmonics (HARMONICS=AUTO, TRYCOS3X, TRYCOS4X); the faint sources
of the residual are masked and the galaxy fitted again (RESIDMASK); and N fits, the fit kept
and N - 1 with R0 and R1 varied, are averaged (PARALLEL).  Every elliprof call goes through
fitting.fit_array, as the pair fit's.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import pair
from .fitting import fit_array


@dataclass
class SingleSettings:
    x0: float
    y0: float
    r0: float = 3.0
    r1: float = 100.0
    nr: int = 0  # 0: automatic (isophotes about 30 % apart)
    start: Optional[float] = None  # the first R1 (None: no growth)
    factor: float = pair.AUTO_FACTOR  # R1 grows by this each step
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
    try_factor: List[float] = field(default_factory=list)
    try_cos3x: List[int] = field(default_factory=list)
    try_cos4x: List[int] = field(default_factory=list)
    harmonic_search: bool = False
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
    sky_iterations: List[dict] = field(default_factory=list)
    search: List[dict] = field(default_factory=list)
    ensemble: dict = field(default_factory=dict)
    resid_runs: List[object] = field(default_factory=list)
    resid_mask: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None  # the fitting mask used
    harmonics: List[dict] = field(default_factory=list)  # the harmonic analysis, every try
    harmonic_choice: Tuple[int, int] = (2, 2)  # COS3X, COS4X of the fit kept

    @property
    def complete(self) -> bool:
        return self.profile is not None and self.model is not None


Progress = Optional[Callable[[dict], None]]
Cancelled = Optional[Callable[[], bool]]


HARMONIC_ROUNDS = 3  # the harmonics and the search, alternated at most this many times


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


def grow(
    image: np.ndarray,
    mask: np.ndarray,
    s: SingleSettings,
    header=None,
    progress: Progress = None,
    cancelled: Cancelled = None,
    fitter=fit_array,
    counter: Optional[Dict[str, int]] = None,
) -> SingleResult:
    """One fit of the galaxy: R1 grown from START by FACTOR (else R1 at once), each step
    checked; the largest R1 that passes is kept, then the outer-shape guard."""
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
            if progress is not None:
                progress({"phase": "fit", "r1": r1, "call": counter["n"]})
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
                        "r1": r1,
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
        schedule = [s.r1]
        if s.start and s.start < s.r1 - 1:
            schedule = [s.start]
            while schedule[-1] * s.factor < s.r1 - 1e-9:
                schedule.append(schedule[-1] * s.factor)
            schedule.append(s.r1)
        best = None  # (fit, r1, nr)
        for k, r in enumerate(schedule):
            have = best[1] if best else 0.0
            got = call(r, floor=have)
            if got[0] is None and best is not None and r > have + 0.5:
                half = have + 0.5 * (r - have)
                got = call(half, floor=have)
                if got[0] is not None:
                    out.steps.append(f"R1 {r:.1f} failed ({got[1]}); it grew to {half:.1f} instead")
            if got[0] is None:
                if best is None:
                    out.reason = f"no fit passed: {got[1]}"
                    out.seconds = time.perf_counter() - start
                    out.calls = counter["n"]
                    return out
                out.steps.append(f"R1 {r:.1f} failed ({got[1]}); it stays at {have:.1f}")
                break
            if got[1] < r - 0.5:
                out.steps.append(f"R1 {r:.1f}: fitted to {got[1]:.1f}")
            best = got
            if got[1] < r - 0.5:
                break  # it could not reach this step: no further growth
        fit, r1, nr = best
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


def score(result: SingleResult, mask: np.ndarray, centre, reach: float) -> Optional[float]:
    """How well the model describes the galaxy: the RMS of the residual (clipped at 10
    sigma) within ``reach`` px of the centre, beyond 10 px from it, over its local noise."""
    from . import sky as skies

    if result.residual is None:
        return None
    ny, nx = result.residual.shape
    yy, xx = np.mgrid[0:ny, 0:nx] + 0.5
    r = np.hypot(xx - centre[0], yy - centre[1])
    good = ~mask & np.isfinite(result.residual) & (r > 10) & (r < reach)
    if good.sum() < 50:
        return None
    sigma = skies.local_noise(result.residual, good)
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    v = np.clip(result.residual[good], -10 * sigma, 10 * sigma)
    return float(np.sqrt(np.mean(v * v)) / sigma)


def with_sky(image, mask, s, support, header, progress, cancelled, fitter, counter):
    """SKYITER: the galaxy fitted up to SKYITER times, the sky factor updated each time from
    the residual on the sky support (f + median / base), as the pair fit does; the fit with
    the smallest |median| is kept.  Stops when the residual does not follow the sky (a halo
    that fills the frame)."""
    from . import sky as skies

    if s.sky_iter <= 0 or support is None:
        return grow(image, mask, s, header, progress, cancelled, fitter, counter)
    base = s.sky_base if s.sky_base is not None else s.sky
    use = support & ~mask
    factor = s.sky_factor
    history, best, previous, first = [], None, None, None
    for k in range(s.sky_iter):
        trial = replace(s, sky=factor * base, sky_factor=factor)
        result = grow(image, mask, trial, header, progress, cancelled, fitter, counter)
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
    reach = 1.1 * s.r1

    def one(settings, m=mask, img=image):
        r = with_sky(img, m, settings, support, header, progress, cancelled, fitter, counter)
        r.mask = m
        r.score = score(r, m, centre, reach)
        return r

    def each(items, workers):
        if workers <= 1 or len(items) <= 1:
            return [one(x) for x in items]
        with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
            return list(pool.map(one, items))

    def search(base: SingleSettings, tag: str = ""):
        """Every combination of TRYFIX and TRYFACTOR on ``base``: the fit kept (fixed centres
        unless free ones lower the residual by more than FREE_GAIN), its settings, the rows."""
        variants = []
        for fix in base.try_fix or [None]:
            for factor in base.try_factor or [None]:
                label = []
                v = base
                if fix is not None:
                    v = replace(v, fixctr=fix)
                    label.append("centres " + {0: "free", 1: "fixed", 2: "median"}[fix])
                if factor is not None and base.start:
                    v = replace(v, factor=factor)
                    label.append(f"growth x{factor:g}")
                variants.append(((", ".join(label) or "as asked") + tag, v))
        results = each([v for _l, v in variants], s.parallel)
        rows_ = [
            {"variant": lbl, "status": r.status, "score": r.score, "r1": r.r1}
            for (lbl, _v), r in zip(variants, results)
        ]
        done = [(r.score, i) for i, r in enumerate(results) if r.complete and r.score is not None]
        if not done:
            return results[0], variants[0][1], rows_
        best_i = min(done)[1]
        fixed = [(sc, i) for sc, i in done if (variants[i][1].fixctr or 0) != 0]
        if fixed and (variants[best_i][1].fixctr or 0) == 0:
            f_sc, f_i = min(fixed)
            if results[best_i].score > f_sc - pair.FREE_GAIN:
                best_i = f_i
        if len(rows_) > 1:
            rows_[best_i]["chosen"] = True
        return results[best_i], variants[best_i][1], rows_

    kept, kept_s, rows = search(s)
    if not kept.complete:
        kept.seconds = time.perf_counter() - t0
        kept.calls = counter["n"]
        return kept
    steps = list(kept.steps)
    harmonic_rows: List[dict] = []
    # the harmonics, iteratively: every COS3X and COS4X asked (all with HARMONICS=AUTO) on
    # the fit kept, its sky kept; when another combination lowers the residual, the search
    # is made again with it, until the harmonics stay (at most HARMONIC_ROUNDS times).  A
    # combination whose ellipses cross is fitted only out to where they do not (the checks).
    c3s = s.try_cos3x or (list(pair.COS3X_VALUES) if s.harmonic_search else [])
    c4s = s.try_cos4x or (list(pair.COS4X_VALUES) if s.harmonic_search else [])
    for rnd in range(1, HARMONIC_ROUNDS + 1):
        if not (c3s or c4s) or stopped():
            break
        own = (
            kept_s.cos3x if kept_s.cos3x is not None else 2,
            kept_s.cos4x if kept_s.cos4x is not None else 2,
        )
        combos = [(a, b) for a in (c3s or [own[0]]) for b in (c4s or [own[1]]) if (a, b) != own]
        fixed_sky = replace(
            kept_s, sky=kept.settings.sky, sky_factor=kept.settings.sky_factor, sky_iter=0
        )
        tried = each([replace(fixed_sky, cos3x=a, cos4x=b) for a, b in combos], s.parallel)
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
                    "short": r.r1 < s.r1 - 0.5,
                    "kept": False,
                }
            )
        # the smallest residual; on a tie, the combination closest to the one it has
        good = [
            (r.score, abs(combos[i][0] - own[0]) + abs(combos[i][1] - own[1]), i)
            for i, r in enumerate(tried)
            if r.complete and r.score is not None
        ]
        if not good or min(good)[0] >= kept.score - pair.HARMONIC_GAIN:
            steps.append(
                f"harmonics (round {rnd}): COS3X={own[0]} COS4X={own[1]} kept; no other of "
                f"{len(combos)} lowered the residual by more than {pair.HARMONIC_GAIN:g}"
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
        again, again_s, again_rows = search(replace(s, cos3x=a, cos4x=b), f" (COS3X={a} COS4X={b})")
        rows += again_rows
        if again.complete and again.score is not None and again.score <= tried[i].score:
            kept, kept_s = again, again_s
        else:
            kept, kept_s = tried[i], replace(kept_s, cos3x=a, cos4x=b)
    # SKYPLANE: a tilt of the sky in the residual, taken out, and the fit again (kept if
    # the residual is lower)
    sky_map = None
    if s.sky_plane and kept.complete and not stopped():
        tilt = _tilt(kept.residual, mask)
        if tilt is not None:
            plane, words = tilt
            again = one(kept_s, mask, image - plane)
            if again.complete and again.score is not None and again.score < kept.score:
                kept, image = again, image - plane
                sky_map = kept_s.sky + plane
                steps.append(words + ": taken out of the image; the fit on it kept")
            else:
                steps.append(words + ": the fit without it kept")
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
            again = one(replace(kept_s, sky=kept.settings.sky, sky_iter=0), kept.mask | add, image)
            if not again.complete:
                steps.append(f"residual sources round {k}: the refit was incomplete; not used")
                break
            steps.append(
                f"residual sources round {k}: {len(run)} found, {int(add.sum())} px masked"
            )
            again.resid_runs = runs_so_far
            kept = again
        kept.resid_mask = added if added.any() else None
    # PARALLEL / RUNS: N fits averaged, the fit kept and N - 1 with R0 and R1 varied
    n_avg = s.runs if s.runs > 1 else s.parallel
    if n_avg > 1 and kept.complete and not stopped():
        rng = np.random.default_rng(s.seed if s.seed is not None else 0)
        jittered = []
        for k in range(2, n_avg + 1):
            f0, f1 = (
                1 + rng.choice((-1.0, 1.0)) * rng.uniform(2 * s.jitter / 3, s.jitter)
                for _ in range(2)
            )
            r1 = kept_s.r1 * f1
            r0 = min(kept_s.r0 * f0, 0.5 * r1)
            start = kept_s.start * f1 if kept_s.start else None
            jittered.append(replace(kept_s, r0=round(r0, 2), r1=round(r1, 2), start=start))
        more = each(jittered, s.parallel)
        members = [kept] + [r for r in more if r.complete]
        if len(members) > 1:
            kept.ensemble = _mean(members, image, mask, kept_s, jittered, more)
    kept.search = rows if len(rows) > 1 else []
    kept.harmonics = harmonic_rows
    kept.harmonic_choice = (
        kept_s.cos3x if kept_s.cos3x is not None else 2,
        kept_s.cos4x if kept_s.cos4x is not None else 2,
    )
    kept.steps = steps
    kept.sky_map = sky_map
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


def _mean(members, image, mask, kept_s, jittered, more) -> dict:
    """The mean of the fits: the mean model and sky, and the mean residual image - mean sky
    - mean model; each fit's own model and residual."""
    models = np.stack([m.model for m in members])
    skies_ = np.array([m.settings.sky for m in members])
    model = models.mean(axis=0)
    labels = ["the fit kept"] + [
        f"R0 {j.r0:g} R1 {j.r1:g}" for j, r in zip(jittered, more) if r.complete
    ]
    return {
        "members": len(members),
        "of": len(jittered) + 1,
        "model": model,
        "sky": float(skies_.mean()),
        "sky_spread": float(skies_.std(ddof=1)),
        "residual": image - skies_.mean() - model,
        "spread": models.std(axis=0, ddof=1),
        "fits": [
            {
                "label": lbl,
                "sky": float(m.settings.sky),
                "model": m.model,
                "residual": image - m.settings.sky - m.model,
                "score": m.score,
            }
            for lbl, m in zip(labels, members)
        ],
    }


def largest_crossing(profile) -> Optional[float]:
    """How far, at most, an ellipse reaches out of the next one (1: touching)."""
    ell = pair.profile_ellipses(profile)
    if len(ell) < 2:
        return None
    return max(pair.crossing(ell[k], ell[k + 1]) for k in range(len(ell) - 1))


def summary(result: SingleResult) -> str:
    s = result.settings
    lines = [
        f"{result.status}: R1 {result.r1:.1f} (asked {s.r1:g}), NR {result.nr}, sky "
        f"{s.sky:.6g}"
        + (f" (factor {s.sky_factor:.5f})" if result.sky_iterations else "")
        + (f"; residual score {result.score:.3f}" if result.score is not None else "")
    ]
    if result.reason:
        lines.append(result.reason)
    for row in result.search:
        sc = "-" if row["score"] is None else f"{row['score']:.3f}"
        lines.append(
            f"  {row['variant']}: {row['status']}, R1 {row['r1']:.1f}, score {sc}"
            + ("  <- kept" if row.get("chosen") else "")
        )
    for e in result.sky_iterations:
        med = e.get("median")
        lines.append(
            f"  sky {e['iteration']}: factor {e['factor']:.5f}"
            + (f", residual sky {med:+.4g}" if med is not None else "")
            + ("  <- kept" if e.get("chosen") else "")
            + (f"; {e['note']}" if e.get("note") else "")
        )
    lines += [f"  {x}" for x in result.steps]
    if result.harmonics:
        lines.append(
            "harmonic analysis (residual score; R1 reached; * its ellipses crossed or "
            "failed further out):"
        )
        for h in result.harmonics:
            sc = "-" if h["score"] is None else f"{h['score']:.3f}"
            lines.append(
                f"  round {h['round']}: COS3X={h['cos3x']:+d} COS4X={h['cos4x']}: score {sc}, "
                f"R1 {h['r1']:.1f}{' *' if h.get('short') else ''}"
                + ("  <- kept" if h.get("kept") else "")
            )
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
            f"mean of {e['members']} fits (the fit kept and {e['of'] - 1} with R0 and R1 varied, "
            f"each with its own sky): mean sky {e['sky']:.6g} ± {e['sky_spread']:.2g}; mean "
            "residual = image - mean sky - mean model"
        )
    lines.append(f"{result.calls} elliprof calls in {result.seconds:.1f} s")
    return "\n".join(lines)
