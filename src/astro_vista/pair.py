"""Fitting a pair of overlapping galaxies: ELLIPROF on each in turn, with the other's
current model taken out.

    initialisation:  A on I (B's core, or B's BLOCK, left out), then B on I - A (A's core
                     left out), each at its START radius (R1 by default)
    growth step g:   A on I - B, then B on I - A, with R1 grown from START to R1
                     (g = 1 .. STEPS); a galaxy whose growth fails stays at its last radius
    cycle k:         A on I - B, then B on I - A          (k = 1 .. CYCLES)
    result:          A, B, A + B and the residual I - S - A - B on every pixel

Every call gets the sky S once (elliprof subtracts it).  A model is *committed* only when
its fit passes every check; a rejected candidate is never used, and the last cycle in which
both fits passed is kept as the checkpoint, which is what the result shows.  Nothing here
touches Qt: the window runs :func:`run` in a worker thread on a :class:`Snapshot` made
before it starts, and reads the :class:`PairResult` back in its own thread.

Coordinates are ELLIPROF's (the first pixel's centre is at 0.5), as everywhere in the
program.  The cores (the region around one galaxy left out while the other is fitted) and
the evaluation pixels are fixed before the first call and do not change during the run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import masks as maskops
from .fitting import MODES, SIXTH_ORDER_WARNING, ArrayFit, fit_array

STATUSES = (
    "stable_valid_pair",
    "valid_pair_not_stable",
    "retained_after_failed_update",
    "incomplete_initialization",
    "cancelled",
)
STABILITY = 0.02  # relative change of A, B and A + B below which a pair is stable
CORECAP = 50.0
NR_TEACHING = 15  # below this an NR is flagged as a coarse (teaching) fit
NR_RATIO = 1.3  # NR=0 (automatic): isophotes about 30 % apart in Rmaj, from R0 to each R1
NR_RANGE = (6, 40)


def auto_nr(r0: float, r1: float) -> int:
    """The number of isophotes NR=0 asks for: about NR_RATIO apart from r0 to r1, within
    NR_RANGE.  A fixed NR packs a small R1 (the first fits of a growing schedule, a small
    companion) with isophotes a fraction of a pixel apart, and noise makes them cross."""
    n = int(round(math.log(max(r1, r0 * 1.01) / r0) / math.log(NR_RATIO))) + 1
    return int(min(max(n, NR_RANGE[0]), NR_RANGE[1]))


# the rogue-contour thresholds (starting values; see the plan, T7)
JUMP_PIXELS, JUMP_FRACTION = 2.0, 0.15
Q_EXCURSION = 0.15
PA_EXCURSION = 20.0
PA_MIN_ELLIP = 0.05
CROSSING = 1.02  # an inner ellipse may reach 2 % beyond the next one out
OUTER_SUPPORT = 0.5


class PairError(Exception):
    pass


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- settings
@dataclass
class Component:
    id: str  # "A" or "B": the identity, never changed
    x0: float
    y0: float
    r0: float
    r1: float
    nr: int = 25
    name: str = ""  # the label shown
    mode: str = ""  # a named harmonic mode (fitting.MODES), or ""
    cos3x: Optional[int] = None
    cos4x: Optional[int] = None
    fixctr: int = 0
    core: Optional[float] = None  # the semi-major axis of the core centred on this galaxy
    adaptive: bool = False
    block: Optional[float] = None  # BLOCK: left out around this galaxy for the other's 1st fit
    start: Optional[float] = None  # START: the R1 of its first fit (then grown to R1)

    step: Optional[float] = None  # STEP: R1 grows by this many pixels per cycle
    factor: Optional[float] = None  # FACTOR: R1 grows by this factor per cycle
    radii: Optional[List[float]] = None  # RADII: the R1 of each fit, listed
    try_cos3x: List[int] = field(default_factory=list)  # TRYCOS3X=: harmonics to search
    try_cos4x: List[int] = field(default_factory=list)  # TRYCOS4X=

    @property
    def first_r1(self) -> float:
        if self.radii:
            return self.radii[0]
        return self.start if self.start else self.r1

    def schedule(self, steps: int = 0) -> List[float]:
        """The R1 of each fit of this galaxy, from the first to R1: RADII as listed; else
        from START by STEP pixels or by FACTOR each cycle; else STEPS geometric steps (the
        shared setting); always ending at R1."""
        if self.radii:
            out = [r for r in self.radii if r < self.r1 - 1e-9]
        else:
            r = self.first_r1
            out = [r]
            if r < self.r1 - 1e-9:
                if self.step:
                    while out[-1] + self.step < self.r1 - 1e-9:
                        out.append(out[-1] + self.step)
                elif self.factor:
                    while out[-1] * self.factor < self.r1 - 1e-9:
                        out.append(out[-1] * self.factor)
                elif steps:
                    out = [r * (self.r1 / r) ** (g / steps) for g in range(steps)]
                else:
                    out = []
        return [float(x) for x in out if x < self.r1 - 1e-9] + [float(self.r1)]

    @property
    def label(self) -> str:
        return self.name or self.id

    def harmonics(self) -> Tuple[Optional[int], Optional[int]]:
        if self.mode:
            return MODES[self.mode]
        return self.cos3x, self.cos4x


@dataclass
class Settings:
    sky: float  # the sky every call subtracts: sky_factor x sky_base
    cycles: int = 4
    niter: int = 10
    rlaw: int = 2
    gain: float = 1.0
    tie: int = -1
    rmstar: bool = False
    corecap: float = CORECAP
    early_stop: bool = False
    timeout: float = 300.0
    bootstrap: bool = True  # a failed first fit may start from a smaller R1 (never a result)
    sky_base: Optional[float] = None  # SKY=, the level the factor multiplies
    sky_factor: float = 1.0  # SKYFACTOR=
    sky_iter: int = 0  # SKYITER=: refit with the factor updated from the residual sky
    steps: int = 0  # STEPS=: growth steps from each galaxy's START to its R1
    cores: bool = True  # CORES=0: nothing is left out around the companion
    converge: float = STABILITY  # CONVERGE=: the relative change below which it is stable
    consecutive: int = 0  # CONSECUTIVE=: stop after this many stable cycles at the targets
    on_fail: str = "reduce"  # ONFAIL=reduce|hold|retry|stop
    tail: str = "elliprof"  # TAIL=: the model beyond the last isophote (see tail_model)
    sky_plane: bool = False  # SKYPLANE=1: a tilt fitted to a first pass's residual, removed
    damp: float = 1.0  # DAMP=: the share of a new model in the one subtracted (1: no damping)
    try_damp: List[float] = field(default_factory=list)  # TRYDAMP=: variants to search
    try_fix: List[int] = field(default_factory=list)  # TRYFIX=: centre fitting variants
    try_factor: List[float] = field(default_factory=list)  # TRYFACTOR=: growth factors
    polish: int = 2  # POLISH=: undamped final cycles after damped ones
    parallel: int = 1  # PARALLEL=: variants fitted at the same time (threads)
    wings: bool = False  # WINGS=1: mask what a first pass's residual shows around bright sources
    runs: int = 1  # RUNS=: the fit kept, and RUNS - 1 repeats with R0 and R1 changed at random
    jitter: float = 0.15  # JITTER=: each repeat's R0 and R1 change by 2/3 of this to this
    seed: Optional[int] = None  # SEED=: the repeats' random numbers (default: the run's number)
    harmonic_search: bool = False  # HARMONICS=AUTO: every COS3X and COS4X tried for each


@dataclass
class Core:
    owner: str  # the galaxy it is centred on; it is left out while the other is fitted
    shape: str  # "ellipse", "circle" or "none"
    x: float
    y: float
    a: float
    b: float
    angle: float  # degrees from +x
    reason: str
    masked: np.ndarray = field(repr=False, default=None)

    def describe(self) -> dict:
        return {
            "owner": self.owner,
            "excluded_while_fitting": "B" if self.owner == "A" else "A",
            "shape": self.shape,
            "x": self.x,
            "y": self.y,
            "a": self.a,
            "b": self.b,
            "angle": self.angle,
            "reason": self.reason,
            "pixels": int(self.masked.sum()) if self.masked is not None else 0,
        }


@dataclass
class Match:
    """A component's match in the source finder's catalogue (or why there is none)."""

    ok: bool
    reason: str
    number: Optional[int] = None
    x: float = 0.0
    y: float = 0.0
    a: float = 0.0  # Kron semi-major axis, pixels (KRON_RADIUS x A_IMAGE)
    q: float = 1.0  # B / A
    theta: float = 0.0  # degrees


@dataclass
class Snapshot:
    """Everything a run uses, fixed before it starts."""

    image: np.ndarray  # float64 copy of the science buffer
    image_name: str
    header: object
    settings: Settings
    components: Dict[str, Component]
    fit_mask: np.ndarray  # M0: the selected fitting exclusions and the invalid pixels
    fit_mask_label: str
    invalid: np.ndarray
    cores: Dict[str, Core]  # keyed by owner
    eval: np.ndarray
    matches: Dict[str, Match]
    run_id: int = 0
    sky_id: str = ""  # the sky measurement S came from, or ""
    tilt: Optional[Dict[str, float]] = None  # SKYPLANE: the plane removed from the image
    sky_sigma: Optional[float] = None
    provenance: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    blocks: Dict[str, Core] = field(default_factory=dict)  # keyed by owner; BLOCK only
    # R1=AUTO / START=AUTO: which galaxies' radii run_search measures again on the image
    # minus the other's model (a first pass), and the noise those radii use
    auto: Dict[str, Tuple[bool, bool]] = field(default_factory=dict)  # cid: (R1, START)
    auto_sigma: Optional[float] = None

    def mask_for(self, cid: str, first: bool = False) -> np.ndarray:
        """The pixels left out while fitting component ``cid``: M0 and the other's core
        (and, for ``cid``'s first fit, the other's block)."""
        other = "B" if cid == "A" else "A"
        mask = self.fit_mask | self.cores[other].masked
        if first and other in self.blocks:
            mask = mask | self.blocks[other].masked
        return mask


# ---------------------------------------------------------------- results
@dataclass
class Commit:
    """A component's accepted state: model, profile and ellipses always together."""

    model: np.ndarray  # float64
    profile: object
    ellipses: List[tuple]
    tag: str  # "init" or the cycle number
    r1_call: float  # the R1 of the call that made it (its effective R1)
    support: List[dict]
    coverage_uncertain: bool
    bootstrap: bool = False  # a partial first fit: it only starts the cycles
    requested_r1: float = 0.0

    @property
    def next_r1(self) -> float:
        """The R1 the next fit of this galaxy starts from."""
        return self.requested_r1 if self.bootstrap else self.r1_call

    @property
    def rmaj_last(self) -> float:
        return float(self.profile["Rmaj"].iloc[-1])


@dataclass
class Call:
    cycle: str
    component: str
    r1: float
    arguments: List[str]
    returncode: int
    seconds: float
    passed: bool
    reason: str
    precision: Optional[str] = None
    input: str = ""  # "D", "D - M_A" or "D - M_B"
    companion: str = ""  # the companion model subtracted, with the cycle that made it
    centre_drift: Optional[float] = None  # largest fitted-centre distance from X0, Y0
    mask_pixels: int = 0


@dataclass
class PairResult:
    status: str
    stability_reason: str = ""
    checkpoint_cycle: Optional[str] = None  # "0" (after initialisation), "1", ...
    commits: Dict[str, Commit] = field(default_factory=dict)  # the result's A and B
    partial: Dict[str, Commit] = field(default_factory=dict)  # newer, not joint
    total: Optional[np.ndarray] = None
    residual: Optional[np.ndarray] = None
    notes_extra: List[str] = field(default_factory=list)  # said by run_search (SKYPLANE)
    metrics: List[dict] = field(default_factory=list)
    calls: List[Call] = field(default_factory=list)
    attempted_r1: Dict[str, float] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)  # the last failure of each
    diagnostics: Dict[str, ArrayFit] = field(default_factory=dict)  # rejected fits
    bootstraps: Dict[str, float] = field(default_factory=dict)  # partial first fits' R1
    sky_check: dict = field(default_factory=dict)
    growth: List[dict] = field(default_factory=list)  # one entry per growth step
    schedules: Dict[str, List[float]] = field(default_factory=dict)  # each galaxy's R1s
    # every call, with references to its model and the companion model it saw (not copies)
    history: List[dict] = field(default_factory=list)
    quality: dict = field(default_factory=dict)  # see quality()
    search: List[dict] = field(default_factory=list)  # TRY...: every variant, and its score
    search_choice: str = ""  # why run_search kept the fit it kept
    snapshot: Optional["Snapshot"] = None  # the snapshot of the fit kept, if not the job's
    notes_growth: List[str] = field(default_factory=list)  # galaxies that stopped growing
    # SKYITER: one entry per pair fit (factor, sky, residual median, status); the result is
    # the fit marked "chosen"
    sky_iterations: List[dict] = field(default_factory=list)
    # PARALLEL / the search: the mean of the fits close to the best (see ensemble())
    ensemble: dict = field(default_factory=dict)
    seconds: float = 0.0
    wall: float = 0.0  # run_search: the whole search, from start to end (parallel or not)
    r0_raised: Dict[str, float] = field(default_factory=dict)  # INNER_R0: the R0 used instead
    repeats: List[dict] = field(default_factory=list)  # RUNS: the fit kept and every repeat
    run_id: int = 0

    @property
    def complete(self) -> bool:
        return "A" in self.commits and "B" in self.commits


# ---------------------------------------------------------------- the command's words
def parse(text: str):
    """The shared keywords and those of A and B of a ``pair`` command (everything after
    ``pair``): ``N <shared> A: <A's> B: <B's>``.  Returns (words, shared, {A: .., B: ..})."""
    text = re.sub(r"\\\s*\n", " ", text)
    try:
        tokens = shlex.split(text, comments=True)
    except ValueError as e:
        raise PairError(f"cannot read the line: {e}")
    words: List[str] = []
    scopes: Dict[str, Dict[str, str]] = {"": {}, "A": {}, "B": {}}
    scope = ""
    seen = []
    for token in tokens:
        label = token.upper()
        if label in ("A:", "B:"):
            name = label[0]
            if name in seen:
                raise PairError(f"{label} appears twice")
            if name == "A" and "B" in seen:
                raise PairError("A: must come before B:")
            seen.append(name)
            scope = name
            continue
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", token)
        if m:
            key, value = m.group(1).upper(), m.group(2)
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token) and token.upper() in FLAGS:
            key, value = token.upper(), "1"
        else:
            if scope:
                raise PairError(f"{token!r} after {scope}: is not KEY=value")
            words.append(token)
            continue
        if key in scopes[scope]:
            raise PairError(f"{key} is given twice" + (f" for {scope}" if scope else ""))
        allowed = SHARED if not scope else OWN
        if key not in allowed:
            if not scope and key in OWN:
                raise PairError(f"{key} belongs to one galaxy: put it after A: or B:")
            if scope and key in SHARED:
                raise PairError(f"{key} is shared: put it before A:")
            raise PairError(f"unknown keyword {key}" + (f" for {scope}" if scope else ""))
        scopes[scope][key] = value
    for name in ("A", "B"):
        if name not in seen:
            raise PairError(f"{name}: is missing (pair N <shared> A: X0= Y0= R0= R1= B: ...)")
    return words, scopes[""], {"A": scopes["A"], "B": scopes["B"]}


SHARED = {
    "SKY",
    "MASK",
    "MASKFILE",
    "MASKBUF",
    "CYCLES",
    "NITER",
    "RLAW",
    "GAIN",
    "TIE",
    "RMSTAR",
    "CORECAP",
    "STOP",
    "TIMEOUT",
    "TOMODEL",
    "TORESID",
    "TOMODELA",
    "TOMODELB",
    "VIEWS",
    "BOOTSTRAP",
    "SKYFACTOR",
    "SKYITER",
    "STEPS",
    "CORES",
    "CONVERGE",
    "CONSECUTIVE",
    "ONFAIL",
    "DAMP",
    "POLISH",
    "TRYDAMP",
    "TRYFIX",
    "TRYFACTOR",
    "TAIL",
    "SKYPLANE",
    "PARALLEL",
    "WINGS",
    "RUNS",
    "JITTER",
    "SEED",
    "HARMONICS",
}
OWN = {
    "NAME",
    "X0",
    "Y0",
    "R0",
    "R1",
    "NR",
    "MODE",
    "COS3X",
    "COS4X",
    "FIXCTR",
    "CORE",
    "ADAPTIVE",
    "BLOCK",
    "START",
    "STEP",
    "FACTOR",
    "RADII",
    "TRYCOS3X",
    "TRYCOS4X",
}
FLAGS = {"RMSTAR", "STOP", "ADAPTIVE", "VIEWS"}


def _num(keys, key, what, default=None, kind=float):
    if key not in keys:
        if default is None:
            raise PairError(f"{key}= is required{what}")
        return default
    try:
        value = kind(float(keys[key])) if kind is int else float(keys[key])
    except ValueError:
        raise PairError(f"{key} must be a number{what}, not {keys[key]!r}")
    if kind is int and float(keys[key]) != int(float(keys[key])):
        raise PairError(f"{key} must be a whole number{what}")
    if not math.isfinite(value):
        raise PairError(f"{key} must be finite{what}")
    return value


def _numbers(keys, key) -> List[float]:
    if key not in keys:
        return []
    try:
        return [float(x) for x in re.split(r"[,\s]+", keys[key].strip()) if x]
    except ValueError:
        raise PairError(f"{key} must be numbers separated by commas, not {keys[key]!r}")


def _radii(text: str, where: str) -> List[float]:
    try:
        values = [float(x) for x in re.split(r"[,\s]+", text.strip()) if x]
    except ValueError:
        raise PairError(f"RADII must be numbers separated by commas{where}, not {text!r}")
    if not values or any(b <= a for a, b in zip(values, values[1:])):
        raise PairError(f"RADII must increase{where}")
    return values


# The automatic radii (R1=AUTO, START=AUTO), chosen on simulated pairs (see the changelog)
AUTO_R1_SIGMA = 1.0  # R1: where the light away from the companion falls below this x sigma
AUTO_R1_EDGE = 0.9  # ... at most this fraction of the distance to the frame's edge
START_SHARE = 1 / 3  # the first R1: this share of the final R1, ...
START_SEPARATION = 0.35  # ... at most this share of the distance between the centres, ...
START_MIN = 9.0  # ... and at least this many pixels
AUTO_FACTOR = 1.5  # the growth factor with START=AUTO


def auto_r1(
    light: np.ndarray,
    masked: np.ndarray,
    centre: Tuple[float, float],
    other: Tuple[float, float],
    sigma: float,
    k: float = AUTO_R1_SIGMA,
    width: float = 3.0,
) -> float:
    """R1=AUTO: the inner radius of the first annulus (``width`` px wide, from 5 px) whose
    median ``light`` (the image minus the sky), on the half facing away from the companion
    and on pixels not ``masked``, is below ``k`` x ``sigma``; at most AUTO_R1_EDGE of the
    distance to the nearest edge, and at least 10 px."""
    ny, nx = light.shape
    x, y = centre
    edge = min(x, y, nx - x, ny - y)
    rmax = max(12.0, 0.95 * edge)
    r0 = int(math.ceil(rmax)) + 2
    i0, i1 = max(int(x) - r0, 0), min(int(x) + r0, nx)
    j0, j1 = max(int(y) - r0, 0), min(int(y) + r0, ny)
    yy, xx = np.mgrid[j0:j1, i0:i1]
    dx, dy = xx + 0.5 - x, yy + 0.5 - y
    r = np.hypot(dx, dy)
    away = (dx * (x - other[0]) + dy * (y - other[1])) >= 0
    sub = light[j0:j1, i0:i1]
    ok = away & ~masked[j0:j1, i0:i1] & np.isfinite(sub)
    cap = AUTO_R1_EDGE * edge
    for lo in np.arange(5.0, rmax, width):
        sel = ok & (r >= lo) & (r < lo + width)
        if sel.sum() >= 8 and np.median(sub[sel]) < k * sigma:
            return round(float(max(10.0, min(lo, cap))), 1)
    return round(float(max(10.0, cap)), 1)


def reaches_edge(r1: float, centre: Tuple[float, float], shape) -> bool:
    """Whether an R1=AUTO stopped at its cap: the galaxy's light is still above the noise
    near the frame's edge."""
    ny, nx = shape
    x, y = centre
    edge = min(x, y, nx - x, ny - y)
    return abs(r1 - max(10.0, AUTO_R1_EDGE * edge)) < 0.06


def auto_start(r1: float, separation: float) -> Optional[float]:
    """START=AUTO: START_SHARE of R1, at most START_SEPARATION of the separation (so that
    the first fits stay off the companion's centre), at least START_MIN; None (no growth)
    when that is not below R1."""
    start = max(min(START_SHARE * r1, START_SEPARATION * separation), START_MIN)
    return round(start, 1) if start < r1 - 1 else None


def resolve_auto(
    own: Dict[str, Dict[str, str]],
    light: np.ndarray,
    masked: np.ndarray,
    sigma: float,
) -> List[str]:
    """Replace R1=AUTO and START=AUTO in the A: and B: keywords by numbers (in place), and
    say what was chosen.  ``light`` is the image minus the sky; ``masked``, the fitting
    mask."""
    notes = []
    try:
        centres = {c: (float(own[c]["X0"]), float(own[c]["Y0"])) for c in ("A", "B")}
    except (KeyError, ValueError):
        return notes  # components_from reports it
    separation = math.hypot(centres["A"][0] - centres["B"][0], centres["A"][1] - centres["B"][1])
    for cid in ("A", "B"):
        k = own[cid]
        other = centres["B" if cid == "A" else "A"]
        if str(k.get("R1", "")).upper() == "AUTO":
            k["R1"] = format(auto_r1(light, masked, centres[cid], other, sigma), "g")
            notes.append(
                f"{cid}: R1=AUTO -> {k['R1']} (light below {AUTO_R1_SIGMA:g} sigma = "
                f"{AUTO_R1_SIGMA * sigma:.4g}, away from the companion)"
            )
        if str(k.get("START", "")).upper() == "AUTO":
            start = auto_start(float(k["R1"]), separation)
            del k["START"]
            if start is None:
                notes.append(f"{cid}: START=AUTO -> no growth (R1 is small)")
            else:
                k["START"] = format(start, "g")
                if not any(w in k for w in ("STEP", "FACTOR", "RADII")):
                    k["FACTOR"] = format(AUTO_FACTOR, "g")
                notes.append(f"{cid}: START=AUTO -> {k['START']}")
    return notes


# ELLIPROF's harmonic keywords, which choose what goes into the model (every isophote's fit
# measures the terms): 0 none; 1 the median coefficients over all isophotes; 2 each
# isophote its own (the default); a negative COS3X fits the 6th order instead of the 3rd
# (-2 each, -1 median, -3 none)
COS3X_VALUES = (-3, -2, -1, 0, 1, 2)
COS4X_VALUES = (0, 1, 2)
DEFAULT_HARMONICS = (2, 2)
HARMONIC_COMBINATIONS = 18  # TRYCOS3X x TRYCOS4X of one galaxy: at most this many to fit


def _listed(values) -> str:
    return ", ".join(str(v) for v in values)


def components_from(own: Dict[str, Dict[str, str]]) -> Dict[str, Component]:
    out = {}
    for cid in ("A", "B"):
        k = own[cid]
        where = f" for {cid}"
        if "MODE" in k and ("COS3X" in k or "COS4X" in k):
            raise PairError(f"give MODE= or COS3X=/COS4X= for {cid}, not both")
        mode = k.get("MODE", "").strip().lower()
        if mode and mode not in MODES:
            raise PairError(f"unknown MODE {mode!r}{where} ({', '.join(MODES)})")
        c = Component(
            id=cid,
            x0=_num(k, "X0", where),
            y0=_num(k, "Y0", where),
            r0=_num(k, "R0", where),
            r1=_num(k, "R1", where),
            nr=0 if str(k.get("NR", "")).upper() == "AUTO" else _num(k, "NR", where, 25, int),
            name=k.get("NAME", ""),
            mode=mode,
            cos3x=_num(k, "COS3X", where, None, int) if "COS3X" in k else None,
            cos4x=_num(k, "COS4X", where, None, int) if "COS4X" in k else None,
            fixctr=_num(k, "FIXCTR", where, 0, int),
            core=_num(k, "CORE", where) if "CORE" in k else None,
            adaptive="ADAPTIVE" in k,
            block=_num(k, "BLOCK", where) if "BLOCK" in k else None,
            start=_num(k, "START", where) if "START" in k else None,
            step=_num(k, "STEP", where) if "STEP" in k else None,
            factor=_num(k, "FACTOR", where) if "FACTOR" in k else None,
            radii=_radii(k["RADII"], where) if "RADII" in k else None,
            try_cos3x=[int(x) for x in _numbers(k, "TRYCOS3X")],
            try_cos4x=[int(x) for x in _numbers(k, "TRYCOS4X")],
        )
        if not 0 < c.r0 < c.r1:
            raise PairError(f"0 < R0 < R1 is needed{where}")
        if not (c.nr == 0 or 2 <= c.nr <= 100):
            raise PairError(f"NR must be 2 to 100, or 0 (automatic){where}")
        for key, value, allowed in (
            ("COS3X", c.cos3x, COS3X_VALUES),
            ("COS4X", c.cos4x, COS4X_VALUES),
        ):
            if value is not None and value not in allowed:
                raise PairError(f"{key} must be one of {_listed(allowed)}{where}")
        for key, values, allowed in (
            ("TRYCOS3X", c.try_cos3x, COS3X_VALUES),
            ("TRYCOS4X", c.try_cos4x, COS4X_VALUES),
        ):
            if any(v not in allowed for v in values):
                raise PairError(f"{key} values must be among {_listed(allowed)}{where}")
        if mode and (c.try_cos3x or c.try_cos4x):
            raise PairError(f"give MODE= or TRYCOS3X=/TRYCOS4X= for {cid}, not both")
        if c.fixctr not in (0, 1, 2):
            raise PairError(f"FIXCTR must be 0, 1 or 2{where}")
        if c.core is not None and c.core < 0:
            raise PairError(f"CORE must be 0 or more{where}")
        if c.block is not None and c.block < 0:
            raise PairError(f"BLOCK must be 0 or more{where}")
        if c.start is not None and not c.r0 < c.start <= c.r1:
            raise PairError(f"R0 < START <= R1 is needed{where}")
        if sum(x is not None for x in (c.step, c.factor, c.radii)) > 1:
            raise PairError(f"give one of STEP=, FACTOR= and RADII={where}")
        if c.step is not None and c.step <= 0:
            raise PairError(f"STEP must be positive{where}")
        if c.factor is not None and c.factor <= 1:
            raise PairError(f"FACTOR must be above 1{where}")
        if c.radii and not (c.r0 < c.radii[0] and c.radii[-1] <= c.r1 + 1e-9):
            raise PairError(f"RADII must lie between R0 and R1{where}")
        out[cid] = c
    return out


MAX_PARALLEL = 16
MAX_RUNS = 50


def settings_from(shared: Dict[str, str]) -> Settings:
    base = _num(shared, "SKY", " (the frozen sky: measure it with sky N SEGMENT)")
    factor = _num(shared, "SKYFACTOR", "", 1.0)
    s = Settings(
        sky=base * factor,
        sky_base=base,
        sky_factor=factor,
        sky_iter=_num(shared, "SKYITER", "", 0, int),
        steps=_num(shared, "STEPS", "", 0, int),
        cores=_num(shared, "CORES", "", 1, int) != 0,
        converge=_num(shared, "CONVERGE", "", STABILITY),
        consecutive=_num(shared, "CONSECUTIVE", "", 1 if "STOP" in shared else 0, int),
        on_fail=shared.get("ONFAIL", "reduce").strip().lower(),
        damp=_num(shared, "DAMP", "", 1.0),
        try_damp=_numbers(shared, "TRYDAMP"),
        try_fix=[int(x) for x in _numbers(shared, "TRYFIX")],
        try_factor=_numbers(shared, "TRYFACTOR"),
        polish=_num(shared, "POLISH", "", 2, int),
        cycles=_num(shared, "CYCLES", "", 4, int),
        niter=_num(shared, "NITER", "", 10, int),
        rlaw=_num(shared, "RLAW", "", 2, int),
        gain=_num(shared, "GAIN", "", 1.0),
        tie=_num(shared, "TIE", "", -1, int),
        rmstar="RMSTAR" in shared,
        corecap=_num(shared, "CORECAP", "", CORECAP),
        early_stop="STOP" in shared,
        timeout=_num(shared, "TIMEOUT", "", 300.0),
        bootstrap=_num(shared, "BOOTSTRAP", "", 1, int) != 0,
        tail=shared.get("TAIL", "elliprof").strip().lower(),
        sky_plane=_num(shared, "SKYPLANE", "", 0, int) != 0,
        parallel=_num(shared, "PARALLEL", "", 1, int),
        wings=_num(shared, "WINGS", "", 0, int) != 0,
        runs=_num(shared, "RUNS", "", 1, int),
        jitter=_num(shared, "JITTER", "", 0.15),
        seed=_num(shared, "SEED", "", None, int) if "SEED" in shared else None,
        harmonic_search=shared.get("HARMONICS", "").strip().upper() == "AUTO",
    )
    if s.tail not in TAILS:
        raise PairError("TAIL must be one of " + ", ".join(TAILS))
    if "HARMONICS" in shared and not s.harmonic_search:
        raise PairError("HARMONICS must be AUTO (every COS3X and COS4X tried for each galaxy)")
    if not 1 <= s.runs <= MAX_RUNS:
        raise PairError(f"RUNS must be 1 to {MAX_RUNS}")
    if not 0 < s.jitter <= 0.5:
        raise PairError("JITTER must be above 0 and at most 0.5 (a share of R0 and R1)")
    if not 1 <= s.parallel <= MAX_PARALLEL:
        raise PairError(f"PARALLEL must be 1 to {MAX_PARALLEL}")
    if s.cycles < 1 or s.cycles > 50:
        raise PairError("CYCLES must be 1 to 50")
    if s.niter < 1 or s.niter > 1000:
        raise PairError("NITER must be 1 to 1000")
    if s.timeout <= 0:
        raise PairError("TIMEOUT must be positive")
    if not 0.5 <= factor <= 1.5:
        raise PairError("SKYFACTOR must be 0.5 to 1.5")
    if not 0 <= s.sky_iter <= 50:
        raise PairError("SKYITER must be 0 to 50")
    if not 0 <= s.steps <= 20:
        raise PairError("STEPS must be 0 to 20")
    if not 0 < s.converge < 1:
        raise PairError("CONVERGE must be between 0 and 1 (a relative change)")
    if not 0 <= s.consecutive <= 20:
        raise PairError("CONSECUTIVE must be 0 to 20")
    if not 0 < s.damp <= 1:
        raise PairError("DAMP must be above 0 and at most 1 (1: no damping)")
    if not 1 <= s.polish <= 10:
        raise PairError("POLISH must be 1 to 10")
    if any(not 0 < x <= 1 for x in s.try_damp):
        raise PairError("TRYDAMP values must be above 0 and at most 1")
    if any(x not in (0, 1, 2) for x in s.try_fix):
        raise PairError("TRYFIX values must be 0, 1 or 2")
    if any(x <= 1 for x in s.try_factor):
        raise PairError("TRYFACTOR values must be above 1")
    n = max(len(s.try_damp), 1) * max(len(s.try_fix), 1) * max(len(s.try_factor), 1)
    if n > 16:
        raise PairError(f"{n} variants to search: at most 16")
    if s.on_fail not in ("reduce", "hold", "retry", "stop"):
        raise PairError("ONFAIL must be reduce, hold, retry or stop")
    return s


# ---------------------------------------------------------------- before the run
def match_source(run, comp: Component) -> Match:
    """The source of a finder run at the component's centre, if it is a clean match."""
    if run is None or not len(run):
        return Match(False, "no source-finder run on this image")
    c = run.columns
    x = c["X_IMAGE"] - 0.5
    y = c["Y_IMAGE"] - 0.5
    d = np.hypot(x - comp.x0, y - comp.y0)
    order = np.argsort(d)
    k = int(order[0])
    if d[k] > 3:
        return Match(False, f"no source within 3 px (nearest {d[k]:.1f} px)")
    if len(order) > 1:
        second = d[int(order[1])]
        if second <= 3 and second < 2 * d[k]:
            return Match(False, "two sources within 3 px: ambiguous")
    flags = int(c["FLAGS"][k]) if "FLAGS" in c else 0
    kron, a, b = float(c["KRON_RADIUS"][k]), float(c["A_IMAGE"][k]), float(c["B_IMAGE"][k])
    number = int(c["NUMBER"][k])
    if flags & (8 | 16):
        return Match(False, f"source {number} is truncated or corrupted (FLAGS {flags})", number)
    if not (math.isfinite(kron) and kron > 0 and math.isfinite(a) and a > 0 and b > 0):
        return Match(False, f"source {number} has no valid Kron radius or axes", number)
    if float(c["ISOAREA_IMAGE"][k]) < 50:
        return Match(False, f"source {number} is small (ISOAREA < 50)", number)
    return Match(
        True,
        f"source {number}",
        number,
        float(x[k]),
        float(y[k]),
        kron * a,
        b / a,
        float(c["THETA_IMAGE"][k]),
    )


def _elliptical_distance(dx, dy, q, theta):
    t = math.radians(theta)
    u = dx * math.cos(t) + dy * math.sin(t)
    v = -dx * math.sin(t) + dy * math.cos(t)
    return math.sqrt(u * u + (v / q) ** 2)


def make_core(
    owner: Component, target: Component, match: Match, shape, fit_mask, invalid, corecap: float
) -> Core:
    """The region centred on ``owner`` left out while ``target`` is fitted."""
    dx, dy = target.x0 - owner.x0, target.y0 - owner.y0
    if match.ok:
        d_ell = _elliptical_distance(dx, dy, match.q, match.theta)
        a = min(match.a, 0.65 * d_ell, corecap)
        reason = (
            f"ellipse from {match.reason}: min(Kron {match.a:.1f}, 0.65 x {d_ell:.1f}, "
            f"cap {corecap:g})"
        )
        if owner.core is not None:
            a = min(owner.core, 0.65 * d_ell)
            reason = f"ellipse, CORE={owner.core:g} (at most 0.65 x {d_ell:.1f})"
        q, angle, kind = match.q, match.theta, "ellipse"
    else:
        d = math.hypot(dx, dy)
        a = min(0.65 * d, corecap)
        reason = (
            f"circle (no clean source match: {match.reason}): min(0.65 x {d:.1f}, cap {corecap:g})"
        )
        if owner.core is not None:
            a = min(owner.core, 0.65 * d)
            reason = f"circle, CORE={owner.core:g} (at most 0.65 x {d:.1f})"
        q, angle, kind = 1.0, 0.0, "circle"
    # the core must not newly cover the target's centre pixel when that pixel is usable
    ny, nx = shape
    i, j = int(math.floor(target.x0)), int(math.floor(target.y0))
    if 0 <= i < nx and 0 <= j < ny and not fit_mask[j, i] and not invalid[j, i]:
        d_pix = _elliptical_distance(i + 0.5 - owner.x0, j + 0.5 - owner.y0, q, angle)
        if a >= d_pix:
            a = max(d_pix * (1 - 1e-6) - 1e-6, 0.0)
            reason += f"; reduced to {a:.2f} to leave {target.id}'s centre pixel usable"
    masked = np.zeros(shape, bool)
    if a <= 0:
        kind = "none"
        reason += "; no core"
    else:
        maskops.ellipse(masked, owner.x0, owner.y0, a, a * q, angle, True)
    return Core(owner.id, kind, owner.x0, owner.y0, a, a * q, angle, reason, masked)


def make_block(
    owner: Component, target: Component, match: Match, shape, fit_mask, invalid
) -> Optional[Core]:
    """BLOCK: the region centred on ``owner`` left out of ``target``'s first fit (an
    ellipse shaped as the owner's source match, else a circle); like a core, it never
    newly covers the target's centre pixel."""
    if not owner.block:
        return None
    q, angle = (match.q, match.theta) if match.ok else (1.0, 0.0)
    a = float(owner.block)
    reason = f"BLOCK={owner.block:g}, " + (f"shaped as {match.reason}" if match.ok else "a circle")
    ny, nx = shape
    i, j = int(math.floor(target.x0)), int(math.floor(target.y0))
    if 0 <= i < nx and 0 <= j < ny and not fit_mask[j, i] and not invalid[j, i]:
        d_pix = _elliptical_distance(i + 0.5 - owner.x0, j + 0.5 - owner.y0, q, angle)
        if a >= d_pix:
            a = max(d_pix * (1 - 1e-6) - 1e-6, 0.0)
            reason += f"; reduced to {a:.2f} to leave {target.id}'s centre pixel usable"
    masked = np.zeros(shape, bool)
    if a > 0:
        maskops.ellipse(masked, owner.x0, owner.y0, a, a * q, angle, True)
    return Core(
        owner.id,
        "ellipse" if q < 1 else "circle",
        owner.x0,
        owner.y0,
        a,
        a * q,
        angle,
        reason,
        masked,
    )


def prepare(
    image: np.ndarray,
    image_name: str,
    header,
    components: Dict[str, Component],
    settings: Settings,
    fit_mask: Optional[np.ndarray] = None,
    fit_mask_label: str = "",
    run=None,
    run_id: int = 0,
    sky_id: str = "",
    sky_sigma: Optional[float] = None,
    provenance: Optional[dict] = None,
) -> Snapshot:
    """Fix everything a run uses: a float64 copy of the image, M0, the cores, EVAL."""
    data = np.array(image, dtype=np.float64, copy=True)
    shape = data.shape
    invalid = ~np.isfinite(data)
    m0 = invalid.copy()
    if fit_mask is not None:
        if fit_mask.shape != shape:
            raise PairError("the fitting mask is not the size of the image")
        m0 |= fit_mask
    a, b = components["A"], components["B"]
    if math.hypot(a.x0 - b.x0, a.y0 - b.y0) < 3:
        raise PairError("the two centres are less than 3 px apart: the same object")
    ny, nx = shape
    for c in (a, b):
        if not (0 <= c.x0 < nx and 0 <= c.y0 < ny):
            raise PairError(f"{c.id}'s centre {c.x0:g}, {c.y0:g} is outside the image")
    notes = []
    matches = {cid: match_source(run, c) for cid, c in components.items()}
    if matches["A"].ok and matches["B"].ok and matches["A"].number == matches["B"].number:
        note = f"one source covers both centres (blend): source {matches['A'].number}"
        matches = {cid: Match(False, note) for cid in matches}
        notes.append(note)
    if settings.cores:
        cores = {
            "A": make_core(a, b, matches["A"], shape, m0, invalid, settings.corecap),
            "B": make_core(b, a, matches["B"], shape, m0, invalid, settings.corecap),
        }
    else:  # the companion is not masked: its current model is subtracted instead
        cores = {
            c.id: Core(
                c.id, "none", c.x0, c.y0, 0.0, 0.0, 0.0, "CORES=0: no core", np.zeros(shape, bool)
            )
            for c in (a, b)
        }
    blocks = {}
    for owner, target in ((a, b), (b, a)):
        block = make_block(owner, target, matches[owner.id], shape, m0, invalid)
        if block is not None:
            blocks[owner.id] = block
    ev = np.zeros(shape, bool)
    for c in (a, b):
        maskops.circle(ev, c.x0, c.y0, c.r1, True)
    ev &= ~m0
    for c in (a, b):
        if 0 < c.nr < NR_TEACHING:
            notes.append(f"{c.id}: NR={c.nr} is a coarse (teaching) fit")
    return Snapshot(
        image=data,
        image_name=image_name,
        header=header.copy() if header is not None else None,
        settings=settings,
        components=components,
        fit_mask=m0,
        fit_mask_label=fit_mask_label,
        invalid=invalid,
        cores=cores,
        eval=ev,
        matches=matches,
        run_id=run_id,
        sky_id=sky_id,
        sky_sigma=sky_sigma,
        provenance=dict(provenance or {}),
        notes=notes,
        blocks=blocks,
    )


# ---------------------------------------------------------------- checks
def profile_ellipses(profile) -> List[tuple]:
    """(x, y, a, b, angle from +x in degrees) of each isophote, ELLIPROF's coordinates."""
    out = []
    for r in profile.itertuples(index=False):
        out.append(
            (
                float(r.x0),
                float(r.y0),
                float(r.Rmaj),
                float(r.Rmaj * (1 - r.ellip)),
                float(r.alpha) - 90.0,
            )
        )
    return out


def crossing(inner, outer, points: int = 180) -> float:
    """How far the inner ellipse reaches out of the outer one: the largest normalised
    radius rho of its points in the outer ellipse's frame (1 on the outer ellipse)."""
    xi, yi, ai, bi, ti = inner
    xo, yo, ao, bo, to = outer
    phi = np.linspace(0, 2 * np.pi, points, endpoint=False)
    t = math.radians(ti)
    px = xi + ai * np.cos(phi) * math.cos(t) - bi * np.sin(phi) * math.sin(t)
    py = yi + ai * np.cos(phi) * math.sin(t) + bi * np.sin(phi) * math.cos(t)
    t = math.radians(to)
    dx, dy = px - xo, py - yo
    u = dx * math.cos(t) + dy * math.sin(t)
    v = -dx * math.sin(t) + dy * math.cos(t)
    return float(np.max(np.sqrt((u / ao) ** 2 + (v / bo) ** 2)))


def _pa_difference(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def nr_for(comp: Component, r1: float) -> int:
    """The NR of a call of ``comp`` at ``r1``: its own, or automatic (NR=0)."""
    return comp.nr if comp.nr else auto_nr(comp.r0, r1)


def check(fit: ArrayFit, comp: Component, r1_call: float, image_finite: np.ndarray):
    """Whether a candidate fit passes: (ok, reason, kind, radius).  ``kind`` is "outer"
    for a failure at one isophote (a value not finite or out of range, a crossing, a rogue
    contour; an adaptive R1 may retry it if it is far enough out) and "other" otherwise;
    ``radius`` is the Rmaj where it failed."""
    if fit.returncode != 0 or fit.profile is None:
        return False, f"elliprof failed: {fit.message or fit.returncode}", "other", None
    p = fit.profile
    nr = nr_for(comp, r1_call)
    if len(p) != nr:
        return False, f"{len(p)} isophotes instead of NR={nr}", "other", None
    cols = {
        name: p[name].to_numpy(dtype=float) for name in ("Rmaj", "x0", "y0", "I0", "ellip", "alpha")
    }
    r = cols["Rmaj"]
    grid = r if np.all(np.isfinite(r)) else np.linspace(comp.r0, r1_call, len(p))
    # rows with a non-finite value or out of range: a failure there, far out, is an outer one
    bad = np.zeros(len(p), bool)
    for name, v in cols.items():
        bad |= ~np.isfinite(v)
    with np.errstate(invalid="ignore"):
        e = cols["ellip"]
        out_of_range = ~((e >= 0) & (e < 0.95)) | ~(cols["I0"] > 0)
    if bad.any() or out_of_range.any():
        row = int(np.flatnonzero(bad | out_of_range)[0])
        what = (
            "a value that is not finite"
            if bad[row]
            else (
                "an ellipticity outside 0 .. 0.95"
                if not (0 <= e[row] < 0.95)
                else "an intensity I0 that is not positive"
            )
        )
        return False, f"{what} at Rmaj {grid[row]:.1f}", "outer", float(grid[row])
    if not np.all(np.diff(r) > 0):
        return False, "Rmaj does not increase", "other", None
    if fit.model is None or fit.model.shape != image_finite.shape:
        return False, "no model of the image's size", "other", None
    if not np.all(np.isfinite(fit.model[image_finite])):
        return False, "the model is not finite where the image is", "other", None
    cos3x, _cos4x = comp.harmonics()
    if cos3x is not None and cos3x < 0 and SIXTH_ORDER_WARNING in fit.output:
        return False, "6th-order model wrong beyond a PA wrap (elliprof's warning)", "other", None
    ell = profile_ellipses(p)
    for k in range(len(ell) - 1):
        rho = crossing(ell[k], ell[k + 1])
        if rho > CROSSING:
            return (
                False,
                f"isophotes cross at Rmaj {r[k + 1]:.1f} (rho {rho:.3f})",
                "outer",
                float(r[k + 1]),
            )
    x, y, q = cols["x0"], cols["y0"], 1 - e
    for k in range(1, len(r)):
        jump = math.hypot(x[k] - x[k - 1], y[k] - y[k - 1])
        if jump > max(JUMP_PIXELS, JUMP_FRACTION * r[k]):
            return False, f"the centre jumps {jump:.1f} px at Rmaj {r[k]:.1f}", "outer", float(r[k])
    # an isolated shape: one isophote whose shape (ellipticity and angle together, as the
    # point e exp(2i PA)) differs from both neighbours by more than Q_EXCURSION while the
    # neighbours agree; an angle change counts in proportion to the ellipticity, so a
    # nearly round isophote is not failed for its ill-defined angle
    z = e * np.exp(2j * np.radians(np.asarray(cols["alpha"], float)))
    for k in range(1, len(r) - 1):
        d1, d2 = abs(z[k] - z[k - 1]), abs(z[k] - z[k + 1])
        if d1 > Q_EXCURSION and d2 > Q_EXCURSION and abs(z[k + 1] - z[k - 1]) < Q_EXCURSION:
            what = "axis ratio" if abs(e[k] - e[k - 1]) > 0.7 * d1 else "angle (PA)"
            return (
                False,
                f"an isolated isophote shape ({what}) at Rmaj {r[k]:.1f}",
                "outer",
                float(r[k]),
            )
    if abs(r[-1] - r1_call) > 0.01 * r1_call + 0.5:
        return False, f"the last isophote is at {r[-1]:.1f}, not R1={r1_call:.1f}", "other", None
    return True, "", "", None


TAILS = ("measured", "sersic", "elliprof", "exp", "cut")

# A failure of the innermost isophotes (within INNER_R0 px: a few PSF widths, where an
# isophote's shape is set by the pixels and the PSF more than by the galaxy) is retried with
# R0 at INNER_FACTOR times the radius that failed, at most INNER_TRIES times and below
# INNER_SHARE of R1; the report says so.  NGC 5090's two innermost isophotes (3 and 4.5 px,
# a PSF of about 1 px) crossed, and the whole fit failed.
INNER_R0 = 10.0
# A first fit below BOOTSTRAP_UP_BELOW px whose bootstrap fits failed is also tried at these
# multiples of its radius (a close pair's START=AUTO, 0.35 of the separation, can be ~10 px)
BOOTSTRAP_UP = (1.5, 2.25, 3.4)
BOOTSTRAP_UP_BELOW = 30.0
INNER_FACTOR = 1.5
INNER_TRIES = 2
INNER_SHARE = 0.25


def _outer_law(a: np.ndarray, lnI: np.ndarray, law: str):
    """ln I(a) / I(a_n) beyond the last isophote a_n, from the outer profile: a function of a,
    or None (keep elliprof's own)."""
    an = a[-1]
    if law == "sersic":  # ln I = c - k a^(1/n), n chosen on a grid, on the outer 60 %
        sel = a >= 0.4 * an
        if sel.sum() < 4:
            sel = np.arange(a.size) >= a.size - 4
        best = None
        for n in np.arange(0.5, 8.01, 0.1):
            x = a[sel] ** (1 / n)
            k, c = np.polyfit(x, lnI[sel], 1)
            if k >= 0:
                continue
            err = float(np.sum((lnI[sel] - (c + k * x)) ** 2))
            if best is None or err < best[0]:
                best = (err, n, k)
        if best is None:
            return None
        _, n, k = best
        return lambda r: k * (r ** (1 / n) - an ** (1 / n))
    if law == "exp":  # the outer 30 %: ln I linear in a
        sel = a >= 0.7 * an
        if sel.sum() < 3:
            sel = np.arange(a.size) >= a.size - 3
        k = np.polyfit(a[sel], lnI[sel], 1)[0]
        if k >= 0:
            return None
        return lambda r: k * (r - an)
    if law == "cut":  # a Gaussian taper over a tenth of R1
        w = 0.1 * an
        return lambda r: -(((r - an) / w) ** 2)
    return None


def _decreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The closest non-increasing sequence (weighted least squares: pool adjacent
    violators)."""
    blocks = []  # [value, weight, count]
    for v, w in zip(values, weights):
        blocks.append([float(v), float(w), 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([np.full(n, v) for v, _w, n in blocks])


def _ellipse_radius(profile_row, shape) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx] + 0.5
    t = np.radians(float(profile_row["alpha"]) - 90.0)
    dx, dy = xx - float(profile_row["x0"]), yy - float(profile_row["y0"])
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    q = max(1.0 - float(profile_row["ellip"]), 0.05)
    return np.sqrt(u * u + (v / q) ** 2)


TAIL_RATIO = 1.06  # the rings of the measured tail: each 6 % wider in radius (at least 3 px)
TAIL_PIXELS = 30  # a ring is measured with at least this many pixels


def measured_tail(
    light: np.ndarray, usable: np.ndarray, r: np.ndarray, an: float, first: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The light beyond the last isophote, measured: the median of ``light`` (the image
    minus the sky and the companion's model) in rings of the last isophote's shape, made
    non-increasing outwards and never negative, starting from the last isophote's
    intensity ``first``.  Radii and values, or None when no ring has enough pixels."""
    rmax = float(r.max())
    edges = [an]
    while edges[-1] < rmax:
        edges.append(max(edges[-1] * TAIL_RATIO, edges[-1] + 3.0))
    edges = np.array(edges)
    sel = usable & (r > an) & np.isfinite(light)
    idx = np.digitize(r[sel], edges) - 1
    vals = light[sel]
    radii, meds, counts = [an], [first], [1e9]
    order = np.argsort(idx, kind="stable")
    idx, vals = idx[order], vals[order]
    bounds = np.searchsorted(idx, np.arange(len(edges) + 1))
    for k in range(len(edges) - 1):
        chunk = vals[bounds[k] : bounds[k + 1]]
        if chunk.size >= TAIL_PIXELS:
            radii.append(0.5 * (edges[k] + edges[k + 1]))
            meds.append(float(np.median(chunk)))
            counts.append(float(chunk.size))
    if len(radii) < 2:
        return None
    meds = _decreasing(np.array(meds), np.sqrt(np.array(counts)))
    return np.array(radii), np.maximum(meds, 0.0)


def tail_model(
    model: np.ndarray,
    profile,
    law: str,
    light: Optional[np.ndarray] = None,
    usable: Optional[np.ndarray] = None,
) -> np.ndarray:
    """elliprof extrapolates its model beyond the last isophote to the image's corners, with
    ln I linear in r^(1/4) from the last two isophotes: right for a de Vaucouleurs profile,
    much too bright for a shallower one or when the last two are noisy.  In a pair, that
    light is subtracted from the companion's image.  Here the model beyond the last isophote
    is replaced, on the last isophote's ellipse (without harmonics), by: ``measured``, the
    light measured there (``light``: the image minus the sky and the companion's model, on
    the ``usable`` pixels), continued by a Sersic law where no ring can be measured; or the
    last isophote's intensity continued by ``law`` (sersic, exp, cut)."""
    if law == "elliprof" or model is None or profile is None:
        return model
    p = profile[np.isfinite(profile["I0"]) & (profile["I0"] > 0)]
    if len(p) < 4:
        return model
    a = p["Rmaj"].to_numpy(float)
    lnI = np.log(p["I0"].to_numpy(float))
    last = p.iloc[-1]
    an = a[-1]
    r = _ellipse_radius(last, model.shape)
    beyond = r > an
    out = model.copy()
    first = float(last["I0"])
    sersic = _outer_law(a, lnI, "sersic" if law == "measured" else law)
    if law == "measured" and light is not None:
        ok = np.ones(model.shape, bool) if usable is None else usable
        tail = measured_tail(light, ok, r, an, first)
        if tail is not None:
            radii, values = tail
            rb = r[beyond]
            out[beyond] = np.interp(rb, radii, values)
            further = rb > radii[-1]  # beyond the last ring measured: the Sersic law
            if further.any() and sersic is not None and values[-1] > 0:
                shift = sersic(np.array([radii[-1]]))[0]
                vals = out[beyond]
                vals[further] = values[-1] * np.exp(sersic(rb[further]) - shift)
                out[beyond] = vals
            return out
    if sersic is None:
        return model
    out[beyond] = first * np.exp(sersic(r[beyond]))
    return out


def outer_support(profile, usable: np.ndarray, count: int = 3, points: int = 180) -> List[dict]:
    """For the outermost isophotes: the fraction of points inside the frame and the
    fraction on usable pixels (in the frame, not left out of the fit)."""
    ny, nx = usable.shape
    out = []
    for x, y, a, b, angle in profile_ellipses(profile)[-count:]:
        phi = np.linspace(0, 2 * np.pi, points, endpoint=False)
        t = math.radians(angle)
        px = x + a * np.cos(phi) * math.cos(t) - b * np.sin(phi) * math.sin(t)
        py = y + a * np.cos(phi) * math.sin(t) + b * np.sin(phi) * math.cos(t)
        i, j = np.floor(px).astype(int), np.floor(py).astype(int)
        inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        good = np.zeros(points, bool)
        good[inside] = usable[j[inside], i[inside]]
        out.append(
            {
                "rmaj": a,
                "in_frame": float(inside.mean()),
                "usable": float(good.mean()),
                "cut_by_edge": bool(not inside.all()),
            }
        )
    return out


# ---------------------------------------------------------------- the run
Fitter = Callable[..., ArrayFit]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def run(
    snap: Snapshot,
    progress: Optional[Callable[[dict], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    fitter: Fitter = fit_array,
) -> PairResult:
    """Run the pair fit on a snapshot (see the module's description).

    Every fit works on the original image D minus the companion's current model (D - M_B
    for A, D - M_A for B), made afresh from D each time; D itself is never changed.  Each
    galaxy's R1 follows its schedule (Component.schedule) for the first cycles, then the
    cycles continue at the radii reached."""
    s = snap.settings
    comps = dict(snap.components)  # R0 may be raised during the run (INNER_R0)
    image = snap.image
    finite = np.isfinite(image)
    result = PairResult(status="incomplete_initialization", run_id=snap.run_id)
    start = time.perf_counter()
    sched = {cid: comps[cid].schedule(s.steps) for cid in ("A", "B")}
    n_grow = max(len(v) for v in sched.values()) - 1
    result.schedules = {cid: list(v) for cid, v in sched.items()}
    total_calls = 2 + 2 * (n_grow + s.cycles + (s.polish if s.damp < 1 else 0))
    counter = {"n": 0}

    def say(**info):
        if progress is not None:
            info.update(call=counter["n"], calls=total_calls)
            progress(info)

    def stop_requested():
        return cancelled is not None and cancelled()

    def fit_component(
        cid: str,
        companion: Optional[Commit],
        r1_start: float,
        cycle: str,
        adaptive: Optional[bool] = None,
        bootstrap: bool = False,
        first: bool = False,
    ):
        """A candidate for ``cid`` (with adaptive-R1 retries): a Commit, or None.  It is
        fitted on D - (the companion's model), made here from D."""
        comp = comps[cid]
        other = "B" if cid == "A" else "A"
        conditional = image if companion is None else image - companion.model
        source = "D" if companion is None else f"D - M_{other}"
        companion_tag = "none" if companion is None else f"M_{other} ({companion.tag})"
        cos3x, cos4x = comp.harmonics()
        mask = snap.mask_for(cid, first=first)
        usable = finite & ~mask
        r1 = r1_start
        adaptive = comp.adaptive if adaptive is None else adaptive
        tries = 4 if adaptive else 1
        inner_tries = INNER_TRIES
        attempt = -1
        while attempt + 1 < tries:
            attempt += 1
            if stop_requested():
                raise Cancelled()
            counter["n"] += 1 if attempt == 0 and not bootstrap else 0
            say(phase="fit", cycle=cycle, component=cid, r1=r1, retry=attempt, input=source)
            fit = fitter(
                conditional,
                comp.x0,
                comp.y0,
                comp.r0,
                r1,
                nr_for(comp, r1),
                niter=s.niter,
                sky=s.sky,
                masked=mask,
                rmstar=s.rmstar,
                timeout=s.timeout,
                header=snap.header,
                fixctr=comp.fixctr,
                cos3x=cos3x,
                cos4x=cos4x,
                rlaw=s.rlaw,
                gain=s.gain,
                tie=s.tie,
            )
            ok, reason, kind, radius = check(fit, comp, r1, finite)
            result.attempted_r1[cid] = r1
            drift = _centre_drift(fit.profile, comp) if ok else None
            call = Call(
                cycle,
                cid,
                r1,
                _clean_arguments(fit.arguments),
                fit.returncode,
                fit.seconds,
                ok,
                reason,
                fit.precision,
                input=source,
                companion=companion_tag,
                centre_drift=drift,
                mask_pixels=int(mask.sum()),
            )
            result.calls.append(call)
            entry = {
                "call": len(result.calls),
                "cycle": cycle,
                "galaxy": cid,
                "input": source,
                "companion": companion_tag,
                "r1": r1,
                "passed": ok,
                "reason": reason,
                "model": fit.model if ok else None,
                "profile": fit.profile if ok else None,
                "companion_model": None if companion is None else companion.model,
            }
            result.history.append(entry)
            say(
                phase="call",
                cycle=cycle,
                component=cid,
                r1=r1,
                input=source,
                passed=ok,
                reason=reason,
                drift=drift,
                index=len(result.calls),
                ellipses=profile_ellipses(fit.profile) if ok else None,
            )
            if ok:
                support = outer_support(fit.profile, usable)
                return Commit(
                    model=tail_model(fit.model, fit.profile, s.tail, conditional - s.sky, usable),
                    profile=fit.profile,
                    ellipses=profile_ellipses(fit.profile),
                    tag=cycle,
                    r1_call=r1,
                    support=support,
                    coverage_uncertain=any(x["usable"] < OUTER_SUPPORT for x in support),
                    bootstrap=bootstrap,
                    requested_r1=comp.r1,
                )
            result.failures[cid] = f"{cycle}: {reason} (R1 {r1:.1f})"
            result.diagnostics[cid] = fit
            if kind == "outer" and radius is not None and radius <= INNER_R0 and inner_tries:
                # the innermost isophotes, within a few PSF widths, failed: start further out
                new_r0 = round(INNER_FACTOR * radius, 1)
                if new_r0 < INNER_SHARE * r1:
                    inner_tries -= 1
                    comp = comps[cid] = replace(comp, r0=new_r0)
                    result.r0_raised[cid] = new_r0
                    attempt -= 1  # not an adaptive try
                    continue
            if not adaptive or kind != "outer" or radius is None or radius < 0.65 * r1:
                break
            new = min(0.9 * r1, 0.95 * radius)
            if not (new > 1.5 * comp.r0 and new >= 0.5 * comp.r1):
                result.failures[cid] += f" (adaptive R1 stops: {new:.1f} is below its bounds)"
                break
            r1 = new
        return None

    def initialise(cid: str, companion: Optional[Commit]):
        """The first model of ``cid``: at the first radius of its schedule, or else a
        partial bootstrap fit at 0.7, 0.5 or 0.35 of it (only to start the cycles; never
        part of a result)."""
        r_first = sched[cid][0]
        grows = len(sched[cid]) > 1
        c0 = fit_component(
            cid, companion, r_first, "init", adaptive=False if grows else None, first=True
        )
        if c0 is not None or not s.bootstrap:
            return c0
        failure = result.failures.get(cid, "")
        for fraction in (0.7, 0.5, 0.35):
            r1 = fraction * r_first
            if r1 <= 1.5 * comps[cid].r0:
                break
            c = fit_component(
                cid,
                companion,
                r1,
                "init (bootstrap)",
                adaptive=False,
                bootstrap=True,
                first=True,
            )
            if c is not None:
                result.bootstraps[cid] = r1
                return c
        # a small first radius (a close pair) may be too small for elliprof: further out
        for factor in BOOTSTRAP_UP:
            r1 = factor * r_first
            if r_first >= BOOTSTRAP_UP_BELOW or r1 > comps[cid].r1:
                break
            c = fit_component(
                cid, companion, r1, "init (bootstrap)", adaptive=False, bootstrap=True, first=True
            )
            if c is not None:
                result.bootstraps[cid] = r1
                return c
        result.failures[cid] = (
            failure + " (no bootstrap fit at 0.7, 0.5 or 0.35 R1, or further out, passed either)"
        )
        return None

    committed: Dict[str, Commit] = {}
    checkpoint: Optional[Tuple[str, Dict[str, Commit]]] = None
    try:
        a0 = initialise("A", None)
        if a0 is None:
            result.status = "incomplete_initialization"
            result.stability_reason = f"A: {result.failures.get('A', '')}"
            return _finish(result, snap, start, None, committed)
        committed["A"] = a0
        b0 = initialise("B", a0)  # B on D - M_A
        if b0 is None:
            result.status = "incomplete_initialization"
            result.stability_reason = f"B: {result.failures.get('B', '')}"
            result.partial = {"A": a0}
            return _finish(result, snap, start, None, committed)
        committed["B"] = b0
        if not (a0.bootstrap or b0.bootstrap):
            checkpoint = ("0", dict(committed))
        last_ok = {"A": True, "B": True}
        held: Dict[str, float] = {}  # galaxies whose growth failed: the radius they stay at
        state = {"streak": 0, "stopped": None, "checkpoint": checkpoint}
        # what is subtracted from D when the companion is fitted: its latest model, or with
        # damping (DAMP < 1) a blend of its new and previous ones
        used: Dict[str, object] = {"A": committed["A"], "B": committed["B"]}

        def take(cid: str, cand: Commit, alpha: float) -> None:
            committed[cid] = cand
            if alpha >= 1:
                used[cid] = cand
            else:
                used[cid] = _Used(
                    alpha * cand.model + (1 - alpha) * used[cid].model,
                    f"{cand.tag}, damped {alpha:g}",
                )

        def one_cycle(k: int, growing: bool, alpha: float, polish: bool) -> bool:
            """One A -> B cycle; True when the run should stop here."""
            tag = str(k)
            prev = dict(committed)
            prev_t = prev["A"].model + prev["B"].model
            radii = {}
            stopped_on_failure = None
            for cid in ("A", "B"):
                other = "B" if cid == "A" else "A"
                want = sched[cid][min(k, len(sched[cid]) - 1)]
                current = committed[cid]
                grows = want > current.r1_call + 1e-9 and cid not in held and not polish
                if grows and (growing or s.on_fail in ("retry", "reduce")):
                    r1, adaptive = want, False  # a growth step
                elif cid in held and s.on_fail in ("hold", "reduce"):
                    r1, adaptive = current.r1_call, False
                else:
                    r1, adaptive = current.next_r1, None
                cand = fit_component(cid, used[other], r1, tag, adaptive=adaptive)
                tried = [r1]
                if cand is None and s.on_fail == "reduce" and r1 > current.r1_call + 1e-9:
                    # a smaller step: halfway, then a quarter of the way, from the last
                    # valid radius to the one that failed
                    for fraction in (0.5, 0.25):
                        step = r1 - current.r1_call
                        r_try = current.r1_call + fraction * step
                        if fraction * step < 0.02 * current.r1_call:
                            break
                        tried.append(r_try)
                        cand = fit_component(cid, used[other], r_try, tag, adaptive=False)
                        if cand is not None:
                            r1 = r_try
                            result.notes_growth.append(
                                f"{cid}: R1 {tried[0]:.1f} failed in cycle {k} "
                                f"({result.failures.get(cid, '')}); it grew to {r_try:.1f} "
                                "instead (a smaller step)"
                            )
                            break
                radii[cid] = r1 if cand is not None else tried[0]
                last_ok[cid] = cand is not None
                if cand is not None:
                    take(cid, cand, alpha)
                else:
                    if tried[0] > current.r1_call + 1e-9 and cid not in held:
                        if s.on_fail in ("hold", "reduce"):
                            held[cid] = current.r1_call
                        smaller = (
                            f" (and {', '.join(f'{x:.1f}' for x in tried[1:])})"
                            if len(tried) > 1
                            else ""
                        )
                        result.notes_growth.append(
                            f"{cid} could not grow to R1 {tried[0]:.1f}{smaller} in cycle {k}; "
                            f"it keeps its model at R1 {current.r1_call:.1f}: "
                            f"{result.failures.get(cid, '')}"
                        )
                    if s.on_fail == "stop":
                        stopped_on_failure = cid
                        break
            m = _metrics(snap, committed, prev["A"].model, prev["B"].model, prev_t, s.converge)
            m.update(
                cycle=k,
                A_ok=last_ok["A"],
                B_ok=last_ok.get("B", False) if stopped_on_failure != "A" else False,
                A_tag=committed["A"].tag,
                B_tag=committed["B"].tag,
                A_r1=radii.get("A"),
                B_r1=radii.get("B"),
                growing=growing,
                damping=alpha,
                polish=polish,
            )
            for cid in ("A", "B"):
                m.update(
                    {
                        f"{cid}_{key}": v
                        for key, v in _profile_change(committed[cid], prev[cid]).items()
                    }
                )
            result.metrics.append(m)
            if growing:
                result.growth.append(
                    {
                        "step": k,
                        "A_r1": radii.get("A"),
                        "A_ok": last_ok["A"],
                        "B_r1": radii.get("B"),
                        "B_ok": m["B_ok"],
                        "frozen": sorted(held),
                    }
                )
            say(phase="cycle", cycle=tag, metrics=m)
            joint = m["A_ok"] and m["B_ok"]
            # a checkpoint is a joint cycle of undamped fits; with damping, only the
            # final undamped (polish) cycles count, so the result is ELLIPROF's own models
            undamped = alpha >= 1
            if joint and undamped and not (committed["A"].bootstrap or committed["B"].bootstrap):
                state["checkpoint"] = (tag, dict(committed))
            if stopped_on_failure:
                state["stopped"] = stopped_on_failure
                return True
            if polish:
                return False
            # converged cycles count once the radii have stopped changing
            at_target = k > n_grow and all(
                prev[c].r1_call == committed[c].r1_call for c in ("A", "B")
            )
            state["streak"] = state["streak"] + 1 if (at_target and joint and m["stable"]) else 0
            m["converged_streak"] = state["streak"]
            if s.consecutive and state["streak"] >= s.consecutive:
                m["converged"] = True
                return True
            return False

        k = 0
        for k in range(1, n_grow + s.cycles + 1):
            if one_cycle(k, k <= n_grow, s.damp, False):
                break
        if s.damp < 1 and not state["stopped"]:
            # undamped final cycles: the models subtracted are the fitted models again
            used.update(committed)
            state["checkpoint"] = None if state["checkpoint"] is None else state["checkpoint"]
            for j in range(1, s.polish + 1):
                if one_cycle(k + j, False, 1.0, True):
                    break
        checkpoint = state["checkpoint"]
        stopped_on_failure = state["stopped"]
        last = result.metrics[-1]
        if checkpoint is None:
            result.status = "incomplete_initialization"
            result.stability_reason = (
                "no cycle in which both galaxies passed at their full radii after the "
                "bootstrap: "
                + "; ".join(
                    f"{c}: {result.failures.get(c, '')}" for c in "AB" if c in result.failures
                )
            )
            result.partial = dict(committed)
            return _finish(result, snap, start, None, committed)
        if last["A_ok"] and last["B_ok"]:
            if last["stable"]:
                result.status = "stable_valid_pair"
            else:
                result.status = "valid_pair_not_stable"
                result.stability_reason = last["reason"]
        else:
            failed = [c for c in ("A", "B") if not last[f"{c}_ok"]]
            result.status = "retained_after_failed_update"
            result.stability_reason = "; ".join(
                f"{c} failed in cycle {last['cycle']}: {result.failures.get(c, '')}" for c in failed
            )
            if stopped_on_failure:
                result.stability_reason += " (stopped there: on failure = stop)"
    except Cancelled:
        result.status = "cancelled"
        result.stability_reason = "stopped by the user"
    return _finish(result, snap, start, checkpoint, committed)


@dataclass
class _Used:
    """A damped model: what is subtracted from D when the companion is fitted."""

    model: np.ndarray
    tag: str


def _centre_drift(profile, comp: Component) -> Optional[float]:
    """The largest distance of the fitted centres from X0, Y0 (pixels)."""
    try:
        return float(np.nanmax(np.hypot(profile["x0"] - comp.x0, profile["y0"] - comp.y0)))
    except (KeyError, ValueError):
        return None


def _profile_change(new: Commit, old: Commit) -> dict:
    """How a galaxy's profile changed from one cycle to the next, over the radii both
    cover: the largest relative change of I0, of the ellipticity, of the angle (where the
    ellipticity is above 0.05) and of the centre.  Diagnostics only."""
    if new is old:
        return {"dI0": 0.0, "dE": 0.0, "dPA": 0.0, "dXY": 0.0}
    a, b = new.profile, old.profile
    r = a["Rmaj"].to_numpy(float)
    r_old = b["Rmaj"].to_numpy(float)
    keep = (r >= r_old.min()) & (r <= r_old.max())
    if keep.sum() < 2:
        return {"dI0": None, "dE": None, "dPA": None, "dXY": None}
    r = r[keep]

    def at(col):
        return np.interp(r, r_old, b[col].to_numpy(float)), a[col].to_numpy(float)[keep]

    i_old, i_new = at("I0")
    e_old, e_new = at("ellip")
    p_old, p_new = at("alpha")
    x_old, x_new = at("x0")
    y_old, y_new = at("y0")
    with np.errstate(divide="ignore", invalid="ignore"):
        di = np.nanmax(np.abs(i_new / i_old - 1))
    dpa = np.abs(((p_new - p_old) + 90) % 180 - 90)
    dpa = dpa[(e_new > PA_MIN_ELLIP) & (e_old > PA_MIN_ELLIP)]
    return {
        "dI0": float(di),
        "dE": float(np.nanmax(np.abs(e_new - e_old))),
        "dPA": float(dpa.max()) if dpa.size else None,
        "dXY": float(np.nanmax(np.hypot(x_new - x_old, y_new - y_old))),
    }


def with_sky(snap: Snapshot, factor: float) -> Snapshot:
    """The snapshot with the sky factor x base (everything else shared, not copied)."""
    s = snap.settings
    base = s.sky_base if s.sky_base is not None else s.sky
    return replace(snap, settings=replace(s, sky=factor * base, sky_factor=factor))


SKY_RESPONSE = 0.5  # the least share of a sky change the residual sky must follow
SKY_STEP = 2.0  # the largest change of the sky from one sky fit to the next, in sigma


def run_sky(
    snap: Snapshot,
    support: Optional[np.ndarray] = None,
    progress: Optional[Callable[[dict], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    fitter: Fitter = fit_array,
) -> PairResult:
    """The pair fit, refitted with the sky factor updated from the residual sky.

    With SKYITER=n (and the sky support of a measurement), the pair is fitted up to n times:
    after each fit, the 3-sigma clipped median m of image - sky - A - B on the support gives
    the next factor, f + m / base.  It stops early once |m| is below 0.01 sigma of the
    residual.  The result is the complete fit with the smallest |m|; every fit is listed in
    ``sky_iterations``.  Without SKYITER (or support) it is :func:`run`.

    When the galaxies' light fills the frame, the sky and their outer light are degenerate:
    a lower sky makes the outer isophotes (and the model beyond R1) brighter by as much,
    the residual on the sky pixels does not change, and the factor would run away.  So
    when the residual follows less than SKY_RESPONSE of a change of the sky, the iterations
    stop and the fit at the sky asked for is kept."""
    from . import sky as skies

    s = snap.settings
    if s.sky_iter <= 0 or support is None:
        return run(snap, progress, cancelled, fitter)
    base = s.sky_base if s.sky_base is not None else s.sky
    use = support & ~snap.fit_mask
    factor = s.sky_factor
    history: List[dict] = []
    best: Optional[Tuple[float, PairResult, int]] = None
    last = None
    first: Optional[Tuple[float, PairResult, int]] = None
    previous = None  # (sky, median) of the last fit
    degenerate = False
    for k in range(s.sky_iter):
        current = with_sky(snap, factor)

        def say(info, k=k, factor=factor):
            if progress is not None:
                info.update(sky_iteration=k + 1, sky_iterations=s.sky_iter, sky_factor=factor)
                progress(info)

        result = run(current, say, cancelled, fitter)
        last = result
        entry = {
            "iteration": k + 1,
            "factor": factor,
            "sky": current.settings.sky,
            "status": result.status,
            "calls": len(result.calls),
            "seconds": result.seconds,
        }
        if result.residual is None:
            entry["note"] = "no complete pair: nothing to measure"
            history.append(entry)
            break
        level = skies.residual_level(result.residual, use, base, factor)
        entry.update({key: level.get(key) for key in ("median", "sigma", "pixels")})
        entry["next_factor"] = level.get("next_factor")
        history.append(entry)
        if "median" not in level:
            break
        if first is None:
            first = (abs(level["median"]), result, k)
        step = current.settings.sky - previous[0] if previous is not None else 0.0
        if step and abs(step) >= 0.5 * level["sigma"]:  # big enough to measure the response
            # the median falls by the sky's rise when the models stay; 0 when they absorb it
            response = (previous[1] - level["median"]) / step
            entry["response"] = response
            if response < SKY_RESPONSE:
                entry["note"] = (
                    f"the residual sky followed {100 * response:.0f} % of the sky's change: the "
                    "sky and the galaxies' outer light are degenerate on this frame; the fit "
                    "at the sky asked for is kept"
                )
                degenerate = True
                break
        previous = (current.settings.sky, level["median"])
        if best is None or abs(level["median"]) < best[0]:
            best = (abs(level["median"]), result, k)
        if result.status == "cancelled" or (cancelled is not None and cancelled()):
            break
        if abs(level["median"]) < 0.01 * max(level["sigma"], 1e-300):
            entry["note"] = "converged: |median| below 0.01 sigma"
            break
        nxt = level.get("next_factor")
        if nxt is None or not 0.5 <= nxt <= 1.5:
            entry["note"] = "the next factor would leave 0.5 to 1.5: stopped"
            break
        # at most SKY_STEP sigma at a time: a degenerate sky is caught before it runs away
        largest = SKY_STEP * level["sigma"] / base if base else abs(nxt - factor)
        if abs(nxt - factor) > largest:
            nxt = factor + math.copysign(largest, nxt - factor)
            entry["note"] = f"the step was limited to {SKY_STEP:g} sigma"
        factor = nxt
    if degenerate and first is not None:
        best = first
    chosen = best[1] if best is not None else last
    if best is not None:
        history[best[2]]["chosen"] = True
    chosen.sky_iterations = history
    chosen.snapshot = with_sky(snap, chosen_factor(chosen, snap.settings))
    return chosen


def variants(snap: Snapshot) -> List[Tuple[str, Snapshot]]:
    """The snapshots TRYDAMP=, TRYFIX= and TRYFACTOR= ask for (every combination), each
    with a label; just the snapshot when none is asked for."""
    s = snap.settings
    if not (s.try_damp or s.try_fix or s.try_factor):
        return [("", snap)]
    out = []
    for damp in s.try_damp or [s.damp]:
        for fix in s.try_fix or [None]:
            for factor in s.try_factor or [None]:
                comps = {}
                for cid, c in snap.components.items():
                    changes = {}
                    if fix is not None:
                        changes["fixctr"] = fix
                    if factor is not None and c.start and not c.radii:
                        changes.update(factor=factor, step=None)
                    comps[cid] = replace(c, **changes)
                label = f"damp {damp:g}"
                if fix is not None:
                    label += ", centres " + {0: "free", 1: "fixed", 2: "median"}[fix]
                if factor is not None:
                    label += f", growth x{factor:g}"
                out.append((label, replace(snap, settings=replace(s, damp=damp), components=comps)))
    return out


def quality(result: PairResult, snap: Snapshot, inner: float = 10.0) -> dict:
    """How well the two models together describe the pair: the RMS of the residual (clipped
    at 10 sigma, so unmasked stars do not dominate) over the evaluation pixels farther than
    ``inner`` pixels from both centres, in units of the pixel noise; 1 is a residual of pure
    noise.  Also the ring structure around each galaxy (see residual_rings) as a fraction of
    the model plus 3 sigma.  On 12 simulated pairs this score tracked the error of A + B
    well (median rank correlation 0.92 across fit variants) but not how the light is split
    between A and B: a residual cannot tell that split.  Smaller is better; it compares fits
    of one image and is not an absolute measure."""
    from . import sky as skies

    if result.residual is None:
        return {}
    good = snap.eval & np.isfinite(result.residual)
    sigma = skies.local_noise(result.residual, good) if good.sum() >= 50 else float("nan")
    if not np.isfinite(sigma) or sigma <= 0:
        return {}
    ny, nx = snap.image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    outside = good.copy()
    for c in snap.components.values():
        outside &= np.hypot(xx + 0.5 - c.x0, yy + 0.5 - c.y0) > inner
    if outside.sum() < 50:
        return {}
    r = np.clip(result.residual[outside], -10 * sigma, 10 * sigma)
    score = float(np.sqrt(np.mean(r * r)) / sigma)
    rings = residual_rings(result, snap, requested=True)
    per = {}
    for cid, rows in rings.items():
        vals = []
        for x in rows:
            asym = (
                abs(x["facing"] - x["away"]) / 2
                if x["facing"] is not None and x["away"] is not None
                else 0.0
            )
            vals.append((abs(x["residual"]) + asym) / (abs(x["model"]) + 3 * sigma))
        per[cid] = float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")
    return {"score": score, "per_galaxy": per, "sigma": float(sigma), "pixels": int(outside.sum())}


PLANE_BOX = 32  # SKYPLANE: the boxes of the residual the tilt is fitted to (pixels)


def remove_tilt(snap: Snapshot, first: PairResult) -> Snapshot:
    """SKYPLANE: a plane fitted to the residual of a first pass (image - sky - A - B; the
    medians of PLANE_BOX boxes of unmasked pixels, clipped), and, when its gradient is
    significant (3 times its error), the image minus its tilt (zero at the frame's centre:
    the sky level stays the one asked for).  The elliptical models cannot take up a
    gradient, so the residual shows it where the sky alone cannot be measured."""
    from . import sky as skies

    if not first.complete or first.residual is None:
        return replace(snap, notes=snap.notes + ["SKYPLANE: no complete first pass; no plane"])
    res = first.residual
    good = ~snap.fit_mask & np.isfinite(res)
    values = res[good]
    sigma = skies.mad(values)
    good &= np.abs(res - np.median(values)) < 5 * sigma
    fitted = skies.plane_by_boxes(res, good, sigma, PLANE_BOX)
    if fitted is None:
        return replace(snap, notes=snap.notes + ["SKYPLANE: too few boxes for a plane"])
    plane, errors = fitted[0], fitted[1]
    g = math.hypot(plane["gx"], plane["gy"])
    err = math.hypot(errors["gx"], errors["gy"]) / math.sqrt(2) or float("inf")
    text = (
        f"SKYPLANE: gradient {1000 * plane['gx']:+.4g}, {1000 * plane['gy']:+.4g} per 1000 px "
        f"(x, y; ± {1000 * errors['gx']:.2g}, {1000 * errors['gy']:.2g}) in the residual of "
        "the fit kept"
    )
    if g < 3 * err:
        return replace(snap, notes=snap.notes + [text + ": not significant, nothing removed"])
    tilt = skies.plane_image(dict(plane, s0=0.0), snap.image.shape)
    return replace(
        snap,
        image=snap.image - tilt,
        tilt=dict(plane, s0=0.0),
        notes=snap.notes + [text + ": removed from the image (zero at its centre)"],
    )


# WINGS=1: the light of bright sources beyond their masks (the wings of a star, its
# diffraction spikes, a halo), found in the residual of a first pass.  The residual is smoothed
# (WING_SMOOTH px) and compared with its own local spread; what is above WING_SIGMA times it
# and joined to a masked patch of at least WING_SEED px is added to the mask, at most
# WING_REACH px from that patch and never within WING_PROTECT px of a galaxy's centre.  Specks
# of SBF or noise are not joined to a mask, so they stay.
WING_SMOOTH = 1.5
WING_SIGMA = 2.5
WING_SEED = 40
WING_REACH = 150
WING_PROTECT = 12


def _local_spread(img: np.ndarray, good: np.ndarray, block: int = 8, size: int = 5):
    """A robust local spread (1.4826 x the median |x|) over size x size blocks of block px."""
    from scipy import ndimage

    ny, nx = img.shape
    by, bx = -(-ny // block), -(-nx // block)
    pad = np.full((by * block, bx * block), np.nan)
    pad[:ny, :nx] = np.where(good, np.abs(img), np.nan)
    import warnings

    with warnings.catch_warnings():  # blocks with no good pixel: filled below
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(
            pad.reshape(by, block, bx, block).transpose(0, 2, 1, 3).reshape(by, bx, -1), axis=2
        )
    fill = np.nanmedian(med) if np.isfinite(med).any() else 1.0
    med = np.where(np.isfinite(med), med, fill)
    med = ndimage.median_filter(med, size=size, mode="nearest")
    up = np.kron(med, np.ones((block, block)))[:ny, :nx]
    return 1.4826 * up


def mask_wings(snap: Snapshot, first: PairResult) -> Snapshot:
    """WINGS=1: the snapshot with the wings of bright sources masked (see WING_*)."""
    from scipy import ndimage

    if not first.complete or first.residual is None:
        return replace(snap, notes=snap.notes + ["WINGS: no complete first pass; nothing masked"])
    res = first.residual
    good = ~snap.fit_mask & np.isfinite(res)
    w = ndimage.gaussian_filter(good.astype(float), WING_SMOOTH)
    sm = ndimage.gaussian_filter(np.where(good, res, 0.0), WING_SMOOTH)
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = np.where(w > 0.3, sm / w, 0.0)
    spread = _local_spread(sm, good)
    bright = good & (sm > WING_SIGMA * spread)
    labels, n = ndimage.label(snap.fit_mask & ~snap.invalid)
    if not n:
        return replace(snap, notes=snap.notes + ["WINGS: no masked source to look around"])
    areas = np.bincount(labels.ravel())
    big = np.zeros(areas.size, bool)
    big[1:] = areas[1:] >= WING_SEED
    seeds = big[labels]
    near = ndimage.binary_dilation(seeds, iterations=2)
    joined, _m = ndimage.label(bright | near)
    touching = np.unique(joined[near & (joined > 0)])
    found = np.isin(joined, touching) & bright
    found &= ndimage.distance_transform_edt(~seeds) <= WING_REACH
    ny, nx = res.shape
    yy, xx = np.mgrid[0:ny, 0:nx] + 0.5
    for c in snap.components.values():
        found &= np.hypot(xx - c.x0, yy - c.y0) > WING_PROTECT
    count = int(found.sum())
    if not count:
        return replace(snap, notes=snap.notes + ["WINGS: no light found beyond the masks"])
    grown = ndimage.binary_dilation(found, iterations=1) & ~snap.fit_mask
    for c in snap.components.values():
        grown &= np.hypot(xx - c.x0, yy - c.y0) > WING_PROTECT
    note = (
        f"WINGS: {int(grown.sum())} px around {int(big[1:].sum())} masked sources (their wings "
        "and spikes in the residual of a first pass) added to the mask"
    )
    return replace(
        snap,
        fit_mask=snap.fit_mask | grown,
        eval=snap.eval & ~grown,
        fit_mask_label=(snap.fit_mask_label + " + wings").strip(" +"),
        notes=snap.notes + [note],
    )


def refine_auto(snap: Snapshot, first: PairResult) -> Tuple[Snapshot, List[str]]:
    """R1=AUTO measured again for each galaxy on the image minus the sky and minus the
    other's model of a first pass (and START=AUTO from it).  Measured on the image itself,
    the light of a small companion inside a large galaxy's halo never falls to the noise on
    its far side: its R1 came out many times its size, and its model took the halo."""
    if not first.complete or not snap.auto or not snap.auto_sigma:
        return snap, []
    comps, notes = dict(snap.components), []
    separation = math.hypot(comps["A"].x0 - comps["B"].x0, comps["A"].y0 - comps["B"].y0)
    for cid, (r1_auto, start_auto) in snap.auto.items():
        if not r1_auto:
            continue
        other = "B" if cid == "A" else "A"
        c, o = comps[cid], snap.components[other]
        light = snap.image - snap.settings.sky - first.commits[other].model
        r1 = auto_r1(light, snap.fit_mask, (c.x0, c.y0), (o.x0, o.y0), snap.auto_sigma)
        if r1 <= c.r0 or r1 == c.r1:
            continue
        changes = {"r1": r1}
        if start_auto:
            start = auto_start(r1, separation)
            changes.update(
                start=start, factor=(c.factor or AUTO_FACTOR) if start else None, step=None
            )
        notes.append(
            f"{cid}: R1=AUTO measured again on the image minus {other}'s model: "
            f"{c.r1:g} -> {r1:g}"
        )
        comps[cid] = replace(c, **changes)
    if not notes:
        return snap, []
    ev = np.zeros(snap.image.shape, bool)
    for c in comps.values():
        maskops.circle(ev, c.x0, c.y0, c.r1, True)
    ev &= ~snap.fit_mask
    return replace(snap, components=comps, eval=ev, notes=snap.notes + notes), notes


def run_search(
    snap: Snapshot,
    support: Optional[np.ndarray] = None,
    progress: Optional[Callable[[dict], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    fitter: Fitter = fit_array,
) -> PairResult:
    """Fit every variant TRY... asks for (each with SKYITER if asked) and keep the one whose
    two models describe the pair best (see choose()).  One result, not an average; every
    variant is listed in ``search``.  With R1=AUTO, a first pass (the snapshot as asked)
    gives the models on which the radii are measured again (refine_auto).

    PARALLEL=n fits n variants at a time.  TRYCOS3X= and TRYCOS4X= then try each galaxy's
    harmonics on the fit kept (choose_harmonics).  The mean of the fits close to the best is
    kept as well (ensemble())."""
    t0 = time.perf_counter()
    result = _search(snap, support, progress, cancelled, fitter)
    if snap.settings.runs > 1 and result.complete and not (cancelled is not None and cancelled()):
        result = _repeat(result, support, progress, cancelled, fitter)
    result.wall = time.perf_counter() - t0
    return result


def _repeat(result, support, progress, cancelled, fitter) -> PairResult:
    """RUNS: the fit kept, fitted again RUNS - 1 times with R0 and R1 changed at random
    (jittered()); the mean of the complete ones (ensemble()) replaces the search's mean."""
    kept_snap = result.snapshot
    s = kept_snap.settings
    more = _fit_each(jittered(kept_snap, s), support, progress, cancelled, fitter, s.parallel)
    rows = [_search_row("the fit kept", result, kept_snap)]
    rows += [dict(_search_row(label, r, sn), repeat=True) for label, r, sn in more]
    result.ensemble = ensemble(
        [(result, kept_snap)] + [(r, sn) for _l, r, sn in more], rows, REPEAT_SHARE, True
    )
    result.repeats = rows
    return result


def _search(snap, support, progress, cancelled, fitter) -> PairResult:
    if snap.settings.harmonic_search:
        snap = all_harmonics(snap)
    auto = any(r for r, _s in snap.auto.values())
    if auto or snap.settings.wings:

        def first_say(info):
            if progress is not None:
                info.update(variant="first pass (R1 on D minus the companion)")
                progress(info)

        first = run(snap, first_say, cancelled, fitter)
        if snap.settings.wings:
            snap = mask_wings(snap, first)
            snap = replace(snap, settings=replace(snap.settings, wings=False))
    if auto:
        snap, _notes = refine_auto(snap, first)
        # a halo that fills the frame takes up any change of the sky: no sky iterations
        filling = [
            cid
            for cid, (r1_auto, _s) in snap.auto.items()
            if r1_auto
            and reaches_edge(
                snap.components[cid].r1,
                (snap.components[cid].x0, snap.components[cid].y0),
                snap.image.shape,
            )
        ]
        if filling and snap.settings.sky_iter > 0:
            note = (
                f"no sky iterations: the light of {' and '.join(filling)} is above the noise "
                "out to the frame's edge, so the sky and the halo cannot be told apart here; "
                "the sky asked for is kept"
            )
            snap = replace(
                snap, settings=replace(snap.settings, sky_iter=0), notes=snap.notes + [note]
            )
    base = snap
    options = variants(snap)
    results = _fit_each(options, support, progress, cancelled, fitter, snap.settings.parallel)
    rows = [_search_row(label, r, sn) for label, r, sn in results]
    if not results:
        return run(snap, progress, cancelled, fitter)  # cancelled before any variant ran
    pairs_ = [(r, sn) for _l, r, sn in results]
    if len(pairs_) == 1:
        best, why = 0, ""
    else:
        best, why = choose(pairs_)
        rows[best]["chosen"] = True
    result, chosen_snap = pairs_[best]
    # TRYCOS3X / TRYCOS4X: each galaxy's harmonics in turn, tried on the fit kept
    for cid in ("A", "B"):
        tried = harmonic_variants(chosen_snap, cid)
        if not tried or (cancelled is not None and cancelled()):
            continue
        tried = [
            (label, replace(sn, settings=replace(sn.settings, sky_iter=0))) for label, sn in tried
        ]
        more = _fit_each(tried, None, progress, cancelled, fitter, snap.settings.parallel)
        extra = [_search_row(label, r, sn) for label, r, sn in more]
        for row in extra:
            row["harmonics"] = cid
        k, why_h = choose_harmonics(result, [(r, sn) for _l, r, sn in more])
        if k is not None:
            for row in rows:
                row.pop("chosen", None)
            extra[k]["chosen"] = True
            result, chosen_snap = more[k][1], more[k][2]
        if k is not None:
            why_h = f"{_harmonic_words(chosen_snap.components[cid])} {why_h}"
        why = (why + "; " if why else "") + f"{cid}: {why_h}"
        rows += extra
        pairs_ += [(r, sn) for _l, r, sn in more]
    base = replace(
        base,
        components={
            cid: replace(
                c,
                mode=chosen_snap.components[cid].mode,
                cos3x=chosen_snap.components[cid].cos3x,
                cos4x=chosen_snap.components[cid].cos4x,
            )
            for cid, c in base.components.items()
        },
    )
    base = replace(
        base,
        components={
            cid: replace(c, try_cos3x=[], try_cos4x=[]) for cid, c in base.components.items()
        },
    )
    result.ensemble = ensemble(pairs_, rows)
    if len(rows) > 1:
        result.search = rows
        result.search_choice = why
    result.snapshot = chosen_snap
    return _with_plane(result, base, support, progress, cancelled, fitter)


def _fit_each(options, support, progress, cancelled, fitter, workers: int):
    """run_sky on every (label, snapshot), ``workers`` at a time: a Python thread each,
    since the time goes into elliprof's subprocesses (and NumPy, which mostly lets go of
    the interpreter lock).  Returns (label, result, the snapshot of the fit) in the order
    given; a variant not started before a stop is left out."""

    def one(item):
        k, (label, variant) = item
        if cancelled is not None and cancelled():
            return None

        def say(info):
            if progress is not None:
                if len(options) > 1:
                    info.update(variant=label, variant_number=k, variants=len(options))
                progress(info)

        result = run_sky(variant, support, say, cancelled, fitter)
        chosen_snap = with_r0(result.snapshot or variant, result)
        result.quality = quality(result, chosen_snap)
        return label, result, chosen_snap

    items = list(enumerate(options, 1))
    if workers <= 1 or len(items) <= 1:
        out = [one(item) for item in items]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=min(workers, len(items)), thread_name_prefix="astrovista-variant"
        ) as pool:
            out = list(pool.map(one, items))
    return [x for x in out if x is not None]


def with_r0(snap: Snapshot, result: PairResult) -> Snapshot:
    """The snapshot with the R0 a run raised (INNER_R0), so that its replay repeats it."""
    if not result.r0_raised:
        return snap
    comps = {
        cid: replace(c, r0=result.r0_raised.get(cid, c.r0)) for cid, c in snap.components.items()
    }
    return replace(snap, components=comps)


def _search_row(label: str, result: PairResult, chosen_snap: Snapshot) -> dict:
    return {
        "variant": label or "as asked",
        "status": result.status,
        "score": result.quality.get("score"),
        "A": result.quality.get("per_galaxy", {}).get("A"),
        "B": result.quality.get("per_galaxy", {}).get("B"),
        "sky": chosen_snap.settings.sky,
        "cycles": len(result.metrics),
        "calls": len(result.calls),
        "A_r1": result.commits["A"].r1_call if "A" in result.commits else None,
        "B_r1": result.commits["B"].r1_call if "B" in result.commits else None,
        "seconds": result.seconds,
    }


def effective_harmonics(c: Component) -> Tuple[int, int]:
    """COS3X and COS4X as ELLIPROF uses them (unset: its default, DEFAULT_HARMONICS)."""
    c3, c4 = c.harmonics()
    return (DEFAULT_HARMONICS[0] if c3 is None else c3, DEFAULT_HARMONICS[1] if c4 is None else c4)


def _harmonic_words(c: Component) -> str:
    c3, c4 = effective_harmonics(c)
    return f"COS3X={c3} COS4X={c4}"


def harmonic_variants(snap: Snapshot, cid: str) -> List[Tuple[str, Snapshot]]:
    """The snapshots TRYCOS3X= and TRYCOS4X= of galaxy ``cid`` ask for (every combination of
    its two lists, the other galaxy as it is), but not the snapshot's own."""
    c = snap.components[cid]
    if not (c.try_cos3x or c.try_cos4x):
        return []
    c3, c4 = effective_harmonics(c)
    own = (c3, c4)
    out = []
    for a in c.try_cos3x or [c3]:
        for b in c.try_cos4x or [c4]:
            if (a, b) == own:
                continue
            comps = dict(snap.components)
            comps[cid] = replace(c, mode="", cos3x=a, cos4x=b)
            out.append((f"{cid} COS3X={a} COS4X={b}", replace(snap, components=comps)))
    return out[:HARMONIC_COMBINATIONS]


# A harmonic variant replaces the fit kept only when it lowers the residual score by more
# than this (in units of the pixel noise), as free centres must (FREE_GAIN): more terms
# always fit the noise a little better.
HARMONIC_GAIN = 0.01


def choose_harmonics(kept: PairResult, tried: List[Tuple[PairResult, Snapshot]]):
    """The index of the harmonic variant to keep instead of ``kept`` (None: keep it), and
    why."""
    score = kept.quality.get("score", float("inf")) if kept.quality else float("inf")
    good = [
        (r.quality.get("score", float("inf")), i)
        for i, (r, _s) in enumerate(tried)
        if r.complete
        and r.quality
        and (r.status == "stable_valid_pair" or kept.status != "stable_valid_pair")
    ]
    if not good:
        return None, "no other harmonics gave as complete and stable a fit: its own kept"
    best, i = min(good)
    if best < score - HARMONIC_GAIN:
        return i, f"lowered the residual by {score - best:.3f}"
    return None, (
        f"its harmonics kept: the best others lowered the residual by {max(score - best, 0):.3f}, "
        f"not more than {HARMONIC_GAIN:g}"
    )


# The mean of a search: the fits whose residual score is within this share of the best
ENSEMBLE_SHARE = 0.05
# The mean of RUNS: every complete repeat but those whose score is this share above the best
# (a repeat that went wrong, not the variance the repeats are for)
REPEAT_SHARE = 0.5


def all_harmonics(snap: Snapshot) -> Snapshot:
    """HARMONICS=AUTO: every COS3X and COS4X to try for each galaxy (unless it lists its
    own TRYCOS3X / TRYCOS4X)."""
    comps = {
        cid: replace(
            c,
            try_cos3x=c.try_cos3x or list(COS3X_VALUES),
            try_cos4x=c.try_cos4x or list(COS4X_VALUES),
        )
        for cid, c in snap.components.items()
    }
    return replace(snap, components=comps)


def jittered(snap: Snapshot, s: Settings) -> List[Tuple[str, Snapshot]]:
    """RUNS: RUNS - 1 copies of the fit kept, each galaxy's R0 and R1 (and its first R1
    and listed radii with R1) changed by 2/3 JITTER to JITTER at random, up or down, so that
    the fit may settle on another model; their mean averages that variance out."""
    rng = np.random.default_rng(s.seed if s.seed is not None else snap.run_id)
    out = []
    for k in range(2, s.runs + 1):
        comps, words = {}, []
        for cid, c in snap.components.items():
            f0, f1 = (
                1 + rng.choice((-1.0, 1.0)) * rng.uniform(2 * s.jitter / 3, s.jitter)
                for _ in range(2)
            )
            r0, r1 = c.r0 * f0, c.r1 * f1
            if r0 >= 0.5 * r1:
                r0 = c.r0
            changes = dict(r0=round(r0, 2), r1=round(r1, 2), try_cos3x=[], try_cos4x=[])
            if c.radii:
                radii = [round(x * f1, 2) for x in c.radii if r0 < x * f1 <= r1]
                changes["radii"] = radii or None
            elif c.start:
                start = round(c.start * f1, 2)
                changes["start"] = start if r0 < start <= r1 else None
            comps[cid] = replace(c, **changes)
            words.append(f"{cid} R0 {comps[cid].r0:g} R1 {comps[cid].r1:g}")
        out.append((f"repeat {k}: " + ", ".join(words), replace(snap, components=comps)))
    return out


def ensemble(
    fits: List[Tuple[PairResult, Snapshot]],
    rows: List[dict],
    share: float = ENSEMBLE_SHARE,
    repeats: bool = False,
) -> dict:
    """The mean sky, A + B model and residual of the complete fits of a search whose score
    is within ENSEMBLE_SHARE of the best, and the spread of their models (the standard
    deviation, pixel by pixel): how much the answer depends on the choices searched.
    Empty with fewer than two such fits.  Marks the members' rows with "mean"."""
    scores = [
        (r.quality.get("score"), i)
        for i, (r, _s) in enumerate(fits)
        if r.complete and r.residual is not None and r.quality and r.quality.get("score")
    ]
    if len(scores) < 2:
        return {}
    best = min(sc for sc, _i in scores)
    members = [i for sc, i in scores if sc <= best * (1 + share)]
    if len(members) < 2:
        return {}
    totals = np.stack([fits[i][0].total for i in members])
    residuals = np.stack([fits[i][0].residual for i in members])
    skies_ = np.array([fits[i][1].settings.sky for i in members])
    for i in members:
        rows[i]["mean"] = True
    return {
        "members": len(members),
        "of": len(fits),
        "share": share,
        "repeats": repeats,
        "A": np.mean([fits[i][0].commits["A"].model for i in members], axis=0),
        "B": np.mean([fits[i][0].commits["B"].model for i in members], axis=0),
        "sky": float(skies_.mean()),
        "sky_spread": float(skies_.std(ddof=1)),
        "model": totals.mean(axis=0),
        "residual": residuals.mean(axis=0),
        "spread": totals.std(axis=0, ddof=1),
    }


def _with_plane(result, base, support, progress, cancelled, fitter) -> PairResult:
    """SKYPLANE: the search again, on the image minus the tilt of a plane fitted to the
    residual of the fit kept (when significant); the new result is kept when its residual
    score is lower.  The whole search, not the fit kept alone: free centres drift outwards
    to take up a gradient, and on the flattened image fixed ones may do better."""
    if not base.settings.sky_plane or not result.complete:
        return result
    probe = replace(base, image=base.image)  # the residual is the kept fit's
    flat = remove_tilt(probe, result)
    if flat.tilt is None:
        result.notes_extra = [flat.notes[-1]]
        return result

    def say(info):
        if progress is not None:
            info["variant"] = "again on the image minus the sky's tilt: " + str(
                info.get("variant", "")
            )
            progress(info)

    flat = replace(flat, settings=replace(flat.settings, sky_plane=False), auto={})
    again = _search(flat, support, say, cancelled, fitter)
    if not again.complete:
        result.notes_extra = [flat.notes[-1] + "; the search on it ended incomplete: not used"]
        return result
    old = result.quality.get("score") if result.quality else None
    new = again.quality.get("score") if again.quality else None
    if old is not None and new is not None and new > old:
        result.notes_extra = [
            flat.notes[-1] + f"; not kept: the residual score rose from {old:.3f} to {new:.3f}"
        ]
        return result
    again.notes_extra = [
        flat.notes[-1]
        + (f"; kept: the residual score fell from {old:.3f} to {new:.3f}" if old else "")
    ]
    return again


# Free centres are kept only when they lower the residual score by more than this (in units
# of the pixel noise).  On 24 simulated pairs (12 scenes, 2 noise seeds) searched over damping
# 1 and 0.5, fixed and free centres and growth x1.5 and x1.25, free centres often matched
# fixed ones in residual while moving light between A and B.  Against keeping the smallest
# score, this rule lowered the median residual (0.141 to 0.137 of the noise) and the profile
# error of A + B (median 1.66 % to 1.20 %, 90th percentile 6.8 % to 4.0 %).  Lopsided
# galaxies, whose centre of light moves with radius, are where free centres clearly help.
FREE_GAIN = 0.01


def _fixed(snap: Snapshot) -> bool:
    return all(c.fixctr != 0 for c in snap.components.values())


def choose(results: List[Tuple[PairResult, Snapshot]]) -> Tuple[int, str]:
    """The index of the fit run_search keeps, and why.  Complete fits before incomplete ones
    and stable ones before the rest, then the smallest residual score (quality()); among fits
    with fixed centres and fits with free ones, fixed centres win unless free centres lower
    the score by more than FREE_GAIN."""

    def rank(i):
        r, _s = results[i]
        stable = r.status == "stable_valid_pair"
        return (not r.complete, not stable, r.quality.get("score", float("inf")))

    every = range(len(results))
    fixed = [i for i in every if _fixed(results[i][1])]
    free = [i for i in every if not _fixed(results[i][1])]
    if not fixed or not free:
        return min(every, key=rank), "the smallest residual"
    f, g = min(fixed, key=rank), min(free, key=rank)
    if not results[f][0].complete:
        return g, "fixed centres gave no complete fit"
    if not results[g][0].complete:
        return f, "the smallest residual with fixed centres"
    sf, sg = rank(f)[2], rank(g)[2]
    if rank(g)[:2] < rank(f)[:2]:
        return g, "fixed centres gave no stable fit"
    if rank(g)[:2] == rank(f)[:2] and sg < sf - FREE_GAIN:
        return g, f"free centres lowered the residual by {sf - sg:.3f} (more than {FREE_GAIN:g})"
    return f, (
        f"fixed centres kept: free centres lowered the residual by {max(sf - sg, 0):.3f}, "
        f"not more than {FREE_GAIN:g}"
    )


def command_for(snap: Snapshot, mask: str = "") -> str:
    """A pair command (for buffer 1) that repeats a snapshot's fit exactly: every setting
    explicit, the sky factor and the variant kept, no search or sky iterations."""
    s = snap.settings
    base = s.sky_base if s.sky_base is not None else s.sky
    parts = [f"pair 1 SKY={base:.10g}"]
    if s.sky_factor != 1.0:
        parts.append(f"SKYFACTOR={s.sky_factor:.10g}")
    if mask:
        parts.append(mask)
    parts += [
        f"CORES={1 if s.cores else 0}",
        f"CYCLES={s.cycles}",
        f"NITER={s.niter}",
        f"RLAW={s.rlaw}",
        f"GAIN={s.gain:g}",
        f"TIE={s.tie}",
        f"CORECAP={s.corecap:g}",
        f"TIMEOUT={s.timeout:g}",
        f"BOOTSTRAP={1 if s.bootstrap else 0}",
        f"ONFAIL={s.on_fail}",
        f"CONVERGE={s.converge:g}",
        f"CONSECUTIVE={s.consecutive}",
    ]
    if s.steps:
        parts.append(f"STEPS={s.steps}")
    if s.damp < 1:
        parts += [f"DAMP={s.damp:g}", f"POLISH={s.polish}"]
    if s.rmstar:
        parts.append("RMSTAR")
    if s.tail != "elliprof":
        parts.append(f"TAIL={s.tail}")
    if s.sky_plane:
        parts.append("SKYPLANE=1")
    for cid, c in snap.components.items():
        words = [f"{cid}:"]
        if c.name:
            words.append("NAME=" + shlex.quote(c.name))
        words += [f"X0={c.x0:.10g}", f"Y0={c.y0:.10g}", f"R0={c.r0:g}", f"R1={c.r1:g}"]
        words += [f"NR={c.nr}", f"FIXCTR={c.fixctr}"]
        if c.mode:
            words.append(f"MODE={c.mode}")
        else:
            if c.cos3x is not None:
                words.append(f"COS3X={c.cos3x}")
            if c.cos4x is not None:
                words.append(f"COS4X={c.cos4x}")
        if c.core is not None:
            words.append(f"CORE={c.core:g}")
        if c.block:
            words.append(f"BLOCK={c.block:g}")
        if c.adaptive:
            words.append("ADAPTIVE")
        if c.radii:
            words.append("RADII=" + ",".join(f"{r:g}" for r in c.radii))
        elif c.start:
            words.append(f"START={c.start:g}")
            if c.factor:
                words.append(f"FACTOR={c.factor:g}")
            elif c.step:
                words.append(f"STEP={c.step:g}")
        parts.append(" ".join(words))
    return " ".join(parts)


def chosen_factor(result: PairResult, settings: Settings) -> float:
    """The sky factor of the fit a result shows."""
    for entry in result.sky_iterations:
        if entry.get("chosen"):
            return float(entry["factor"])
    return settings.sky_factor


def _metrics(snap, committed, prev_a, prev_b, prev_t, threshold: float = STABILITY) -> dict:
    ev = snap.eval
    a, b = committed["A"].model, committed["B"].model
    t = a + b
    good = ev & np.isfinite(t) & np.isfinite(prev_t)
    n = int(good.sum())
    sigma = snap.sky_sigma
    if not sigma or not np.isfinite(sigma):
        r = (snap.image - snap.settings.sky - t)[good]
        sigma = float(1.4826 * np.median(np.abs(r - np.median(r)))) if r.size else float("nan")
    floor = 3 * sigma / math.sqrt(n) if n and np.isfinite(sigma) else float("inf")
    out = {"eval_pixels": n}
    reasons = []
    stable = True
    for name, new, old in (("A", a, prev_a), ("B", b, prev_b), ("T", t, prev_t)):
        key = f"d{name}"
        level = _rms(new[good])
        if n < 1000 or not np.isfinite(level) or level < floor:
            out[key] = None
            reasons.append(f"d{name} not_evaluable")
            stable = False
            continue
        value = _rms((new - old)[good]) / level
        out[key] = value
        if value >= threshold:
            stable = False
            reasons.append(f"d{name}={value:.3f}")
    if stable is False and out.get("dT") is not None and out["dT"] < threshold:
        if any(out.get(k) is not None and out[k] >= threshold for k in ("dA", "dB")):
            reasons.append("components changing; total stable")
    out["stable"] = stable
    out["reason"] = ", ".join(reasons)
    return out


def _finish(result: PairResult, snap: Snapshot, start, checkpoint, committed) -> PairResult:
    if checkpoint is not None:
        result.checkpoint_cycle, result.commits = checkpoint[0], dict(checkpoint[1])
        newer = {c: committed[c] for c in committed if committed[c] is not result.commits.get(c)}
        if newer:
            result.partial = newer
    if result.complete:
        a, b = result.commits["A"].model, result.commits["B"].model
        result.total = a + b
        result.residual = snap.image - snap.settings.sky - result.total
    result.seconds = time.perf_counter() - start
    return result


def _clean_arguments(arguments: List[str]) -> List[str]:
    """The arguments of a call with its scratch folder written as <tmp>."""
    out = []
    for a in arguments:
        out.append(re.sub(r"\S*astro-vista-pair-[^/\s]+", "<tmp>", a))
    return out


# ---------------------------------------------------------------- telling the result
def summary(result: PairResult, snap: Snapshot) -> str:
    lines = []
    for cid, block in snap.blocks.items():
        lines.append(
            f"block: {block.reason} (around {cid}, left out of "
            f"{'B' if cid == 'A' else 'A'}'s first fit)"
        )
    for g in result.growth:
        lines.append(
            f"growth step {g['step']}: A R1 {g['A_r1']:.1f} {'ok' if g['A_ok'] else 'failed'}, "
            f"B R1 {g['B_r1']:.1f} {'ok' if g['B_ok'] else 'failed'}"
        )
    lines += [f"note: {n}" for n in result.notes_growth]
    for m in result.metrics:
        parts = []
        for k in ("dA", "dB", "dT"):
            v = m.get(k)
            parts.append(f"{k[1]} {'n/a' if v is None else f'{100 * v:.2f} %'}")
        oks = f"A {'ok' if m['A_ok'] else 'failed'}, B {'ok' if m['B_ok'] else 'failed'}"
        lines.append(f"cycle {m['cycle']}: {oks}; change {', '.join(parts)}")
    head = f"status: {result.status}"
    if result.stability_reason:
        head += f" ({result.stability_reason})"
    if result.checkpoint_cycle is not None:
        head += f"; result = checkpoint of cycle {result.checkpoint_cycle}"
    lines.append(head)
    for cid in ("A", "B"):
        comp = snap.components[cid]
        c = result.commits.get(cid)
        who = f"{cid} ({comp.name})" if comp.name and comp.name != cid else cid
        if c is None:
            lines.append(f"{who}: no valid model")
            continue
        text = (
            f"{who}: model of {('initialisation' if c.tag == 'init' else 'cycle ' + c.tag)}"
            f"; R1 requested {comp.r1:g}"
        )
        attempted = result.attempted_r1.get(cid)
        if attempted is not None and abs(attempted - c.r1_call) > 1e-9:
            text += f", last attempted {attempted:.1f} (rejected)"
        text += f", effective {c.r1_call:.1f}, last isophote {c.rmaj_last:.1f}"
        worst = min(x["usable"] for x in c.support)
        text += f"; outer support {100 * worst:.0f} %"
        if c.coverage_uncertain:
            text += " (coverage: uncertain)"
        lines.append(text)
    s = snap.settings
    sky = f"sky {s.sky:.6g} (frozen"
    if s.sky_base is not None and s.sky_factor != 1.0:
        sky += f"; factor {s.sky_factor:.5f} x base {s.sky_base:.6g}"
    sky += f", measurement {snap.sky_id})" if snap.sky_id else ", typed value)"
    if result.sky_check:
        sc = result.sky_check
        if "median" in sc:
            sky += (
                f"; on pixels accepted by the frozen sky mask the residual median is "
                f"{sc['median']:.3g}"
            )
    lines.append(sky)
    if result.sky_iterations:
        total = sum(e["seconds"] for e in result.sky_iterations)
        lines.append(
            f"sky factor: {len(result.sky_iterations)} pair fits in {total:.1f} s (residual "
            "image - sky - A - B on the sky support, 3-sigma clipped; the models extend beyond "
            "R1, so their outer light counts):"
        )
        for e in result.sky_iterations:
            text = f"  {e['iteration']}: factor {e['factor']:.5f}, sky {e['sky']:.6g}"
            if e.get("median") is not None:
                text += f", residual median {e['median']:+.4g} (sigma {e['sigma']:.4g})"
            text += f", {e['status']}"
            if e.get("chosen"):
                text += "   <- the result"
            if e.get("note"):
                text += f"; {e['note']}"
            lines.append(text)
    for cid, r0 in result.r0_raised.items():
        lines.append(
            f"note: {cid}'s innermost isophotes failed; it was fitted from R0 {r0:g} instead "
            f"(isophotes within {INNER_R0:g} px are set by the pixels and the PSF)"
        )
    for cid, r1 in result.bootstraps.items():
        lines.append(
            f"note: {cid} was started from a partial bootstrap fit (R1 {r1:.1f}); it only "
            "started the cycles and is not part of any result"
        )
    if result.search:
        harmonic = [r for r in result.search if r.get("harmonics")]
        lines.append(
            f"search: {len(result.search) - len(harmonic)} variants"
            + (f" and {len(harmonic)} harmonic variants" if harmonic else "")
            + f", {snap.settings.parallel} at a time; kept: "
            + next((r["variant"] for r in result.search if r.get("chosen")), "?")
            + (f" ({result.search_choice})" if result.search_choice else "")
        )
    e, s_ = result.ensemble, snap.settings
    for row in result.repeats[1:]:
        score = "-" if row["score"] is None else format(row["score"], ".3f")
        lines.append(
            f"  {row['variant']}: {row['status']}, score {score}"
            + ("  (in the mean)" if row.get("mean") else "")
        )
    if e:
        lines.append(
            (
                f"mean of {e['members']} runs of {e['of']} (the fit kept and its repeats with "
                f"R0 and R1 changed by {100 * 2 * s_.jitter / 3:.0f}-{100 * s_.jitter:.0f} %; "
                f"those more than {100 * e['share']:g} % above the best score left out)"
                if e.get("repeats")
                else f"mean of the {e['members']} fits within {100 * e['share']:g} % of the "
                f"best score (of {e['of']})"
            )
            + f": sky {e['sky']:.6g} ± {e['sky_spread']:.2g}; the A + B models "
            f"differ by {float(np.median(e['spread'])):.3g} (median per pixel)"
        )
    for note in snap.notes + [n for n in result.notes_extra if n not in snap.notes]:
        lines.append(f"note: {note}")
    calls = len(result.calls)
    lines.append(
        f"{calls} elliprof calls in {result.seconds:.1f} s"
        + (f"; the whole search {result.wall:.1f} s" if result.wall > result.seconds + 0.5 else "")
    )
    lines.append(
        "Passing these checks and a small change do not prove a unique separation of the "
        "two galaxies, nor that the residual is ready for SBF."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------- export
def export(
    result: PairResult,
    snap: Snapshot,
    folder: Path,
    requested: str,
    sky_report: Optional[dict] = None,
    sky_exclusion: Optional[np.ndarray] = None,
    replay: Optional[List[str]] = None,
) -> Path:
    """Write a pair fit to ``folder`` (see the plan, 6.6).  ``transaction.json`` says
    "writing" until everything else, the manifest last, is written."""
    from astropy.io import fits

    from . import __version__

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    def atomic(name: str, write):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        write(tmp)
        tmp.replace(path)
        if name not in ("transaction.json", "manifest.json"):
            written[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def text(name, content):
        atomic(name, lambda p: p.write_text(content))

    def image(name, data, comment):
        def w(p):
            h = fits.Header() if snap.header is None else snap.header.copy()
            for key in ("BSCALE", "BZERO", "BLANK"):
                h.remove(key, ignore_missing=True)
            h["COMMENT"] = comment
            fits.PrimaryHDU(np.asarray(data, np.float64), header=h).writeto(
                p, output_verify="silentfix", overwrite=True
            )

        atomic(name, w)

    def mask(name, masked, comment):
        def w(p):
            h = fits.Header()
            h["MASKCONV"] = ("1=good", "0 = left out, 1 = used (elliprof convention)")
            h["COMMENT"] = comment
            fits.PrimaryHDU((~masked).astype(np.uint8), header=h).writeto(p, overwrite=True)

        atomic(name, w)

    text("transaction.json", json.dumps({"export": "writing"}) + "\n")
    s = snap.settings
    result_doc = {
        "status": result.status,
        "stability_reason": result.stability_reason,
        "checkpoint_cycle": result.checkpoint_cycle,
        "partial_commits": sorted(result.partial),
        "metrics": result.metrics,
        "failures": result.failures,
        "seconds": result.seconds,
        "components": {
            cid: {
                "name": snap.components[cid].label,
                "model_cycle": c.tag,
                "requested_r1": snap.components[cid].r1,
                "last_attempted_r1": result.attempted_r1.get(cid),
                "effective_r1": c.r1_call,
                "last_isophote_rmaj": c.rmaj_last,
                "outer_support": c.support,
                "coverage_uncertain": c.coverage_uncertain,
            }
            for cid, c in result.commits.items()
        },
        "sky_check_on_frozen_support": result.sky_check,
        "sky_factor_iterations": result.sky_iterations,
        "growth_steps": result.growth,
        "radius_schedules": result.schedules,
        "growth_notes": result.notes_growth,
        "blocks": {cid: b.describe() for cid, b in snap.blocks.items()},
    }
    text("result.json", json.dumps(result_doc, indent=2, default=float) + "\n")

    def config(effective: bool) -> str:
        comps = {}
        for cid, c in snap.components.items():
            d = {
                "id": cid,
                "name": c.label,
                "center": {"x": c.x0, "y": c.y0},
                "r0": c.r0,
                "r1": c.r1,
                "nr": c.nr,
                "harmonic_mode": c.mode or None,
                "cos3x": c.harmonics()[0],
                "cos4x": c.harmonics()[1],
                "center_fit": {0: "free", 1: "fixed", 2: "median"}[c.fixctr],
                "outer_policy": "adaptive" if c.adaptive else "fixed",
                "core_semi_major_override": c.core,
                "block_semi_major": c.block,
                "start_r1": c.start,
            }
            if effective and cid in result.commits:
                d["effective_r1"] = result.commits[cid].r1_call
                d["core_used"] = snap.cores[cid].describe()
                d["source_match"] = snap.matches[cid].__dict__
            comps[cid] = d
        doc = {
            "schema_version": "pair-gui-v1 (subset written by AstroVista Lite; no importer exists)",
            "coordinates": "native_first_pixel_center_0.5",
            "input": snap.provenance,
            "sky": {
                "value": s.sky,
                "base": s.sky_base,
                "factor": s.sky_factor,
                "measurement": snap.sky_id or "typed",
                "fixed_across_cycles": True,
            },
            "native": {
                "niter": s.niter,
                "gain": s.gain,
                "tie": s.tie,
                "rlaw": s.rlaw,
                "rmstar": s.rmstar,
                "model_beyond_r1": s.tail,
                "timeout_seconds": s.timeout,
            },
            "pair": {
                "growth_steps": s.steps,
                "cycles": s.cycles,
                "early_stopping": s.early_stop,
                "stability_relative": STABILITY,
                "corecap": s.corecap,
                "update_order": ["A", "B"],
            },
            "targets": comps,
            "ensemble": "off",
            "stage2": "off",
            "outer_sky_estimation": "off",
            "coverage_backoff": "off",
        }
        return _yaml(doc)

    text("requested.yaml", config(False))
    text("effective.yaml", config(True))
    image("inputs/science.fits", snap.image, "the snapshot fitted (float64 copy)")
    mask(
        "inputs/fit_exclusion.fits", snap.fit_mask, f"M0: {snap.fit_mask_label or 'invalid pixels'}"
    )
    mask(
        "inputs/core_A.fits", snap.cores["A"].masked, "core centred on A, left out while fitting B"
    )
    mask(
        "inputs/core_B.fits", snap.cores["B"].masked, "core centred on B, left out while fitting A"
    )
    mask("inputs/eval.fits", ~snap.eval, "EVAL: 1 = an evaluation pixel")
    if sky_exclusion is not None:
        mask("inputs/sky_exclusion.fits", sky_exclusion, "pixels left out of the sky")
    lines = [
        "# replay from the snapshot inputs (requested values)",
        "rd 1 inputs/science.fits",
        "mask load m0 inputs/fit_exclusion.fits 1",
        "mask role m0 fit",
    ]
    if snap.provenance.get("source_run"):  # the cores came from the source finder's run
        lines.append(snap.provenance["source_run"]["command"].strip())
    lines.append(command_for(snap, "MASK=m0"))  # the fit kept, every setting explicit
    if replay:
        lines += ["", "# the route from the original files"] + replay
    text("commands.txt", "\n".join(lines) + "\n")
    text("elliprof_calls.json", json.dumps([c.__dict__ for c in result.calls], indent=2) + "\n")
    # every iteration: what it was fitted on, with what, and what it made
    iterations = []
    for h, c in zip(result.history, result.calls):
        comp = snap.components[h["galaxy"]]
        e = {
            "call": h["call"],
            "cycle": h["cycle"],
            "galaxy": h["galaxy"],
            "original_image": snap.provenance.get("file") or snap.image_name,
            "original_sha256": snap.provenance.get("snapshot_sha256"),
            "input": h["input"],
            "companion_model_subtracted": h["companion"],
            "centre": [comp.x0, comp.y0],
            "fixctr": comp.fixctr,
            "r0": comp.r0,
            "r1": h["r1"],
            "nr": comp.nr,
            "harmonics": list(comp.harmonics()),
            "sky": s.sky,
            "mask": snap.fit_mask_label,
            "mask_pixels": c.mask_pixels,
            "elliprof_arguments": c.arguments,
            "passed": h["passed"],
            "reason": h["reason"],
            "centre_drift": c.centre_drift,
            "seconds": c.seconds,
        }
        if h["model"] is not None:
            e["model_sha256"] = hashlib.sha256(
                np.ascontiguousarray(h["model"]).tobytes()
            ).hexdigest()
            name = f"iterations/{h['call']:03d}_{h['galaxy']}_profile.csv"
            atomic(name, lambda p, prof=h["profile"]: prof.to_csv(p, index=False))
            e["profile"] = name
            both = h["model"] + (0 if h["companion_model"] is None else h["companion_model"])
            r = (snap.image - s.sky - both)[snap.eval & np.isfinite(snap.image)]
            if r.size:
                e["residual_on_eval"] = {
                    "median": float(np.median(r)),
                    "mad_sigma": float(1.4826 * np.median(np.abs(r - np.median(r)))),
                    "pixels": int(r.size),
                }
        iterations.append(e)
    text(
        "iterations.json",
        json.dumps(
            {"radius_schedules": result.schedules, "iterations": iterations},
            indent=2,
            default=float,
        )
        + "\n",
    )
    for cid, c in result.commits.items():
        atomic(f"{cid}_profile.csv", lambda p, c=c: c.profile.to_csv(p, index=False))

        def reg(p, c=c, cid=cid):
            colour = "magenta" if cid == "A" else "orange"
            rows = [
                "# Region file format: DS9 version 4.1",
                f"global color={colour} width=1",
                "image",
            ]
            for x, y, a, b, t in c.ellipses:
                rows.append(f"ellipse({x + 0.5:.4f},{y + 0.5:.4f},{a:.4f},{b:.4f},{t:.4f})")
            p.write_text("\n".join(rows) + "\n")

        atomic(f"{cid}_isophotes.reg", reg)
        image(f"{cid}_model.fits", c.model, f"model of {cid} ({snap.components[cid].label})")
    if result.complete:
        image("total_model.fits", result.total, "A + B")
        image("residual_raw.fits", result.residual, "image - sky - A - B, every pixel")
    if sky_report is not None:
        text("sky.json", json.dumps(sky_report, indent=2, default=float) + "\n")
    rows = ["cycle,A_ok,B_ok,A_model_cycle,B_model_cycle,dA,dB,dT,stable,reason"]
    for m in result.metrics:
        rows.append(
            ",".join(
                str(m.get(k, "")) if m.get(k) is not None else ""
                for k in ("cycle", "A_ok", "B_ok", "A_tag", "B_tag", "dA", "dB", "dT", "stable")
            )
            + f",\"{m.get('reason', '')}\""
        )
    text("cycles.csv", "\n".join(rows) + "\n")
    import elliprof

    manifest = {
        "astrovista_lite": __version__,
        "elliprof": elliprof.__version__,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": result.status,
        "input": snap.provenance,
        "sky": s.sky,
        "files_sha256": written,
    }
    atomic("manifest.json", lambda p: p.write_text(json.dumps(manifest, indent=2) + "\n"))
    text("transaction.json", json.dumps({"export": "complete"}) + "\n")
    return folder


# ---------------------------------------------------------------- the report
def residual_rings(
    result: PairResult, snap: Snapshot, bins: int = 10, requested: bool = False
) -> Dict[str, List[dict]]:
    """For each galaxy: the residual in elliptical rings (the shape of its outer half),
    from R0 to 1.5 x its effective R1, as medians of the model and of the residual, on the
    side facing the other galaxy and on the side away from it.  Fitting exclusions and the
    other galaxy's core are left out."""
    out: Dict[str, List[dict]] = {}
    if result.residual is None:
        return out
    ny, nx = snap.image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    xs, ys = xx + 0.5, yy + 0.5
    for cid, commit in result.commits.items():
        other = "B" if cid == "A" else "A"
        p = commit.profile
        half = p[p["Rmaj"] >= p["Rmaj"].median()]
        e = float(np.clip(half["ellip"].median(), 0, 0.9))
        angle = math.radians(float(half["alpha"].median()) - 90.0)
        comp, far = snap.components[cid], snap.components[other]
        dx, dy = xs - comp.x0, ys - comp.y0
        u = dx * math.cos(angle) + dy * math.sin(angle)
        v = -dx * math.sin(angle) + dy * math.cos(angle)
        r = np.hypot(u, v / (1 - e))
        toward = dx * (far.x0 - comp.x0) + dy * (far.y0 - comp.y0) > 0
        good = np.isfinite(result.residual) & ~snap.fit_mask & ~snap.cores[other].masked
        outer = comp.r1 if requested else commit.r1_call  # requested: the same rings for all
        edges = np.geomspace(max(comp.r0, 2.0), 1.5 * outer, bins + 1)
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            ring = good & (r >= lo) & (r < hi)
            if ring.sum() < 20:
                continue
            model = commit.model[ring]
            res = result.residual[ring]
            row = {
                "r_in": float(lo),
                "r_out": float(hi),
                "pixels": int(ring.sum()),
                "model": float(np.median(model)),
                "residual": float(np.median(res)),
                "beyond_r1": bool(lo >= commit.r1_call),
            }
            for side, sel in (("facing", toward[ring]), ("away", ~toward[ring])):
                row[side] = float(np.median(res[sel])) if sel.sum() >= 10 else None
            rows.append(row)
        out[cid] = rows
    return out


def report(result: PairResult, snap: Snapshot, line: str = "", measured=None) -> str:
    """A plain-text report of a pair fit, to copy and paste: what was asked, what every
    call did, where each galaxy grew to, the sky, the residual around each galaxy, and
    hints about what may have gone wrong.  The hints are heuristics, not verdicts."""
    from . import __version__

    s = snap.settings
    L: List[str] = []
    add = L.append
    add(f"# AstroVista Lite {__version__} pair report, {time.strftime('%Y-%m-%d %H:%M')}")
    prov = snap.provenance or {}
    add(f"image: {snap.image_name}  {snap.image.shape[1]} x {snap.image.shape[0]}")
    if prov.get("file"):
        add(f"file: {prov['file']}")
    if line:
        add(f"command: {line}")
    add("")
    add(f"## status: {result.status}")
    if result.stability_reason:
        add(f"reason: {result.stability_reason}")
    add(
        f"result = checkpoint of cycle {result.checkpoint_cycle}; {len(result.calls)} calls, "
        f"{result.seconds:.1f} s"
    )
    q = result.quality or quality(result, snap)
    if q:
        add(
            f"residual outside the cores: RMS {q['score']:.3f} x the pixel noise "
            f"({q['sigma']:.4g}); this judges A + B together, not how the light is split"
        )
    for row in result.search:
        score = "-" if row["score"] is None else format(row["score"], ".3f")
        add(
            f"variant {row['variant']}: {row['status']}, score {score}"
            + ("  (in the mean)" if row.get("mean") else "")
            + ("  <- kept" if row.get("chosen") else "")
        )
    if result.search_choice:
        add(f"kept because: {result.search_choice}")
    add("")
    add("## settings")
    add(
        f"damping {s.damp:g} (undamped final cycles {s.polish if s.damp < 1 else 0}), on "
        f"failure {s.on_fail}, cores {'on' if s.cores else 'off'}, convergence "
        f"{100 * s.converge:g} % for {s.consecutive} cycles"
    )
    add(
        f"sky used {s.sky:.6g} = factor {s.sky_factor:.5f} x base "
        f"{(s.sky_base if s.sky_base is not None else s.sky):.6g}; measurement "
        f"{snap.sky_id or 'none (typed)'}"
    )
    add(
        f"cycles {s.cycles}, growth steps {s.steps}, niter {s.niter}, rlaw {s.rlaw}, "
        f"corecap {s.corecap:g}, bootstrap {'on' if s.bootstrap else 'off'}"
    )
    add(f"fitting mask: {snap.fit_mask_label or '(none)'}, {int(snap.fit_mask.sum())} px")
    for cid, c in snap.components.items():
        add(
            f"{cid} {c.label}: X0 {c.x0:g} Y0 {c.y0:g} R0 {c.r0:g} R1 {c.r1:g} NR {c.nr} "
            f"start {c.start or c.r1:g} block {c.block or 0:g} mode {c.mode or 'default'} "
            f"fixctr {c.fixctr} adaptive {'yes' if c.adaptive else 'no'}"
        )
    for owner, core in snap.cores.items():
        add(f"core of {owner}: {core.shape} a {core.a:.1f} b {core.b:.1f}; {core.reason}")
    for owner, block in snap.blocks.items():
        add(f"block of {owner}: a {block.a:.1f} b {block.b:.1f}; {block.reason}")
    for cid, m in snap.matches.items():
        add(
            f"source match {cid}: {m.reason}"
            + (f" (Kron a {m.a:.1f}, q {m.q:.2f})" if m.ok else "")
        )
    for n in snap.notes + [n for n in result.notes_extra if n not in snap.notes]:
        add(f"note: {n}")
    if measured is not None:
        add(
            f"sky measurement {measured.id}: {measured.state}, median {measured.level:.6g}, "
            f"sigma {measured.sigma:.4g}, support {100 * measured.support_fraction:.1f} % "
            f"(threshold {measured.settings['THRESH']:g} sigma, grow "
            f"{measured.settings['GROW']:g}, mesh {measured.settings.get('BACK_SIZE', 0)})"
        )
    add("")
    if result.growth:
        add("## growth")
        for g in result.growth:
            add(
                f"step {g['step']}: A R1 {g['A_r1']:.1f} {'ok' if g['A_ok'] else 'FAILED'}; "
                f"B R1 {g['B_r1']:.1f} {'ok' if g['B_ok'] else 'FAILED'}"
            )
        for n in result.notes_growth:
            add(f"- {n}")
        add("")
    add("## cycles (relative change of A, B, A+B)")
    for m in result.metrics:
        ch = ", ".join(
            f"{k[1]} {'n/a' if m.get(k) is None else f'{100 * m[k]:.2f}%'}"
            for k in ("dA", "dB", "dT")
        )
        add(
            f"cycle {m['cycle']}: A {'ok' if m['A_ok'] else 'FAILED'}, "
            f"B {'ok' if m['B_ok'] else 'FAILED'}; {ch}"
        )
    add("")
    add("## calls")
    add("n  stage           gal  R1      pass  s     reason")
    for k, c in enumerate(result.calls, 1):
        add(
            f"{k:<3}{c.cycle:<16}{c.component:<5}{c.r1:<8.1f}{'yes' if c.passed else 'NO':<6}"
            f"{c.seconds:<6.2f}{c.reason}"
        )
    add("")
    add("## profiles of the result")
    for cid, c in result.commits.items():
        p = c.profile
        comp = snap.components[cid]
        drift = np.hypot(p["x0"] - comp.x0, p["y0"] - comp.y0)
        pa = (p["alpha"] - 90.0) % 180.0
        add(
            f"{cid}: model of {c.tag}; R1 effective {c.r1_call:.1f} (asked {comp.r1:g}); "
            f"Rmaj {p['Rmaj'].iloc[0]:.1f}..{p['Rmaj'].iloc[-1]:.1f}; I0 "
            f"{p['I0'].iloc[0]:.4g}..{p['I0'].iloc[-1]:.4g}; ellip "
            f"{p['ellip'].min():.3f}..{p['ellip'].max():.3f}; PA {pa.min():.1f}..{pa.max():.1f}; "
            f"centre drift max {drift.max():.2f} px; outer support "
            f"{100 * min(x['usable'] for x in c.support):.0f} %"
            + ("; coverage uncertain" if c.coverage_uncertain else "")
        )
    for cid, why in result.failures.items():
        add(f"last failure of {cid}: {why}")
    for cid, r0 in result.r0_raised.items():
        add(f"{cid}: the innermost isophotes failed; fitted from R0 {r0:g}")
    for cid, r1 in result.bootstraps.items():
        add(f"{cid} started from a bootstrap fit at R1 {r1:.1f}")
    add("")
    sc = result.sky_check
    sigma = None
    if "clipped_median" in sc:
        sigma = sc.get("clipped_sigma")
        add("## sky")
        add(
            f"residual on the sky pixels: median {sc['clipped_median']:+.4g}, sigma "
            f"{sc['clipped_sigma']:.4g}, {sc['clipped_pixels']} px"
            + (
                f"; next factor {sc['clipped_next_factor']:.5f} (sky {sc['clipped_next_sky']:.6g})"
                if "clipped_next_factor" in sc
                else ""
            )
        )
    for e in result.sky_iterations:
        add(
            f"sky fit {e['iteration']}: factor {e['factor']:.5f} sky {e['sky']:.6g} median "
            f"{e.get('median') if e.get('median') is None else format(e['median'], '+.4g')} "
            f"{e['status']}{' KEPT' if e.get('chosen') else ''}"
        )
    rings = residual_rings(result, snap)
    hints: List[str] = []
    if rings:
        add("")
        add(
            "## residual rings (medians; % of the galaxy's own model; facing = side toward the other galaxy)"
        )
    for cid, rows in rings.items():
        add(f"{cid}:  r_in  r_out   pixels   model      resid     %     facing    away")
        for r in rows:
            pct = 100 * r["residual"] / r["model"] if r["model"] > 0 else float("nan")
            f_ = "" if r["facing"] is None else f"{r['facing']:+.4g}"
            a_ = "" if r["away"] is None else f"{r['away']:+.4g}"
            add(
                f"   {r['r_in']:6.1f} {r['r_out']:6.1f} {r['pixels']:8d} {r['model']:9.4g} "
                f"{r['residual']:+9.4g} {pct:+6.1f} {f_:>9} {a_:>9}"
                + ("  (beyond R1)" if r["beyond_r1"] else "")
            )
            scale = max(abs(r["model"]), 3 * (sigma or 0))
            if r["model"] > 3 * (sigma or 0) and abs(r["residual"]) > 0.1 * scale:
                hints.append(
                    f"{cid}: ring {r['r_in']:.0f}-{r['r_out']:.0f} px is {pct:+.0f} % of the model"
                    + (" (beyond R1: the model's extrapolation)" if r["beyond_r1"] else "")
                )
            if r["facing"] is not None and r["away"] is not None and r["model"] > 0:
                if abs(r["facing"] - r["away"]) > 0.1 * scale and r["r_out"] <= 10:
                    hints.append(
                        f"{cid}: within {r['r_out']:.0f} px of the centre one side is "
                        f"{r['facing'] - r['away']:+.4g} against the other (a dipole): the "
                        "centre is probably off; pick it on the source, or free the centre"
                    )
                elif abs(r["facing"] - r["away"]) > 0.1 * scale:
                    hints.append(
                        f"{cid}: at {r['r_in']:.0f}-{r['r_out']:.0f} px the side facing the other "
                        f"galaxy differs from the far side by {r['facing'] - r['away']:+.4g} "
                        "(the other's light not fully removed, or over-removed)"
                    )
    if sigma and abs(sc["clipped_median"]) > 0.5 * sigma:
        hints.append(
            f"sky: the residual on the sky pixels is {sc['clipped_median'] / sigma:+.1f} sigma: "
            + (
                "the sky (or the models' outer light) is too high"
                if sc["clipped_median"] < 0
                else "the sky is too low"
            )
            + "; try the sky factor (Use the next factor, or Refine)"
        )
    for n in result.notes_growth:
        hints.append(n)
    for cid, c in result.commits.items():
        comp = snap.components[cid]
        if c.r1_call < 0.9 * comp.r1:
            hints.append(f"{cid} ended at R1 {c.r1_call:.1f} of {comp.r1:g} asked")
        p = c.profile
        far = snap.components["B" if cid == "A" else "A"]
        d0 = math.hypot(comp.x0 - far.x0, comp.y0 - far.y0)
        d_last = math.hypot(p["x0"].iloc[-1] - far.x0, p["y0"].iloc[-1] - far.y0)
        drift = np.hypot(p["x0"] - comp.x0, p["y0"] - comp.y0)
        k = int(np.argmax(drift.to_numpy()))
        if drift.iloc[k] > max(2.0, 0.05 * float(p["Rmaj"].iloc[k])):
            hints.append(
                f"{cid}: the fitted centre moves {drift.iloc[k]:.1f} px from X0, Y0 (at Rmaj "
                f"{p['Rmaj'].iloc[k]:.1f}): check the centre, or fix it (Centres: fixed)"
            )
        if d0 - d_last > 2.0:
            hints.append(
                f"{cid}: the outer centres drift {d0 - d_last:.1f} px toward the other galaxy "
                "(try fixed centres, a larger block or core, or a smaller R1)"
            )
    if result.status != "stable_valid_pair" and result.failures:
        hints += [f"failure: {k}: {v}" for k, v in result.failures.items()]
    add("")
    add("## hints (heuristics)")
    L += [f"- {h}" for h in hints] or ["- none"]
    return "\n".join(L)


# ---------------------------------------------------------------- a pair's folder
@dataclass
class Folder:
    """What a pair's folder holds: the image (with its HDU), legacy masks, two centres."""

    image: Path
    hdu: Optional[str] = None  # "SCI", "1", ... (None: the primary)
    masks: List[Path] = field(default_factory=list)
    centres: Dict[str, Tuple[float, float, str]] = field(default_factory=dict)  # native
    notes: List[str] = field(default_factory=list)

    @property
    def rd_name(self) -> str:
        return f"{self.image}[{self.hdu}]" if self.hdu else str(self.image)


_POINT = re.compile(
    r"point\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)(?:.*text=\{([^}]*)\})?", re.I
)


def read_regions(path: Path) -> Dict[str, Tuple[float, float, str]]:
    """The centres of A and B from a DS9 region file in image coordinates: points whose
    text starts "A:" and "B:" (else the first two points), converted to ELLIPROF's
    coordinates (x - 0.5, y - 0.5)."""
    points = []
    for line in Path(path).read_text(errors="replace").splitlines():
        m = _POINT.search(line)
        if m:
            points.append((float(m.group(1)) - 0.5, float(m.group(2)) - 0.5, m.group(3) or ""))
    out = {}
    for x, y, text in points:
        m = re.match(r"\s*([AB])\s*:\s*(.*)", text)
        if m and m.group(1) not in out:
            out[m.group(1)] = (x, y, m.group(2).strip())
    if len(out) < 2 and len(points) >= 2:
        out = {cid: (x, y, text.strip()) for cid, (x, y, text) in zip("AB", points)}
    return out


def read_folder(folder) -> Folder:
    """A pair's folder: one FITS image (or the one metadata.json names), its .dmask masks,
    and a DS9 region file with the two centres.  Raises PairError if there is no image."""
    from astropy.io import fits

    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise PairError(f"{folder} is not a folder")
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".fits", ".fit", ".fz"))
    notes = []
    meta = {}
    if (folder / "metadata.json").exists():
        try:
            meta = json.loads((folder / "metadata.json").read_text())
        except (OSError, ValueError):
            notes.append("metadata.json could not be read")
    named = Path(str(meta.get("science_file", ""))).name
    if named and (folder / named).exists():
        image = folder / named
    elif len(images) == 1:
        image = images[0]
    elif not images:
        raise PairError(f"no FITS image in {folder}")
    else:
        raise PairError(
            f"{len(images)} FITS images in {folder}: open one with File > Open, and the "
            "masks and centres by hand"
        )
    hdu = meta.get("hdu")
    if isinstance(hdu, str) and hdu.strip():
        hdu = hdu.strip()
    elif isinstance(hdu, int) and hdu > 0:
        hdu = str(hdu)
    else:
        hdu = None
        with fits.open(image, memmap=True) as f:
            if f[0].data is None or getattr(f[0].data, "ndim", 0) != 2:
                for k, h in enumerate(f[1:], 1):
                    if h.is_image and h.data is not None and h.data.ndim == 2:
                        hdu = h.name if h.name else str(k)
                        break
    masks = sorted(folder.glob("*.dmask"))
    centres = {}
    regions = sorted(folder.glob("*.reg"))
    if regions:
        centres = read_regions(regions[0])
        if len(centres) < 2:
            notes.append(f"{regions[0].name} has fewer than two points: place A and B by hand")
    else:
        notes.append("no region file: place A and B by hand")
    return Folder(image, hdu, masks, centres, notes)


def peak_centre(data: np.ndarray, x: float, y: float, radius: int = 3):
    """The light peak near (x, y), in ELLIPROF's coordinates: the brightest finite pixel
    within ``radius`` pixels, refined by the centroid of its 3 x 3 neighbourhood; None if
    there is no finite pixel there."""
    ny, nx = data.shape
    i, j = int(math.floor(x)), int(math.floor(y))
    x0, x1 = max(i - radius, 1), min(i + radius + 1, nx - 1)
    y0, y1 = max(j - radius, 1), min(j + radius + 1, ny - 1)
    if x0 >= x1 or y0 >= y1:
        return None
    box = np.where(np.isfinite(data[y0:y1, x0:x1]), data[y0:y1, x0:x1], -np.inf)
    if not np.isfinite(box).any():
        return None
    jj, ii = np.unravel_index(int(np.argmax(box)), box.shape)
    pi, pj = x0 + ii, y0 + jj
    sub = np.nan_to_num(data[pj - 1 : pj + 2, pi - 1 : pi + 2].astype(float))
    sub = sub - sub.min()
    if sub.sum() <= 0:
        return pi + 0.5, pj + 0.5
    yy, xx = np.mgrid[-1:2, -1:2]
    return pi + 0.5 + float((sub * xx).sum() / sub.sum()), pj + 0.5 + float(
        (sub * yy).sum() / sub.sum()
    )


def suggested_r1(centres, shape) -> Dict[str, float]:
    """Starting outer radii (to check, not results): at most 0.9 of the distance to the
    nearest edge, and 0.6 (A) or 0.4 (B) of the distance between the centres."""
    (ax, ay, _), (bx, by, _) = centres["A"], centres["B"]
    d = math.hypot(ax - bx, ay - by)
    ny, nx = shape
    out = {}
    for cid, (x, y, _n), share in (("A", centres["A"], 0.6), ("B", centres["B"], 0.4)):
        edge = min(x, y, nx - x, ny - y)
        out[cid] = round(max(10.0, min(0.9 * edge, share * d)), 1)
    return out


def replay_line(line: str, mask: str) -> str:
    """A pair command for buffer 1 with its masks replaced by ``mask`` (a shared keyword,
    so it goes before A:)."""
    parts = re.split(r"(\s[Aa]:\s)", line, maxsplit=1)
    head, sep_, rest = (parts + ["", ""])[:3]
    drop = "MASK|MASKBUF|MASKFILE" + ("|SKYFACTOR|SKYITER" if "SKYFACTOR=" in mask else "")
    head = re.sub(rf"\s({drop})=(\"[^\"]*\"|'[^']*'|\S+)", "", head, flags=re.I)
    head = re.sub(r"^pair\s+\S+", "pair 1", head.strip())
    return f"{head} {mask}{sep_}{rest}".strip()


def _yaml(doc, indent: int = 0) -> str:
    """A small YAML writer (no dependency): dicts, lists and scalars."""
    pad = "  " * indent
    out = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{k}:")
                out.append(_yaml(v, indent + 1))
            else:
                out.append(f"{pad}{k}: {_scalar(v)}")
    elif isinstance(doc, list):
        for v in doc:
            if isinstance(v, (dict, list)) and v:
                inner = _yaml(v, indent + 1).lstrip()
                out.append(f"{pad}- {inner}")
            else:
                out.append(f"{pad}- {_scalar(v)}")
    return "\n".join(out)


def _scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return repr(float(v)) if isinstance(v, (float, np.floating)) else str(int(v))
    if isinstance(v, (dict, list)):
        return "{}" if isinstance(v, dict) else "[]"
    return json.dumps(str(v))
