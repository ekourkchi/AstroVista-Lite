"""Fitting a pair of overlapping galaxies: ELLIPROF on each in turn, with the other's
current model taken out.

    initialisation:  A on I (B's core left out), then B on I - A (A's core left out)
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
from dataclasses import dataclass, field
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

    @property
    def label(self) -> str:
        return self.name or self.id

    def harmonics(self) -> Tuple[Optional[int], Optional[int]]:
        if self.mode:
            return MODES[self.mode]
        return self.cos3x, self.cos4x


@dataclass
class Settings:
    sky: float
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
    sky_sigma: Optional[float] = None
    provenance: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def mask_for(self, cid: str) -> np.ndarray:
        """The pixels left out while fitting component ``cid``: M0 and the other's core."""
        other = "B" if cid == "A" else "A"
        return self.fit_mask | self.cores[other].masked


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


@dataclass
class PairResult:
    status: str
    stability_reason: str = ""
    checkpoint_cycle: Optional[str] = None  # "0" (after initialisation), "1", ...
    commits: Dict[str, Commit] = field(default_factory=dict)  # the result's A and B
    partial: Dict[str, Commit] = field(default_factory=dict)  # newer, not joint
    total: Optional[np.ndarray] = None
    residual: Optional[np.ndarray] = None
    metrics: List[dict] = field(default_factory=list)
    calls: List[Call] = field(default_factory=list)
    attempted_r1: Dict[str, float] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)  # the last failure of each
    diagnostics: Dict[str, ArrayFit] = field(default_factory=dict)  # rejected fits
    bootstraps: Dict[str, float] = field(default_factory=dict)  # partial first fits' R1
    sky_check: dict = field(default_factory=dict)
    seconds: float = 0.0
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
}
OWN = {"NAME", "X0", "Y0", "R0", "R1", "NR", "MODE", "COS3X", "COS4X", "FIXCTR", "CORE", "ADAPTIVE"}
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
            nr=_num(k, "NR", where, 25, int),
            name=k.get("NAME", ""),
            mode=mode,
            cos3x=_num(k, "COS3X", where, None, int) if "COS3X" in k else None,
            cos4x=_num(k, "COS4X", where, None, int) if "COS4X" in k else None,
            fixctr=_num(k, "FIXCTR", where, 0, int),
            core=_num(k, "CORE", where) if "CORE" in k else None,
            adaptive="ADAPTIVE" in k,
        )
        if not 0 < c.r0 < c.r1:
            raise PairError(f"0 < R0 < R1 is needed{where}")
        if not 2 <= c.nr <= 100:
            raise PairError(f"NR must be 2 to 100{where}")
        if c.fixctr not in (0, 1, 2):
            raise PairError(f"FIXCTR must be 0, 1 or 2{where}")
        if c.core is not None and c.core < 0:
            raise PairError(f"CORE must be 0 or more{where}")
        out[cid] = c
    return out


def settings_from(shared: Dict[str, str]) -> Settings:
    s = Settings(
        sky=_num(shared, "SKY", " (the frozen sky: measure it with sky N SEGMENT)"),
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
    )
    if s.cycles < 1 or s.cycles > 50:
        raise PairError("CYCLES must be 1 to 50")
    if s.niter < 1 or s.niter > 1000:
        raise PairError("NITER must be 1 to 1000")
    if s.timeout <= 0:
        raise PairError("TIMEOUT must be positive")
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
    cores = {
        "A": make_core(a, b, matches["A"], shape, m0, invalid, settings.corecap),
        "B": make_core(b, a, matches["B"], shape, m0, invalid, settings.corecap),
    }
    ev = np.zeros(shape, bool)
    for c in (a, b):
        maskops.circle(ev, c.x0, c.y0, c.r1, True)
    ev &= ~m0
    for c in (a, b):
        if c.nr < NR_TEACHING:
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


def check(fit: ArrayFit, comp: Component, r1_call: float, image_finite: np.ndarray):
    """Whether a candidate fit passes: (ok, reason, kind, radius).  ``kind`` is "outer"
    for a failure at one isophote (a value not finite or out of range, a crossing, a rogue
    contour; an adaptive R1 may retry it if it is far enough out) and "other" otherwise;
    ``radius`` is the Rmaj where it failed."""
    if fit.returncode != 0 or fit.profile is None:
        return False, f"elliprof failed: {fit.message or fit.returncode}", "other", None
    p = fit.profile
    if len(p) != comp.nr:
        return False, f"{len(p)} isophotes instead of NR={comp.nr}", "other", None
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
    for k in range(1, len(r) - 1):
        dq1, dq2 = q[k] - q[k - 1], q[k] - q[k + 1]
        if abs(dq1) > Q_EXCURSION and abs(dq2) > Q_EXCURSION and dq1 * dq2 > 0:
            return False, f"an isolated axis ratio at Rmaj {r[k]:.1f}", "outer", float(r[k])
        if min(e[k - 1], e[k], e[k + 1]) > PA_MIN_ELLIP:
            a1 = _pa_difference(cols["alpha"][k], cols["alpha"][k - 1])
            a2 = _pa_difference(cols["alpha"][k], cols["alpha"][k + 1])
            if a1 > PA_EXCURSION and a2 > PA_EXCURSION:
                return False, f"an isolated PA at Rmaj {r[k]:.1f}", "outer", float(r[k])
    if abs(r[-1] - r1_call) > 0.01 * r1_call + 0.5:
        return False, f"the last isophote is at {r[-1]:.1f}, not R1={r1_call:.1f}", "other", None
    return True, "", "", None


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
    """Run the pair fit on a snapshot (see the module's description)."""
    s = snap.settings
    comps = snap.components
    image = snap.image
    finite = np.isfinite(image)
    result = PairResult(status="incomplete_initialization", run_id=snap.run_id)
    start = time.perf_counter()
    total_calls = 2 + 2 * s.cycles
    counter = {"n": 0}

    def say(**info):
        if progress is not None:
            info.update(call=counter["n"], calls=total_calls)
            progress(info)

    def stop_requested():
        return cancelled is not None and cancelled()

    def fit_component(
        cid: str,
        conditional: np.ndarray,
        r1_start: float,
        cycle: str,
        adaptive: Optional[bool] = None,
        bootstrap: bool = False,
    ):
        """A candidate for ``cid`` (with adaptive-R1 retries): a Commit, or None."""
        comp = comps[cid]
        cos3x, cos4x = comp.harmonics()
        mask = snap.mask_for(cid)
        usable = finite & ~mask
        r1 = r1_start
        adaptive = comp.adaptive if adaptive is None else adaptive
        tries = 4 if adaptive else 1
        for attempt in range(tries):
            if stop_requested():
                raise Cancelled()
            counter["n"] += 1 if attempt == 0 and not bootstrap else 0
            say(phase="fit", cycle=cycle, component=cid, r1=r1, retry=attempt)
            fit = fitter(
                conditional,
                comp.x0,
                comp.y0,
                comp.r0,
                r1,
                comp.nr,
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
            result.calls.append(
                Call(
                    cycle,
                    cid,
                    r1,
                    _clean_arguments(fit.arguments),
                    fit.returncode,
                    fit.seconds,
                    ok,
                    reason,
                    fit.precision,
                )
            )
            if ok:
                support = outer_support(fit.profile, usable)
                return Commit(
                    model=fit.model,
                    profile=fit.profile,
                    ellipses=profile_ellipses(fit.profile),
                    tag=cycle,
                    r1_call=r1,
                    support=support,
                    coverage_uncertain=any(x["usable"] < OUTER_SUPPORT for x in support),
                    bootstrap=bootstrap,
                    requested_r1=comp.r1,
                )
            result.failures[cid] = f"{cycle}: {reason}"
            result.diagnostics[cid] = fit
            if not adaptive or kind != "outer" or radius is None or radius < 0.65 * r1:
                break
            new = min(0.9 * r1, 0.95 * radius)
            if not (new > 1.5 * comp.r0 and new >= 0.5 * comp.r1):
                result.failures[cid] += f" (adaptive R1 stops: {new:.1f} is below its bounds)"
                break
            r1 = new
        return None

    def initialise(cid: str, conditional: np.ndarray):
        """The first model of ``cid``: at its R1, or else a partial bootstrap fit at 0.7,
        0.5 or 0.35 of it (only to start the cycles; never part of a result)."""
        first = fit_component(cid, conditional, comps[cid].r1, "init")
        if first is not None or not s.bootstrap:
            return first
        failure = result.failures.get(cid, "")
        for fraction in (0.7, 0.5, 0.35):
            r1 = fraction * comps[cid].r1
            if r1 <= 1.5 * comps[cid].r0:
                break
            c = fit_component(
                cid, conditional, r1, "init (bootstrap)", adaptive=False, bootstrap=True
            )
            if c is not None:
                result.bootstraps[cid] = r1
                return c
        result.failures[cid] = failure + " (no bootstrap fit at 0.7, 0.5 or 0.35 R1 passed either)"
        return None

    committed: Dict[str, Commit] = {}
    checkpoint: Optional[Tuple[str, Dict[str, Commit]]] = None
    try:
        a0 = initialise("A", image)
        if a0 is None:
            result.status = "incomplete_initialization"
            result.stability_reason = f"A: {result.failures.get('A', '')}"
            return _finish(result, snap, start, None, committed)
        committed["A"] = a0
        b0 = initialise("B", image - a0.model)
        if b0 is None:
            result.status = "incomplete_initialization"
            result.stability_reason = f"B: {result.failures.get('B', '')}"
            result.partial = {"A": a0}
            return _finish(result, snap, start, None, committed)
        committed["B"] = b0
        if not (a0.bootstrap or b0.bootstrap):
            checkpoint = ("0", dict(committed))
        last_ok = {"A": True, "B": True}
        for k in range(1, s.cycles + 1):
            tag = str(k)
            prev_a = committed["A"].model.copy()
            prev_b = committed["B"].model.copy()
            prev_t = prev_a + prev_b
            cand = fit_component("A", image - committed["B"].model, committed["A"].next_r1, tag)
            last_ok["A"] = cand is not None
            if cand is not None:
                committed["A"] = cand
            cand = fit_component("B", image - committed["A"].model, committed["B"].next_r1, tag)
            last_ok["B"] = cand is not None
            if cand is not None:
                committed["B"] = cand
            m = _metrics(snap, committed, prev_a, prev_b, prev_t)
            m.update(
                cycle=k,
                A_ok=last_ok["A"],
                B_ok=last_ok["B"],
                A_tag=committed["A"].tag,
                B_tag=committed["B"].tag,
            )
            result.metrics.append(m)
            say(phase="cycle", cycle=tag, metrics=m)
            joint = last_ok["A"] and last_ok["B"]
            if joint and not (committed["A"].bootstrap or committed["B"].bootstrap):
                checkpoint = (tag, dict(committed))
                if s.early_stop and m["stable"]:
                    break
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
        if last_ok["A"] and last_ok["B"]:
            if last["stable"]:
                result.status = "stable_valid_pair"
            else:
                result.status = "valid_pair_not_stable"
                result.stability_reason = last["reason"]
        else:
            failed = [c for c in ("A", "B") if not last_ok[c]]
            result.status = "retained_after_failed_update"
            result.stability_reason = "; ".join(
                f"{c} failed in cycle {last['cycle']}: {result.failures.get(c, '')}" for c in failed
            )
    except Cancelled:
        result.status = "cancelled"
        result.stability_reason = "stopped by the user"
    return _finish(result, snap, start, checkpoint, committed)


def _metrics(snap, committed, prev_a, prev_b, prev_t) -> dict:
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
        if value >= STABILITY:
            stable = False
            reasons.append(f"d{name}={value:.3f}")
    if stable is False and out.get("dT") is not None and out["dT"] < STABILITY:
        if any(out.get(k) is not None and out[k] >= STABILITY for k in ("dA", "dB")):
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
    sky = f"sky {snap.settings.sky:.6g} (frozen"
    sky += f", measurement {snap.sky_id})" if snap.sky_id else ", typed value)"
    if result.sky_check:
        sc = result.sky_check
        if "median" in sc:
            sky += (
                f"; on pixels accepted by the frozen sky mask the residual median is "
                f"{sc['median']:.3g}"
            )
    lines.append(sky)
    for cid, r1 in result.bootstraps.items():
        lines.append(
            f"note: {cid} was started from a partial bootstrap fit (R1 {r1:.1f}); it only "
            "started the cycles and is not part of any result"
        )
    for note in snap.notes:
        lines.append(f"note: {note}")
    calls = len(result.calls)
    lines.append(f"{calls} elliprof calls in {result.seconds:.1f} s")
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
                "measurement": snap.sky_id or "typed",
                "fixed_across_cycles": True,
            },
            "native": {
                "niter": s.niter,
                "gain": s.gain,
                "tie": s.tie,
                "rlaw": s.rlaw,
                "rmstar": s.rmstar,
                "timeout_seconds": s.timeout,
            },
            "pair": {
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
    lines.append(replay_line(requested, "MASK=m0"))
    if replay:
        lines += ["", "# the route from the original files"] + replay
    text("commands.txt", "\n".join(lines) + "\n")
    text("elliprof_calls.json", json.dumps([c.__dict__ for c in result.calls], indent=2) + "\n")
    for cid, c in result.commits.items():
        atomic(f"{cid}_profile.csv", lambda p, c=c: c.profile.to_csv(p, index=False))

        def reg(p, c=c, cid=cid):
            colour = "cyan" if cid == "A" else "orange"
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


def replay_line(line: str, mask: str) -> str:
    """A pair command for buffer 1 with its masks replaced by ``mask`` (a shared keyword,
    so it goes before A:)."""
    parts = re.split(r"(\s[Aa]:\s)", line, maxsplit=1)
    head, sep_, rest = (parts + ["", ""])[:3]
    head = re.sub(r"\s(MASK|MASKBUF|MASKFILE)=(\"[^\"]*\"|'[^']*'|\S+)", "", head, flags=re.I)
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
