"""Tests of the pair fit, the sky by segmentation, mask roles and windows."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from astro_vista import pair, sky, synthetic
from astro_vista.commands import CommandError, Session
from astro_vista.fitting import ArrayFit
from astro_vista.images import Image

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "pair_synthetic"


# ---------------------------------------------------------------- a fake elliprof
def fake_profile(x0, y0, r0, r1, nr, ellip=0.2, alpha=120.0):
    r = np.linspace(r0, r1, nr)
    return pd.DataFrame(
        {
            "Rmaj": r,
            "x0": np.full(nr, x0),
            "y0": np.full(nr, y0),
            "I0": 1000.0 / r,
            "alpha": np.full(nr, alpha),
            "ellip": np.full(nr, ellip),
            "I3": 0.0,
            "A3": 0.0,
            "I4": 0.0,
            "A4": 0.0,
            "slope": -1.0,
        }
    )


class FakeFitter:
    """Stands in for elliprof: records every call and returns a model whose value says
    which call made it; ``fail`` lists the calls (component, n-th call of it) that fail."""

    def __init__(self, fail=(), crossing=(), output=""):
        self.calls = []
        self.fail = set(fail)
        self.crossing = set(crossing)
        self.output = output

    def __call__(self, data, x0, y0, r0, r1, nr, **kw):
        component = "A" if x0 < 50 else "B"
        k = sum(1 for c in self.calls if c["component"] == component) + 1
        self.calls.append({"component": component, "k": k, "data": data.copy(), "kw": kw, "r1": r1})
        model = np.full(data.shape, float(len(self.calls)))
        profile = fake_profile(x0, y0, r0, r1, nr)
        if (component, k) in self.fail:
            profile.loc[nr - 1, "ellip"] = 0.99  # out of range: rejected
        if (component, k) in self.crossing:
            profile.loc[nr - 2, "ellip"] = 0.0  # a round ellipse sticking out of the next
            profile.loc[nr - 2, "Rmaj"] = profile.loc[nr - 1, "Rmaj"] * 0.99
        return ArrayFit(profile, model, 0, self.output, "", ["fake"], 0.0, "single")


def tiny_snapshot(cycles=4, **settings):
    rng = np.random.default_rng(0)
    image = 100.0 + rng.normal(0, 1, (60, 120))
    comps = {
        "A": pair.Component("A", 30.5, 30.5, 2, 20, 6),
        "B": pair.Component("B", 90.5, 30.5, 2, 20, 6),
    }
    return pair.prepare(
        image, "tiny", None, comps, pair.Settings(sky=100.0, cycles=cycles, **settings)
    )


# ---------------------------------------------------------------- the engine's invariants
def test_each_fit_sees_the_image_minus_the_committed_neighbour_and_the_sky_once():
    snap = tiny_snapshot(cycles=2)
    f = FakeFitter()
    result = pair.run(snap, fitter=f)
    assert len(f.calls) == 2 + 2 * 2  # initialisation, then A and B per cycle
    image = snap.image
    # init: A on I; B on I - A(init)
    assert np.array_equal(f.calls[0]["data"], image)
    assert np.array_equal(f.calls[1]["data"], image - 1.0)
    # cycle 1: A on I - B(init) (model 2), B on I - A(cycle 1) (model 3)
    assert np.array_equal(f.calls[2]["data"], image - 2.0)
    assert np.array_equal(f.calls[3]["data"], image - 3.0)
    for c in f.calls:
        assert c["kw"]["sky"] == 100.0  # the sky goes to every call, once, as elliprof's
    assert result.status in ("stable_valid_pair", "valid_pair_not_stable")
    # the result's arrays: the identities hold on every pixel
    a, b = result.commits["A"].model, result.commits["B"].model
    assert np.array_equal(result.total, a + b)
    assert np.allclose(result.residual, image - 100.0 - result.total, rtol=0, atol=1e-12)


def test_b_uses_the_committed_a_when_a_candidate_is_rejected():
    snap = tiny_snapshot(cycles=2)
    f = FakeFitter(fail={("A", 2)})  # A's cycle-1 candidate is rejected
    result = pair.run(snap, fitter=f)
    # the calls: A init (1), B init (2), A cycle 1 rejected (3), B cycle 1 (4)
    assert np.array_equal(f.calls[3]["data"], snap.image - 1.0)  # A's init model, not 3
    assert result.metrics[0]["A_ok"] is False


def test_a_rejected_update_is_never_called_stable():
    snap = tiny_snapshot(cycles=2)
    f = FakeFitter(fail={("A", 3), ("B", 3)})  # both fail in the last cycle
    result = pair.run(snap, fitter=f)
    assert result.status == "retained_after_failed_update"
    assert result.checkpoint_cycle == "1"
    assert result.commits["A"].tag == "1" and result.commits["B"].tag == "1"


def test_a_failed_last_b_returns_the_previous_joint_checkpoint():
    snap = tiny_snapshot(cycles=3)
    f = FakeFitter(fail={("B", 4)})  # B fails in cycle 3
    result = pair.run(snap, fitter=f)
    assert result.status == "retained_after_failed_update"
    assert result.checkpoint_cycle == "2"
    assert result.commits["A"].tag == "2"  # not A of cycle 3: the views are never mixed
    assert result.partial["A"].tag == "3"


def test_a_failed_initialisation_is_incomplete():
    snap = tiny_snapshot()
    snap.settings.bootstrap = False
    result = pair.run(snap, fitter=FakeFitter(fail={("A", 1)}))
    assert result.status == "incomplete_initialization"
    assert not result.complete and result.total is None


def test_a_bootstrap_only_starts_the_cycles():
    snap = tiny_snapshot(cycles=1)
    f = FakeFitter(fail={("A", 1)})  # A fails at its R1, passes at 0.7 R1
    result = pair.run(snap, fitter=f)
    assert result.bootstraps["A"] == pytest.approx(14.0)
    assert f.calls[1]["r1"] == pytest.approx(14.0)
    # the next fit of A goes back to the requested R1, and the result is from cycle 1
    a_calls = [c for c in f.calls if c["component"] == "A"]
    assert a_calls[-1]["r1"] == 20
    assert result.checkpoint_cycle == "1"
    assert not result.commits["A"].bootstrap


def test_stopping_between_a_and_b_keeps_the_checkpoint_apart_from_the_newer_a():
    snap = tiny_snapshot(cycles=3)
    f = FakeFitter()
    stop = {"now": False}

    def progress(info):
        if info.get("phase") == "fit" and info.get("cycle") == "1" and info["component"] == "A":
            stop["armed"] = True

    def cancelled():
        if stop.get("armed") and sum(1 for c in f.calls if c["component"] == "A") == 2:
            return True
        return False

    result = pair.run(snap, progress=progress, cancelled=cancelled, fitter=f)
    assert result.status == "cancelled"
    assert result.checkpoint_cycle == "0"
    assert result.commits["A"].tag == "init"
    assert result.partial["A"].tag == "1"


def test_a_stable_total_with_changing_components_is_not_stable():
    snap = tiny_snapshot(cycles=1)
    prev_a = np.full(snap.image.shape, 10.0)
    prev_b = np.full(snap.image.shape, 10.0)
    commits = {
        "A": pair.Commit(prev_a + 1.0, None, [], "1", 20, [], False),
        "B": pair.Commit(prev_b - 1.0, None, [], "1", 20, [], False),
    }
    snap.eval[:] = True
    m = pair._metrics(snap, commits, prev_a, prev_b, prev_a + prev_b)
    assert m["dT"] == 0.0
    assert not m["stable"]
    assert "components changing; total stable" in m["reason"]


def test_change_of_a_zero_model_is_not_evaluable():
    snap = tiny_snapshot(cycles=1)
    zero = np.zeros(snap.image.shape)
    commits = {k: pair.Commit(zero, None, [], "1", 20, [], False) for k in "AB"}
    m = pair._metrics(snap, commits, zero, zero, zero)
    assert m["dA"] is None and not m["stable"] and "not_evaluable" in m["reason"]


def test_a_sixth_order_warning_fails_the_component():
    comp = pair.Component("A", 30.5, 30.5, 2, 20, 6, mode="4,6")
    profile = fake_profile(30.5, 30.5, 2, 20, 6)
    finite = np.ones((60, 120), bool)
    fit = ArrayFit(
        profile,
        np.zeros((60, 120)),
        0,
        "elliprof: warning: the fitted PA wraps across 0/180 deg",
        "",
        [],
        0.0,
    )
    ok, reason, *_ = pair.check(fit, comp, 20, finite)
    assert not ok and "6th-order" in reason
    comp.mode = "3,4"  # the same warning without a 6th-order model: no matter
    assert pair.check(fit, comp, 20, finite)[0]


def test_the_crossing_tolerance_is_two_percent():
    outer = (0.0, 0.0, 10.0, 10.0, 0.0)
    assert pair.crossing((0.0, 0.0, 10.19, 10.19, 0.0), outer) == pytest.approx(1.019)
    assert pair.crossing((0.0, 0.0, 10.21, 10.21, 0.0), outer) > pair.CROSSING
    assert pair.crossing((0.0, 0.0, 10.19, 10.19, 0.0), outer) <= pair.CROSSING


def test_checks_refuse_nan_wrong_counts_and_short_radii():
    comp = pair.Component("A", 30.5, 30.5, 2, 20, 6)
    finite = np.ones((60, 120), bool)
    model = np.zeros((60, 120))

    def run_check(profile, r1=20):
        return pair.check(ArrayFit(profile, model, 0, "", "", [], 0.0), comp, r1, finite)

    p = fake_profile(30.5, 30.5, 2, 20, 6)
    assert run_check(p)[0]
    q = p.copy()
    q.loc[2, "x0"] = np.nan
    assert not run_check(q)[0]
    assert not run_check(p.iloc[:5])[0]  # NR rows, exactly
    assert not run_check(p, r1=25)[0]  # the last isophote is not at R1
    q = p.copy()
    q.loc[3, "ellip"] = 0.5  # an isolated axis ratio
    assert not run_check(q)[0]


# ---------------------------------------------------------------- cores and EVAL
def test_cores_belong_to_their_galaxy_and_leave_the_target_centre_usable():
    snap = tiny_snapshot()
    core_a, core_b = snap.cores["A"], snap.cores["B"]
    assert core_a.owner == "A" and core_a.x == 30.5
    assert core_a.shape == "circle"  # no source-finder run
    assert core_a.a == pytest.approx(min(0.65 * 60, pair.CORECAP))
    # A's core is left out while B is fitted, and the other way round
    assert np.array_equal(snap.mask_for("B"), snap.fit_mask | core_a.masked)
    assert np.array_equal(snap.mask_for("A"), snap.fit_mask | core_b.masked)
    assert not core_a.masked[30, 90] and not core_b.masked[30, 30]


def test_a_core_never_covers_the_other_centre_and_never_unmasks():
    image = np.ones((40, 60))
    m0 = np.zeros((40, 60), bool)
    m0[0:3, 0:3] = True
    comps = {
        "A": pair.Component("A", 20.5, 20.5, 1, 5, 4, core=100.0),
        "B": pair.Component("B", 26.5, 20.5, 1, 5, 4, core=100.0),
    }
    snap = pair.prepare(image, "x", None, comps, pair.Settings(sky=0), fit_mask=m0)
    assert not snap.cores["A"].masked[20, 26]  # B's centre pixel stays usable
    assert not snap.cores["B"].masked[20, 20]
    assert snap.fit_mask[0:3, 0:3].all()  # M0 stays as it was
    # a target centre already masked stays masked, and nothing loops
    m0[20, 26] = True
    snap = pair.prepare(image, "x", None, comps, pair.Settings(sky=0), fit_mask=m0)
    assert snap.mask_for("B")[20, 26]


def test_two_centres_on_one_object_are_refused():
    comps = {
        "A": pair.Component("A", 20.5, 20.5, 1, 5, 4),
        "B": pair.Component("B", 21.5, 20.5, 1, 5, 4),
    }
    with pytest.raises(pair.PairError):
        pair.prepare(np.ones((40, 60)), "x", None, comps, pair.Settings(sky=0))


def test_an_elliptical_core_from_a_matched_source():
    s = Session()
    s.folder = EXAMPLE
    s.execute("rd 1 pair.fits")
    s.execute("sources 1")
    job = s.pair_prepare(
        "1 SKY=1000 A: X0=294.3 Y0=263.8 R0=3 R1=120 B: X0=421.18 Y0=322.97 R0=3 R1=50"
    )
    assert job.snap.matches["A"].ok and job.snap.matches["B"].ok
    for core in job.snap.cores.values():
        assert core.shape == "ellipse" and core.b < core.a <= 50.0


# ---------------------------------------------------------------- the command
def test_the_pair_command_reads_its_scopes_strictly():
    words, shared, own = pair.parse(
        '1 SKY=5 CYCLES=3 A: NAME="NGC 1" X0=1 Y0=2 R0=1 R1=9 B: X0=5 Y0=6 R0=1 R1=8 ADAPTIVE'
    )
    assert words == ["1"] and shared == {"SKY": "5", "CYCLES": "3"}
    assert own["A"]["NAME"] == "NGC 1" and own["B"]["ADAPTIVE"] == "1"
    bad = [
        "1 SKY=5 A: X0=1 Y0=2 R0=1 R1=9 B: X0=5 Y0=6 R0=1 R1=8 SKY=3",  # shared after B:
        "1 SKY=5 X0=3 A: X0=1 Y0=2 R0=1 R1=9 B: X0=5 Y0=6 R0=1 R1=8",  # own before A:
        "1 SKY=5 A: X0=1 X0=2 Y0=2 R0=1 R1=9 B: X0=5 Y0=6 R0=1 R1=8",  # twice
        "1 SKY=5 A: X0=1 Y0=2 R0=1 R1=9",  # no B:
        "1 SKY=5 B: X0=5 Y0=6 R0=1 R1=8 A: X0=1 Y0=2 R0=1 R1=9",  # B: before A:
        "1 SKY=5 A: X0=1 Y0=2 R0=1 R1=9 MODE=4 COS4X=2 B: X0=5 Y0=6 R0=1 R1=8",
    ]
    for text in bad:
        with pytest.raises(pair.PairError):
            words, shared, own = pair.parse(text)
            pair.components_from(own)


def test_pair_output_buffers_never_collide():
    s = Session()
    s.folder = EXAMPLE
    s.execute("rd 1 pair.fits")
    s.execute("copy 1 2")
    tail = "A: X0=294.3 Y0=263.8 R0=3 R1=120 B: X0=421.18 Y0=322.97 R0=3 R1=50"
    for keys in ("TOMODEL=1", "MASKBUF=2 TORESID=2", "TOMODEL=5 TORESID=5"):
        with pytest.raises(CommandError):
            s.execute(f"pair 1 SKY=1000 {keys} {tail}")


def test_a_sky_layer_is_never_used_in_a_fit():
    s = Session()
    s.folder = EXAMPLE
    s.execute("rd 1 pair.fits")
    s.execute("mask load stars pair.dmask 1")
    before = hashlib.sha1(np.packbits(s.masks["stars"].masked).tobytes()).hexdigest()
    s.execute("sky 1 SEGMENT")
    assert s.masks["skymask"].role == "sky"
    assert hashlib.sha1(np.packbits(s.masks["stars"].masked).tobytes()).hexdigest() == before
    tail = "A: X0=294.3 Y0=263.8 R0=3 R1=120 B: X0=421.18 Y0=322.97 R0=3 R1=50"
    with pytest.raises(CommandError, match="sky-exclusion"):
        s.execute(f"pair 1 SKY=1000 MASK=skymask {tail}")
    with pytest.raises(CommandError, match="sky-exclusion"):
        s.execute("elliprof 1 X0=294.3 Y0=263.8 R0=3 R1=60 NR=10 MASK=skymask")
    s.execute("mask copy cand stars")
    s.execute("mask role cand candidate")
    with pytest.raises(CommandError, match="candidate"):
        s.execute(f"pair 1 SKY=1000 MASK=cand {tail}")
    # the galaxies keep their samples in a fit even though the sky mask covers them
    assert s.masks["skymask"].masked[263, 294]


def test_the_example_pair_settles_and_recovers_the_truth(tmp_path):
    s = Session()
    reply = s.execute(f"run {EXAMPLE / 'session.txt'}")
    record = s.pairs[s.buffers[1].key]
    result, snap = record.result, record.job.snap
    assert result.status == "stable_valid_pair", reply.text
    scene = synthetic.pair()
    for cid, tolerance in (("A", 0.03), ("B", 0.10)):
        g = scene.galaxies[cid]
        p = result.commits[cid].profile
        r = p["Rmaj"].to_numpy()
        inside = (r > 2 * 3) & (r < 0.8 * r.max())
        di = np.abs(p["I0"].to_numpy()[inside] / g.intensity(r[inside]) - 1)
        assert np.median(di) < tolerance
        de = np.abs(p["ellip"].to_numpy()[inside] - g.ellip)
        assert de.max() < 0.05
        pa = np.abs(((p["alpha"].to_numpy()[inside] - 90 - g.angle + 90) % 180) - 90)
        assert pa.max() < 5.0
    # the residual is the image minus the sky and both models on every pixel
    assert np.allclose(result.residual, snap.image - 1000.0 - result.total, rtol=0, atol=1e-9)
    assert np.isfinite(result.residual).all()
    # its buffers, and the export
    names = [b.name for b in s.buffers.values()]
    assert any(n.endswith("pair residual") for n in names)
    s.execute(f"pairexport 1 {tmp_path / 'out'}")
    out = tmp_path / "out"
    assert json.loads((out / "transaction.json").read_text()) == {"export": "complete"}
    assert json.loads((out / "result.json").read_text())["status"] == "stable_valid_pair"
    from astropy.io import fits

    science = fits.getdata(out / "inputs" / "science.fits")
    total = fits.getdata(out / "total_model.fits")
    residual = fits.getdata(out / "residual_raw.fits")
    assert residual.dtype.kind == "f" and residual.dtype.itemsize == 8
    atol = 8 * 2.2e-16 * (np.abs(science) + 1000.0 + np.abs(total)) + 1e-12
    assert (np.abs(residual - (science - 1000.0 - total)) <= atol).all()
    iterations = json.loads((out / "iterations.json").read_text())["iterations"]
    assert len(iterations) == len(result.calls)
    assert iterations[0]["input"] == "D"
    assert next(e for e in iterations if e["galaxy"] == "B")["input"] == "D - M_A"
    for e in iterations:
        if e["passed"]:
            assert (out / e["profile"]).exists() and len(e["model_sha256"]) == 64
    manifest = json.loads((out / "manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    # replay from the exported inputs gives the same result
    t = Session()
    t.folder = out
    lines = (out / "commands.txt").read_text().split("\n\n")[0]
    for line in lines.splitlines():
        if line and not line.startswith("#"):
            t.execute(line)
    again = t.pairs[t.buffers[1].key].result
    assert again.status == result.status
    assert np.allclose(again.total, result.total, rtol=0, atol=1e-6 * np.abs(result.total).max())


def test_a_hard_blend_ends_honestly():
    scene = synthetic.pair(separation=95.0, noise=4.0)
    a, b = scene.galaxies["A"], scene.galaxies["B"]
    comps = {
        "A": pair.Component("A", a.x, a.y, 3, 180, 25),
        "B": pair.Component("B", b.x, b.y, 3, 80, 20),
    }
    snap = pair.prepare(
        scene.data, "hard", None, comps, pair.Settings(sky=1000.0), fit_mask=scene.star_mask
    )
    result = pair.run(snap)
    assert result.status != "stable_valid_pair"
    if result.status == "retained_after_failed_update":
        assert result.checkpoint_cycle is not None and result.complete


def test_inputs_are_unchanged_by_a_pair_fit():
    s = Session()
    s.folder = EXAMPLE
    s.execute("rd 1 pair.fits")
    s.execute("mask load stars pair.dmask 1")
    s.execute("mask role stars fit")
    image = s.buffers[1].data.copy()
    stars = s.masks["stars"].masked.copy()
    s.execute(
        "pair 1 SKY=1000 MASK=stars CYCLES=1 A: X0=294.3 Y0=263.8 R0=3 R1=120 "
        "B: X0=421.18 Y0=322.97 R0=3 R1=50 NR=20"
    )
    assert np.array_equal(s.buffers[1].data, image)
    assert np.array_equal(s.masks["stars"].masked, stars)
    assert (EXAMPLE / "pair.fits").exists()


def test_elliprof_takes_harmonics_minus_and_raw():
    s = Session()
    s.folder = EXAMPLE
    s.execute("rd 1 pair.fits")
    s.execute("mask load stars pair.dmask 1")
    plain = "elliprof 1 X0=294.3 Y0=263.8 R0=3 R1=80 NR=12 SKY=1000 MASK=stars TOMODEL=3"
    s.execute(plain)
    model = s.buffers[3].data.copy()
    s.execute(plain + " MODE=none FIXCTR=2 TORESID=4 RAW")
    assert not np.array_equal(s.buffers[3].data, model)  # the options reach elliprof
    raw = s.buffers[4].data
    assert np.isfinite(raw[s.masks["stars"].masked]).all()  # RAW: masked pixels kept
    s.execute("[5] = [1] * 0")
    s.execute(plain.replace("TOMODEL=3", "TOMODEL=6") + " MINUS=5")
    assert np.allclose(s.buffers[6].data, model)  # minus nothing: the same fit
    with pytest.raises(CommandError):
        s.execute(plain + " MODE=3,4 COS3X=2")


# ---------------------------------------------------------------- the sky
def test_the_sky_of_a_blank_field():
    scene = synthetic.blank()
    r = sky.measure(scene.data, "blank")
    assert r.state == "usable_support"
    sigma = r.sigma
    # 0.5-sigma masking reads pure noise a little low (about 0.2 sigma); the test allows it
    assert abs(r.level - scene.sky) < 0.3 * sigma
    g = r.gradient()
    assert g["size"] <= 0.2 * sigma  # per 1000 px, an absolute floor at zero gradient


def test_the_sky_gradient_is_measured():
    scene = synthetic.blank(gradient=(0.004, 0.0))
    r = sky.measure(scene.data, "gradient")
    g = r.gradient()
    true = 4.0  # per 1000 px
    assert abs(g["gx"] - true) <= 0.25 * true + 0.2 * r.sigma


def test_correlated_noise_gives_an_honest_sampling_error():
    errors, levels = [], []
    for seed in range(8):
        scene = synthetic.blank(seed=seed, correlated=1.0, n_stars=10)
        r = sky.measure(scene.data, "c")
        assert r.state == "usable_support"
        errors.append(r.sampling_error)
        levels.append(r.level)
    # the quoted error is of the order of the scatter between seeds
    scatter = np.std(levels)
    assert 0.2 * scatter < np.median(errors) < 5 * scatter


def test_correlated_noise_is_not_taken_for_sources():
    # noise correlated over a few pixels (as in drizzled images) averages down less in the
    # filtered image; the detection's noise is scaled for it, so the threshold does not
    # mask the noise itself, and the level stays close to the truth (500, sigma 10)
    r = sky.measure(synthetic.blank(correlated=2.0, n_stars=10).data, "c")
    assert r.state == "usable_support"
    assert abs(r.level - 500.0) < 0.3 * 10.0
    assert "correlated_noise" in r.detection
    white = sky.measure(synthetic.blank(n_stars=10).data, "w")
    assert abs(white.level - 500.0) < 0.3 * 10.0


def test_a_halo_filling_the_frame_is_not_usable_support():
    scene = synthetic.halo()
    r = sky.measure(scene.data, "halo")
    assert r.state in ("candidate_inspect_support", "no_estimate")
    assert r.reasons


def test_valid_data_in_one_corner_only():
    scene = synthetic.blank()
    data = scene.data.copy()
    data[:, 250:] = np.nan
    data[200:, :] = np.nan
    r = sky.measure(data, "corner")
    assert r.state != "usable_support"
    assert any("span" in reason for reason in r.reasons)


def test_grow_zero_keeps_the_segments_exactly():
    scene = synthetic.blank(n_stars=5)
    r0 = sky.measure(scene.data, "g", grow=0)
    r2 = sky.measure(scene.data, "g", grow=2)
    assert r0.exclusion.sum() < r2.exclusion.sum()
    assert (r2.exclusion | ~r0.exclusion).all()  # growing only adds


def test_a_sky_measurement_turns_stale():
    s = Session()
    s.folder = EXAMPLE
    s.execute("rd 1 pair.fits")
    s.execute("mask load stars pair.dmask 1")
    s.execute("sky 1 SEGMENT MASK=stars")
    sid = s.sky_of[s.buffers[1].key]
    assert s.sky_stale(sid, s.buffers[1]) is None
    s.execute("mask circle stars 10 10 5")
    assert "stars" in s.sky_stale(sid, s.buffers[1])
    s.execute("ac 1 1")
    assert "image" in s.sky_stale(sid, s.buffers[1])


# ---------------------------------------------------------------- windows
def test_a_window_keeps_the_wcs_through_two_crops():
    from astropy.io import fits
    from astropy.wcs import WCS

    header = fits.Header()
    header.update(
        {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": 150.0,
            "CRVAL2": 2.0,
            "CRPIX1": 60.0,
            "CRPIX2": 40.0,
            "CD1_1": -1e-4,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 1e-4,
        }
    )
    s = Session()
    s.buffers[1] = Image("w.fits", np.arange(100 * 120, dtype=np.float32).reshape(100, 120), header)
    s.execute("window 1 BOX=10,20,90,80 OUT=2")
    s.execute("window 2 BOX=5,5,50,40 OUT=3")
    assert s.buffers[3].shape == (35, 45)
    assert s.buffers[3].data[0, 0] == s.buffers[1].data[25, 15]
    w1, w3 = WCS(s.buffers[1].header), WCS(s.buffers[3].header)
    for x, y in ((0, 0), (10.3, 20.7), (44, 34)):
        a = w3.pixel_to_world_values(x, y)
        b = w1.pixel_to_world_values(x + 15, y + 25)
        assert np.allclose(a, b, atol=1e-12)
    assert s.buffers[3].header["LTV1"] == -15 and s.buffers[3].header["LTV2"] == -25


def test_window_edges_and_errors():
    s = Session()
    s.buffers[1] = Image("w.fits", np.zeros((50, 60), np.float32))
    reply = s.execute("window 1 BOX=-5,-5,10,10")
    assert "clipped" in reply.text and s.buffers[2].shape == (10, 10)
    for bad in ("BOX=70,70,80,80", "BOX=1.5,0,10,10", "BOX=0,0,10", "BOX=0,0,10,10 OUT=1"):
        with pytest.raises(CommandError):
            s.execute(f"window 1 {bad}")


# ---------------------------------------------------------------- the window
# ---------------------------------------------------------------- the sky factor
class FlatFitter(FakeFitter):
    """A fake elliprof whose models are zero: the residual is the image minus the sky."""

    def __call__(self, data, *args, **kw):
        fit = super().__call__(data, *args, **kw)
        fit.model[:] = 0.0
        return fit


def test_the_residual_sky_gives_the_next_factor():
    rng = np.random.default_rng(3)
    residual = rng.normal(-4.0, 1.0, (200, 200))
    support = np.ones(residual.shape, bool)
    level = sky.residual_level(residual, support, base=200.0, factor=1.1)
    assert abs(level["median"] + 4.0) < 0.05
    assert abs(level["next_factor"] - (1.1 + level["median"] / 200.0)) < 1e-12
    assert abs(level["next_sky"] - (220.0 + level["median"])) < 1e-9
    assert sky.residual_level(residual, np.zeros_like(support))["pixels"] == 0


def test_skyfactor_multiplies_the_base_and_is_bounded():
    s = pair.settings_from({"SKY": "1000", "SKYFACTOR": "0.98", "SKYITER": "3"})
    assert s.sky == pytest.approx(980.0) and s.sky_base == 1000.0 and s.sky_iter == 3
    with pytest.raises(pair.PairError):
        pair.settings_from({"SKY": "1000", "SKYFACTOR": "2"})
    with pytest.raises(pair.PairError):
        pair.settings_from({"SKY": "1000", "SKYITER": "-1"})


def test_sky_iterations_refit_until_the_residual_sky_is_zero():
    snap = tiny_snapshot(cycles=1)
    snap = pair.with_sky(snap, 1.0)
    snap.settings.sky_base, snap.settings.sky_iter = 105.0, 6
    snap = pair.with_sky(snap, 1.0)  # sky 105: 5 too high for an image of 100
    f = FlatFitter()
    result = pair.run_sky(snap, np.ones(snap.image.shape, bool), fitter=f)
    rows = result.sky_iterations
    assert len(rows) >= 2 and rows[0]["factor"] == 1.0
    assert rows[0]["median"] == pytest.approx(-5.0, abs=0.1)
    chosen = [r for r in rows if r.get("chosen")]
    assert len(chosen) == 1
    assert chosen[0]["sky"] == pytest.approx(100.0, abs=0.1)
    assert abs(chosen[0]["median"]) < abs(rows[0]["median"])
    assert pair.chosen_factor(result, snap.settings) == chosen[0]["factor"]
    # each pair fit gave every elliprof call its own sky
    skies_used = sorted({round(c["kw"]["sky"], 6) for c in f.calls})
    assert len(skies_used) == len(rows)
    # without SKYITER it is one ordinary run
    plain = pair.run_sky(tiny_snapshot(cycles=1), np.ones((60, 120), bool), fitter=FlatFitter())
    assert plain.sky_iterations == []


class AbsorbingFitter(FakeFitter):
    """A fake elliprof whose models take up most of any change of the sky (as a halo that
    fills the frame does)."""

    def __call__(self, data, *args, **kw):
        fit = super().__call__(data, *args, **kw)
        fit.model[:] = (float(np.median(data)) - kw["sky"] - 5.0) / 2.0
        return fit


def test_sky_iterations_stop_when_the_sky_is_degenerate_with_the_galaxies():
    snap = tiny_snapshot(cycles=1)
    snap.settings.sky_base, snap.settings.sky_iter = 105.0, 6
    snap = pair.with_sky(snap, 1.0)
    result = pair.run_sky(snap, np.ones(snap.image.shape, bool), fitter=AbsorbingFitter())
    rows = result.sky_iterations
    assert len(rows) == 2 and rows[1]["response"] < pair.SKY_RESPONSE
    assert "degenerate" in rows[1]["note"]
    assert rows[0].get("chosen") and pair.chosen_factor(result, snap.settings) == 1.0


def test_skyiter_needs_a_sky_measurement_and_replays_the_factor_kept(tmp_path):
    s = Session()
    s.execute(f"rd 1 {EXAMPLE / 'pair.fits'}")
    line = (
        "pair 1 SKY=1003 SKYITER=3 A: X0=294.3 Y0=263.8 R0=3 R1=120 "
        "B: X0=421.18 Y0=322.97 R0=3 R1=50 NR=20"
    )
    with pytest.raises(CommandError, match="SKYITER needs"):
        s.execute(line)
    s.execute(f"mask load stars {EXAMPLE / 'pair.dmask'} 1")
    s.execute("mask role stars fit")
    s.execute("sources 1")
    s.execute("sky 1 SEGMENT MASK=stars")
    reply = s.execute(line.replace("SKYITER=3", "SKYITER=3 MASK=stars"))
    record = s.pairs[s.buffers[1].key]
    result, snap = record.result, record.job.snap
    rows = result.sky_iterations
    assert rows and any(r.get("chosen") for r in rows), reply.text
    kept = next(r for r in rows if r.get("chosen"))
    assert snap.settings.sky_factor == kept["factor"]  # the snapshot is the fit kept
    assert abs(kept["median"]) <= abs(rows[0]["median"])
    assert "clipped_median" in result.sky_check and "skyfactor" in reply.values
    s.execute(f"pairexport 1 {tmp_path / 'out'}")
    commands = (tmp_path / "out" / "commands.txt").read_text()
    pair_lines = [x for x in commands.splitlines() if x.startswith("pair 1")]
    assert pair_lines and all("SKYITER" not in x for x in pair_lines)
    if kept["factor"] != 1.0:
        assert all(f"SKYFACTOR={kept['factor']:.10g}" in x for x in pair_lines)
    else:  # here the sky is degenerate with the halo: the sky asked for is kept
        assert "degenerate" in rows[-1].get("note", "")
        assert all("SKYFACTOR" not in x for x in pair_lines)


def test_the_sky_filter_and_a_level_without_support_verdict():
    assert sky.gaussian_kernel(2.0).shape == (5, 5)
    assert sky.gaussian_kernel(2.0).sum() == pytest.approx(1.0, rel=1e-6)
    scene = synthetic.blank(n_stars=10)
    r = sky.measure(scene.data, "blank", fwhm=2.0)
    assert r.settings["FWHM"] == 2.0 and "Gaussian" in r.detection["filter"]
    assert abs(r.level - scene.sky) < 1.0
    # little support: the verdict says so, but the level is kept for the user to judge
    halo = synthetic.halo()
    h = sky.measure(halo.data, "halo", thresh=0.05)
    if h.state == "no_estimate" and h.support.sum() >= 100:
        assert np.isfinite(h.level)


def test_the_light_peak_refines_a_centre():
    g = synthetic.Galaxy(40.3, 30.7, ie=100, re=8, ellip=0.2, angle=30)
    data = g.image((60, 80))
    x, y = pair.peak_centre(data, 38.0, 29.0)
    assert abs(x - 40.3) < 0.5 and abs(y - 30.7) < 0.5


def test_a_pair_folder_is_read(tmp_path):
    from astropy.io import fits

    from astro_vista.masks import Mask, save

    fits.PrimaryHDU(np.zeros((40, 60), np.float32)).writeto(tmp_path / "g.fits")
    m = np.zeros((40, 60), bool)
    m[:3] = True
    save(Mask("m", m), tmp_path / "g.dmask")
    (tmp_path / "centers.reg").write_text(
        "# Region file format: DS9 version 4.1\nimage\n"
        "point(11.000000,21.000000) # point=cross text={B: small}\n"
        "point(41.0,21.0) # point=cross text={A: big}\n"
    )
    folder = pair.read_folder(tmp_path)
    assert folder.image.name == "g.fits" and folder.hdu is None
    assert [p.name for p in folder.masks] == ["g.dmask"]
    assert folder.centres["A"] == (40.5, 20.5, "big")  # ELLIPROF's coordinates
    assert folder.centres["B"] == (10.5, 20.5, "small")
    r1 = pair.suggested_r1(folder.centres, (40, 60))
    assert r1["A"] == pytest.approx(min(0.9 * 19.5, 0.6 * 30), abs=0.06)  # edge, separation
    with pytest.raises(pair.PairError):
        pair.read_folder(tmp_path / "nothing")


# ---------------------------------------------------------------- block and grow
def grow_snapshot(steps=2, block=None, start=8.0, on_fail="hold"):
    rng = np.random.default_rng(0)
    image = 100.0 + rng.normal(0, 1, (60, 120))
    comps = {
        "A": pair.Component("A", 30.5, 30.5, 2, 20, 6, start=start),
        "B": pair.Component("B", 90.5, 30.5, 2, 20, 6, start=start, block=block),
    }
    settings = pair.Settings(sky=100.0, cycles=1, steps=steps, on_fail=on_fail)
    return pair.prepare(image, "tiny", None, comps, settings)


def test_the_galaxies_grow_from_start_to_r1():
    snap = grow_snapshot(steps=2)
    f = FakeFitter()
    result = pair.run(snap, fitter=f)
    r1s = [round(c["r1"], 3) for c in f.calls]
    middle = round(8.0 * (20 / 8.0) ** 0.5, 3)
    assert r1s == [8.0, 8.0, middle, middle, 20.0, 20.0, 20.0, 20.0]
    assert [g["step"] for g in result.growth] == [1, 2]
    assert result.status in ("stable_valid_pair", "valid_pair_not_stable")
    assert result.commits["A"].r1_call == 20.0


def test_a_galaxy_whose_growth_fails_stays_at_its_last_radius():
    snap = grow_snapshot(steps=2)
    f = FakeFitter(fail={("B", 2)})  # B's first growth step fails
    result = pair.run(snap, fitter=f)
    b_r1 = [round(c["r1"], 3) for c in f.calls if c["component"] == "B"]
    assert b_r1 == [8.0, round(8.0 * 2.5**0.5, 3), 8.0, 8.0]
    assert result.notes_growth and "B could not grow" in result.notes_growth[0]
    assert result.commits["B"].r1_call == 8.0


def test_a_failed_growth_step_tries_smaller_steps_and_grows_on():
    snap = grow_snapshot(steps=2, on_fail="reduce")
    f = FakeFitter(fail={("B", 2)})  # B's first growth step fails
    result = pair.run(snap, fitter=f)
    b_r1 = [round(c["r1"], 3) for c in f.calls if c["component"] == "B"]
    full = round(8.0 * 2.5**0.5, 3)
    half = round(8.0 + 0.5 * (8.0 * 2.5**0.5 - 8.0), 3)
    assert b_r1[:3] == [8.0, full, half]  # the failed step, then halfway, which passes
    assert b_r1[3] == 20.0  # and it grows on to its final R1
    assert result.commits["B"].r1_call == 20.0
    assert "smaller step" in result.notes_growth[0] and result.notes_growth[0].startswith("B")


def test_a_round_core_is_not_failed_for_its_angle():
    def profile(ellip, alpha):
        p = fake_profile(30.5, 30.5, 2, 20, 6)
        p["ellip"], p["alpha"] = ellip, alpha
        return p

    comp = pair.Component("A", 30.5, 30.5, 2, 20, 6)
    finite = np.ones((60, 120), bool)

    def verdict(p):
        fit = ArrayFit(p, np.zeros((60, 120)), 0, "", "", ["fake"], 0.0, "single")
        return pair.check(fit, comp, 20, finite)

    # a 25 degree swing of one isophote of a nearly round core: an ill-defined angle
    ok, reason, *_ = verdict(profile([0.10] * 6, [60, 60, 35, 60, 60, 60]))
    assert ok, reason
    # the same swing in a flattened galaxy is a real twist
    ok, reason, *_ = verdict(profile([0.35] * 6, [60, 60, 35, 60, 60, 60]))
    assert not ok and "angle" in reason
    # and an isolated change of the axis ratio
    ok, reason, *_ = verdict(profile([0.1, 0.1, 0.35, 0.1, 0.1, 0.1], [60] * 6))
    assert not ok and "axis ratio" in reason


def test_a_block_is_left_out_of_the_first_fit_only():
    snap = grow_snapshot(steps=0, block=55.0, start=None)  # the core is 39
    assert "B" in snap.blocks and snap.blocks["B"].masked.sum() > 0
    first = snap.mask_for("A", first=True)
    later = snap.mask_for("A")
    assert first.sum() > later.sum()
    assert not first[30, 30]  # never A's centre
    seen = []

    class Watch(FakeFitter):
        def __call__(self, data, x0, y0, r0, r1, nr, **kw):
            seen.append((x0 < 50, int(kw["masked"].sum())))
            return super().__call__(data, x0, y0, r0, r1, nr, **kw)

    pair.run(snap, fitter=Watch())
    a_masks = [n for is_a, n in seen if is_a]
    assert a_masks[0] == int(first.sum()) and set(a_masks[1:]) == {int(later.sum())}
    with pytest.raises(pair.PairError):
        pair.components_from(
            {
                "A": {"X0": "1", "Y0": "1", "R0": "2", "R1": "10", "START": "12"},
                "B": {"X0": "50", "Y0": "1", "R0": "2", "R1": "10"},
            }
        )


def test_the_report_tells_what_happened():
    snap = grow_snapshot(steps=2)
    result = pair.run(snap, fitter=FakeFitter(fail={("B", 2)}))
    text = pair.report(result, snap, "pair 1 ...")
    for section in ("## status", "## settings", "## growth", "## calls", "## hints"):
        assert section in text
    assert "B could not grow" in text
    assert "residual rings" in text


# ---------------------------------------------------------------- the iterative workflow
class SteadyFitter(FakeFitter):
    """A fake elliprof whose model for a galaxy is the same in every call (it converges)."""

    def __call__(self, data, x0, y0, r0, r1, nr, **kw):
        fit = super().__call__(data, x0, y0, r0, r1, nr, **kw)
        fit.model[:] = 1.0 if x0 < 50 else 2.0
        return fit


def iterative_snapshot(a_kw=None, b_kw=None, **settings):
    rng = np.random.default_rng(1)
    image = 100.0 + rng.normal(0, 1, (60, 120))
    comps = {
        "A": pair.Component("A", 30.5, 30.5, 2, 20, 6, fixctr=1, **(a_kw or {})),
        "B": pair.Component("B", 90.5, 30.5, 2, 20, 6, fixctr=1, **(b_kw or {})),
    }
    settings.setdefault("cycles", 2)
    return pair.prepare(
        image, "tiny", None, comps, pair.Settings(sky=100.0, cores=False, **settings)
    )


def test_every_fit_sees_d_minus_the_current_companion_and_d_never_changes():
    snap = iterative_snapshot(a_kw={"start": 8, "factor": 1.5}, b_kw={"start": 6, "step": 5})
    d = snap.image.copy()
    f = FakeFitter()
    result = pair.run(snap, fitter=f)
    assert np.array_equal(snap.image, d)  # D is immutable
    calls = f.calls
    assert calls[0]["component"] == "A" and np.array_equal(calls[0]["data"], d)  # A first, on D
    models = {}  # the model each call made: FakeFitter's model = its call number
    for k, c in enumerate(calls, 1):
        if k == 1:
            continue
        other = "B" if c["component"] == "A" else "A"
        last_other = max(j for j, x in enumerate(calls[: k - 1], 1) if x["component"] == other)
        # exactly D minus the companion's latest model, never a running subtraction
        assert np.array_equal(c["data"], d - float(last_other)), k
    for h, c in zip(result.history, calls):
        assert h["input"] == (
            "D" if h["call"] == 1 else f"D - M_{'B' if c['component'] == 'A' else 'A'}"
        )


def test_no_galaxy_is_masked_from_the_other():
    snap = iterative_snapshot()
    f = FakeFitter()
    pair.run(snap, fitter=f)
    assert all(int(c["kw"]["masked"].sum()) == int(snap.fit_mask.sum()) for c in f.calls)
    assert all(core.masked.sum() == 0 for core in snap.cores.values())


def test_each_galaxy_follows_its_own_radius_schedule():
    a = pair.Component("A", 0, 0, 2, 180, 6, radii=[40, 60, 90, 130, 180])
    assert a.schedule() == [40, 60, 90, 130, 180]
    b = pair.Component("B", 0, 0, 2, 50, 6, start=20, step=10)
    assert b.schedule() == [20, 30, 40, 50]
    c = pair.Component("C", 0, 0, 2, 100, 6, start=30, factor=1.5)
    assert c.schedule() == pytest.approx([30, 45, 67.5, 100])
    snap = iterative_snapshot(a_kw={"radii": [8, 12, 20]}, b_kw={"start": 6, "step": 7}, cycles=1)
    f = FakeFitter()
    pair.run(snap, fitter=f)
    a_r1 = [c["r1"] for c in f.calls if c["component"] == "A"]
    b_r1 = [c["r1"] for c in f.calls if c["component"] == "B"]
    assert a_r1 == [8, 12, 20, 20]  # its list, then the cycle at the target
    assert b_r1 == [6, 13, 20, 20]
    with pytest.raises(pair.PairError):
        pair.components_from(
            {
                "A": {"X0": "1", "Y0": "1", "R0": "2", "R1": "10", "STEP": "2", "FACTOR": "2"},
                "B": {"X0": "50", "Y0": "1", "R0": "2", "R1": "10"},
            }
        )


def test_it_stops_after_two_consecutive_converged_cycles_at_the_targets():
    snap = iterative_snapshot(a_kw={"start": 8, "factor": 2}, cycles=10, consecutive=2)
    result = pair.run(snap, fitter=SteadyFitter())
    assert result.status == "stable_valid_pair"
    assert result.metrics[-1].get("converged")
    # growth (8 -> 16 -> 20: 2 cycles), then 2 converged cycles at the targets, not 10
    assert len(result.metrics) < 2 + 10
    assert [m["converged_streak"] for m in result.metrics][-2:] == [1, 2]


def test_a_failure_keeps_the_last_valid_model_and_can_stop_the_run():
    snap = iterative_snapshot(cycles=3, on_fail="stop")
    f = FakeFitter(fail={("A", 3)})  # A's third fit (cycle 2) fails
    result = pair.run(snap, fitter=f)
    assert result.status == "retained_after_failed_update"
    assert "stopped there" in result.stability_reason
    assert result.commits["A"].tag == "1"  # the last valid A, intact
    failed = [h for h in result.history if not h["passed"]]
    assert failed and failed[0]["galaxy"] == "A" and failed[0]["model"] is None
    assert "R1 20.0" in result.failures["A"]


def test_fixed_centres_are_passed_and_the_drift_is_recorded():
    snap = iterative_snapshot()
    f = FakeFitter()
    result = pair.run(snap, fitter=f)
    assert {c["kw"]["fixctr"] for c in f.calls} == {1}
    assert all(c.centre_drift is not None for c in result.calls if c.passed)


def test_the_products_and_every_iteration_are_kept():
    snap = iterative_snapshot(a_kw={"start": 8, "step": 6})
    result = pair.run(snap, fitter=FakeFitter())
    a, b = result.commits["A"].model, result.commits["B"].model
    assert np.array_equal(result.total, a + b)
    assert np.array_equal(result.residual, snap.image - 100.0 - a - b)
    assert len(result.history) == len(result.calls)
    for h, c in zip(result.history, result.calls):
        assert h["r1"] == c.r1 and h["galaxy"] == c.component
        assert (h["model"] is not None) == c.passed
        assert h["companion"] == c.companion
    assert result.schedules["A"] == [8, 14, 20]


# ---------------------------------------------------------------- damping and the search
def test_damping_subtracts_a_blend_and_undamped_cycles_end_the_run():
    snap = iterative_snapshot(cycles=2, damp=0.5, polish=2)
    d = snap.image.copy()
    f = FakeFitter()
    result = pair.run(snap, fitter=f)
    calls = f.calls  # FakeFitter's model = its call number
    # cycle 1: A on D - B(init) = D - 2; B on D - (0.5 * 3 + 0.5 * 1) = D - 2
    assert np.array_equal(calls[2]["data"], d - 2.0)
    assert np.array_equal(calls[3]["data"], d - 2.0)
    polish = [m for m in result.metrics if m["polish"]]
    assert len(polish) == 2 and all(m["damping"] == 1.0 for m in polish)
    # the result is ELLIPROF's own models of an undamped cycle
    assert result.checkpoint_cycle == str(polish[-1]["cycle"])
    a, b = result.commits["A"].model, result.commits["B"].model
    assert np.array_equal(result.residual, d - 100.0 - a - b)
    k = len(calls)  # the last B fit saw D minus the last A model itself
    assert np.array_equal(calls[k - 1]["data"], d - float(k - 1))


def test_the_search_fits_every_variant_and_keeps_one():
    snap = iterative_snapshot(
        a_kw={"start": 8, "factor": 1.5},
        b_kw={"start": 8, "factor": 1.5},
        try_damp=[1.0, 0.5],
        try_fix=[1, 0],
        try_factor=[1.5, 1.25],
    )
    labels = [label for label, _v in pair.variants(snap)]
    assert len(labels) == 8 and "damp 0.5, centres free, growth x1.25" in labels
    result = pair.run_search(snap, fitter=SteadyFitter())
    assert len(result.search) == 8
    assert sum(1 for r in result.search if r.get("chosen")) == 1
    assert result.snapshot is not None and result.quality.get("score") is not None
    with pytest.raises(pair.PairError):
        pair.settings_from(
            {"SKY": "1", "TRYDAMP": "1,0.8,0.6", "TRYFIX": "0,1,2", "TRYFACTOR": "1.2,1.5"}
        )


def test_the_search_prefers_fixed_centres_unless_free_ones_clearly_win():
    snap = iterative_snapshot(try_fix=[1, 0])
    fixed, free = [v for _label, v in pair.variants(snap)]

    def fit(score, status="stable_valid_pair"):
        r = pair.PairResult(status, checkpoint_cycle="1")
        r.commits = {"A": object(), "B": object()}
        r.quality = {"score": score}
        return r

    assert pair.choose([(fit(1.010), fixed), (fit(1.005), free)])[0] == 0
    k, why = pair.choose([(fit(1.030), fixed), (fit(1.005), free)])
    assert k == 1 and "free centres lowered" in why
    # a stable fit beats an unstable one whatever the scores
    assert pair.choose([(fit(1.001, "valid_pair_not_stable"), fixed), (fit(1.2), free)])[0] == 1
    # without both kinds of centres, the smallest score
    assert pair.choose([(fit(1.02), fixed), (fit(1.01), fixed)])[0] == 1


def test_the_replay_command_repeats_the_snapshot():
    snap = iterative_snapshot(
        a_kw={"start": 8, "factor": 1.5, "name": "big one"}, b_kw={"radii": [6, 12]}, damp=0.7
    )
    line = pair.command_for(snap, "MASK=m0")
    words, shared, own = pair.parse(line.split(None, 1)[1])
    s = pair.settings_from(shared)
    comps = pair.components_from(own)
    assert s.damp == 0.7 and s.cores is False and s.cycles == snap.settings.cycles
    assert comps["A"].schedule() == snap.components["A"].schedule()
    assert comps["B"].radii == [6, 12] and comps["A"].name == "big one"
    assert shared["MASK"] == "m0"


def test_the_pair_tool_in_the_window():
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()

    def settle(seconds=60):
        deadline = QtCore.QDeadlineTimer(int(seconds * 1000))
        while (
            window.busy() or getattr(window, "_pair_worker", None)
        ) and not deadline.hasExpired():
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        app.processEvents()

    window.open_pair_example()
    settle()
    panel = window.pair_panel
    assert "stars" in panel.ticked_layers()
    # the sky: measured, shown, and its layer never offered for fitting
    panel.sky_requested.emit()
    settle()
    assert panel.sky_map.result is not None
    names = [panel.layers.item(k).data(QtCore.Qt.UserRole) for k in range(panel.layers.count())]
    assert "skymask" not in names
    # picking: A, then B, then picking ends
    window.pair_pick("A", "centre")
    window._centre_picked(294.0, 264.0)
    assert window._pick_target == "pair:B:centre"
    window._centre_picked(421.0, 323.0)
    assert window._pick_target == "centre"
    assert abs(panel.cards["B"].x0.value() - 421.0) < 3.5
    panel.cards["A"].set_values({"x0": 294.3, "y0": 263.8, "r1": 120})
    panel.cards["B"].set_values({"x0": 421.18, "y0": 322.97, "r1": 50, "nr": 20})
    # the run, in the worker thread
    panel.run_requested.emit()
    settle(120)
    image = window.viewer.image
    record = window.session.pairs[image.key]
    assert record.result.status == "stable_valid_pair"
    assert window.pair_results.record is record
    drawing = window._pair_drawing()
    assert drawing["A"]["fit"] and drawing["B"]["fit"]
    # the layer ticks hide the pair's isophotes (Isophotes) and everything of the pair (Pair)
    window.layers.set_checked("isophotes", False)
    settle(1)
    assert not window._pair_drawing()["A"].get("fit")
    assert window.viewer.pair_items[("A", "fit")]._path.isEmpty()
    window.layers.set_checked("isophotes", True)
    window.layers.set_checked("pair", False)
    settle(1)
    assert window._pair_drawing() == {"A": {}, "B": {}}
    window.layers.set_checked("pair", True)
    settle(1)
    assert window._pair_drawing()["B"]["fit"]
    # every detected object but A and B, in the mask layer 'objects', ticked for the fit
    window.pair_mask_objects(2.0)
    settle(60)
    objects = window.session.masks["objects"]
    assert objects.role == "fit" and "objects" in panel.ticked_layers()
    assert objects.count() > 0
    for cid in "AB":
        x, y = panel.cards[cid].centre()
        assert not objects.masked[int(y), int(x)]
    window.blink_pair()
    assert len(window.viewer.blink_choice()) == 4
    # the sky: its histogram, the user's decision, the factor and the residual sky
    assert panel.sky_hist.isVisibleTo(panel)
    panel.show_mask.setChecked(True)
    settle()
    assert window.session.masks["skymask"].visible
    panel._use_measured()
    assert panel.sky.value() == pytest.approx(panel._measurement.level, abs=1e-4)
    # the first run had the automatic settings (the default): radii and isophotes resolved
    first = record.job.line
    for word in ("R1=AUTO", "START=AUTO", "NR=AUTO", "TIE=-3", "RMSTAR", "TAIL=measured"):
        assert word in first
    assert any("R1=AUTO ->" in n for n in record.job.snap.notes)
    assert not panel.cards["A"].nr.isVisibleTo(panel)
    # automatic, adjustable: every setting shown at the same values, the same command
    auto = panel.command(1)
    panel.set_mode(1)
    assert panel.cards["A"].nr.isVisibleTo(panel) and panel.cards["A"].nr_auto.isChecked()
    adjustable = panel.command(1)
    for word in auto.split():
        if not word.startswith(("NAME=", "FACTOR=")):
            assert word in adjustable.split(), word
    panel.set_mode(2)  # manual: the radii typed
    assert not panel.cards["A"].r1_auto.isChecked() and "R1=AUTO" not in panel.command(1)
    panel.cards["A"].r1_auto.setChecked(False)
    panel.cards["B"].r1_auto.setChecked(False)
    panel.search.setChecked(False)
    panel.factor.setValue(0.999)
    panel.sky_iter.setValue(2)
    command = panel.command(1)
    assert "SKYFACTOR=0.999" in command and "SKYITER=2" in command
    panel.run_requested.emit()
    settle(180)
    record = window.session.pairs[image.key]
    assert record.result.sky_iterations
    assert panel.factor.value() == pytest.approx(record.job.snap.settings.sky_factor, abs=1e-5)
    assert panel.resid_hist.isVisibleTo(panel)
    assert window.pair_results.sky_table.rowCount() == len(record.result.sky_iterations)
    # growth and block in the panel's command, and the report on the clipboard
    panel.cards["A"].set_values(
        {"r1": 120, "start": 60, "start_auto": False, "growth": 0, "growth_value": 1.5}
    )
    panel.cards["B"].set_values({"growth": 2, "radii": "20, 30, 40"})
    command = panel.command(1)
    assert "START=60 FACTOR=1.5" in command and "RADII=20,30,40" in command
    assert "CORES=0" in command and "CONSECUTIVE=2" in command and "FIXCTR=1" in command
    assert panel.cards["A"].schedule() == [60, 90, 120]
    # the iteration table, and one call's images
    assert panel.iterations.rowCount() == len(record.result.calls)
    window.show_pair_iteration(3)
    settle()
    assert window.viewer.image.name.endswith("model") or "input" in window.viewer.image.name
    window.copy_pair_report()
    assert "## calls" in QtWidgets.QApplication.clipboard().text()
    window.close()


def test_mask_segments_keeps_the_sources_at_several_centres(tmp_path):
    scene = synthetic.pair()
    s = Session()
    s.buffers[1] = Image("pair.fits", scene.data.copy())
    s.execute("sources 1")
    s.execute("mask new objects 1")
    a, b = scene.galaxies["A"], scene.galaxies["B"]
    reply = s.execute(f"mask segments objects X0={a.x:.2f},{b.x:.2f} Y0={a.y:.2f},{b.y:.2f} GROW=1")
    m = s.masks["objects"].masked
    assert "kept unmasked" in reply.text
    for g in (a, b):
        assert not m[int(g.y), int(g.x)]
    star = next(st for st in scene.stars if st[2] > 500)
    assert m[int(star[1]), int(star[0])]
    with pytest.raises(CommandError):
        s.execute("mask segments objects X0=1,2 Y0=3")


def test_automatic_radii_and_isophotes():
    # NR=AUTO: about 30 % apart, within the range
    assert pair.auto_nr(3, 3 * 1.3**10) == 11
    assert pair.auto_nr(3, 5) == pair.NR_RANGE[0]
    assert pair.auto_nr(1, 1e6) == pair.NR_RANGE[1]
    comp = pair.Component("A", 50, 50, 3.0, 120, 0)
    assert pair.nr_for(comp, 120) == pair.auto_nr(3.0, 120)
    # START=AUTO: a third of R1, at most 0.35 of the separation, at least 9 px; no growth
    # when that is not below R1
    assert pair.auto_start(120, 1000) == 40
    assert pair.auto_start(120, 60) == 21
    assert pair.auto_start(10, 100) is None
    # R1=AUTO: where the light away from the companion falls below sigma
    yy, xx = np.mgrid[0:400, 0:400]
    light = 100.0 * np.exp(-np.hypot(xx + 0.5 - 200, yy + 0.5 - 200) / 20.0)  # 1 at r = 92
    masked = np.zeros(light.shape, bool)
    r1 = pair.auto_r1(light, masked, (200, 200), (300, 200), sigma=1.0)
    assert 85 <= r1 <= 95
    assert pair.auto_r1(light, masked, (20, 200), (300, 200), sigma=1e-9) == 18.0  # 0.9 x edge
    own = {
        "A": {"X0": "200", "Y0": "200", "R0": "3", "R1": "AUTO", "START": "AUTO"},
        "B": {"X0": "300", "Y0": "200", "R0": "3", "R1": "40"},
    }
    notes = pair.resolve_auto(own, light, masked, 1.0)
    assert float(own["A"]["R1"]) == r1 and own["A"]["FACTOR"] == "1.5"
    assert float(own["A"]["START"]) == pair.auto_start(r1, 100)
    assert any("R1=AUTO ->" in n for n in notes)
    comps = pair.components_from(own)
    assert comps["A"].schedule()[-1] == r1


def test_the_tail_beyond_r1_follows_a_shallow_profile():
    from scipy.special import gammaincinv

    from astro_vista.fitting import fit_array

    ny, nx, x0, y0, re, n = 300, 300, 150.3, 149.6, 30.0, 2.0
    b = gammaincinv(2 * n, 0.5)
    yy, xx = np.mgrid[0:ny, 0:nx] + 0.5
    t, q = np.radians(20.0), 0.75
    u = (xx - x0) * np.cos(t) + (yy - y0) * np.sin(t)
    v = -(xx - x0) * np.sin(t) + (yy - y0) * np.cos(t)
    a = np.hypot(u, v / q)
    truth = 200.0 * np.exp(-b * ((np.maximum(a, 0.3) / re) ** (1 / n) - 1))
    data = truth + np.random.default_rng(3).normal(0, 0.5, truth.shape)
    fit = fit_array(data, x0, y0, 3, 60, 15, fixctr=1)
    assert fit.model is not None
    beyond = (a > 66) & (a < 130)
    err = {}
    for law in pair.TAILS:
        m = pair.tail_model(fit.model, fit.profile, law)
        err[law] = float(np.median(np.abs(m - truth)[beyond] / truth[beyond]))
    # elliprof's r^1/4 extrapolation is far too bright for n = 2; the Sersic fit follows it
    assert err["elliprof"] > 0.1  # 14 % here
    assert err["sersic"] < 0.01 and err["sersic"] < err["exp"]
    inside = (a > 5) & (a < 55)
    m = pair.tail_model(fit.model, fit.profile, "sersic")
    assert np.array_equal(m[inside], fit.model[inside])  # nothing changes within R1


def test_automatic_isophotes_are_about_30_percent_apart():
    assert pair.auto_nr(3, 150) == 16
    assert pair.auto_nr(3, 6) == 6  # never fewer than 6
    assert pair.auto_nr(3, 5000) == 29


def test_the_measured_tail_follows_the_light_beyond_r1():
    from astro_vista.fitting import fit_array

    ny, nx, x0, y0 = 300, 300, 150.3, 149.6
    yy, xx = np.mgrid[0:ny, 0:nx] + 0.5
    a = np.hypot(xx - x0, (yy - y0) / 0.8)
    # an inner profile that a Sersic tail would continue too far: the halo is cut at 90 px
    truth = 300.0 * np.exp(-((a / 15.0) ** 0.5)) * np.clip((110 - a) / 20.0, 0, 1)
    data = truth + np.random.default_rng(4).normal(0, 0.3, truth.shape)
    fit = fit_array(data, x0, y0, 3, 60, 15, fixctr=1)
    beyond = (a > 66) & (a < 140)
    measured = pair.tail_model(fit.model, fit.profile, "measured", data, np.ones_like(data, bool))
    sersic = pair.tail_model(fit.model, fit.profile, "sersic")
    err_m = float(np.mean(np.abs(measured - truth)[beyond]))
    err_s = float(np.mean(np.abs(sersic - truth)[beyond]))
    assert err_m < 0.5 * err_s and err_m < 0.5
    assert np.all(np.diff(pair.measured_tail(data, np.ones_like(data, bool), a, 60.0, 5.0)[1]) <= 0)


def test_a_halo_reaching_the_edge_stops_the_sky_iterations():
    assert pair.reaches_edge(pair.AUTO_R1_EDGE * 100.0, (100.0, 400.0), (1000, 1000))
    assert not pair.reaches_edge(60.0, (100.0, 400.0), (1000, 1000))


def test_segments_grow_with_their_size():
    from astro_vista.commands import _grow_by_size

    seg = np.zeros((200, 200), int)
    yy, xx = np.mgrid[0:200, 0:200]
    seg[np.hypot(xx - 50, yy - 50) < 20] = 1  # a big source
    seg[np.hypot(xx - 150, yy - 150) < 2] = 2  # a small one
    out = _grow_by_size(seg, [1, 2], 1.0, 0.5)
    assert out[50, 50 + 20 + 9] and not out[50, 50 + 20 + 13]  # 1 + 0.5 x 20 = 11 px
    assert out[150, 150 + 3] and not out[150, 150 + 6]  # 1 + 0.5 x 2 = 2 px


class PureFitter(FakeFitter):
    """A fake elliprof whose answer depends only on its input (safe in threads): A's model
    is the light in its half of the frame, scaled by ``share[cos4x]``."""

    def __init__(self, share=None):
        super().__init__()
        self.share = share or {}
        import threading

        self.lock = threading.Lock()

    def __call__(self, data, x0, y0, r0, r1, nr, **kw):
        with self.lock:
            self.calls.append({"component": "A" if x0 < 50 else "B", "kw": kw})
        model = np.zeros(data.shape)
        if x0 < 50:
            model[:] = 5.0 * self.share.get(kw.get("cos4x"), 0.5)
        profile = fake_profile(x0, y0, r0, r1, nr)
        return ArrayFit(profile, model, 0, "", "", ["fake"], 0.0, "single")


def harmonic_snapshot(**kw):
    snap = iterative_snapshot(**kw)
    image = snap.image.copy()
    image += 5.0  # A's light: all of it is fitted with COS4X=1 only
    return replace_image(snap, image)


def replace_image(snap, image):
    from dataclasses import replace

    return replace(snap, image=image)


def test_parallel_fits_give_the_same_result_and_their_mean():
    snap = iterative_snapshot(try_damp=[1.0, 0.5], try_fix=[1, 0])
    one = pair.run_search(snap, fitter=PureFitter())
    from dataclasses import replace

    many = pair.run_search(
        replace(snap, settings=replace(snap.settings, parallel=4)), fitter=PureFitter()
    )
    assert [r["score"] for r in one.search] == [r["score"] for r in many.search]
    assert [r.get("chosen") for r in one.search] == [r.get("chosen") for r in many.search]
    # every variant gave the same models: all are in the mean, which has no spread
    e = many.ensemble
    assert e["members"] == 4 and e["of"] == 4
    assert np.allclose(e["model"], many.total) and np.allclose(e["spread"], 0)
    assert np.allclose(e["residual"], many.residual) and e["sky"] == pytest.approx(100.0)
    assert "mean of the 4 fits" in pair.summary(many, many.snapshot)
    with pytest.raises(pair.PairError):
        pair.settings_from({"SKY": "1", "PARALLEL": "0"})
    assert pair.settings_from({"SKY": "1", "PARALLEL": "5"}).parallel == 5


def test_each_galaxys_harmonics_are_tried_in_turn():
    snap = harmonic_snapshot(a_kw={"try_cos4x": [0, 1, 2]}, b_kw={"try_cos3x": [0, 2]})
    f = PureFitter(share={1: 1.0, 0: 0.0, 2: 0.5, None: 0.5})
    result = pair.run_search(snap, fitter=f)
    kept = result.snapshot.components
    assert pair.effective_harmonics(kept["A"]) == (2, 1)  # the residual of A's light is gone
    harmonic = [r for r in result.search if r.get("harmonics")]
    # A's two other COS4X values with B as it was, then B's one other COS3X with A's best
    assert [r["variant"] for r in harmonic] == [
        "A COS3X=2 COS4X=0",
        "A COS3X=2 COS4X=1",
        "B COS3X=0 COS4X=2",
    ]
    assert harmonic[1].get("chosen") and "A: COS3X=2 COS4X=1 lowered" in result.search_choice
    b_calls = [c for c in f.calls if c["component"] == "A" and c["kw"].get("cos3x") == 0]
    assert not b_calls  # A's COS3X was never changed
    # a change that does not lower the residual by more than 0.01 keeps the harmonics
    assert pair.effective_harmonics(kept["B"]) == (2, 2)


def test_harmonic_keywords_are_elliprofs_own():
    own = {
        "A": {"X0": "10", "Y0": "10", "R0": "2", "R1": "20", "COS3X": "-3", "COS4X": "1"},
        "B": {"X0": "50", "Y0": "10", "R0": "2", "R1": "20", "TRYCOS4X": "0,1,2"},
    }
    comps = pair.components_from(own)
    assert comps["A"].harmonics() == (-3, 1) and comps["B"].try_cos4x == [0, 1, 2]
    assert pair.effective_harmonics(comps["B"]) == (2, 2)  # elliprof's default
    for bad in ({"COS3X": "3"}, {"COS4X": "-1"}, {"TRYCOS4X": "0,5"}):
        with pytest.raises(pair.PairError):
            pair.components_from({"A": {**own["A"], "COS3X": "0", **bad}, "B": own["B"]})
    with pytest.raises(pair.PairError):
        pair.components_from({"A": own["A"], "B": dict(own["B"], MODE="4")})


def test_the_panel_writes_harmonics_and_parallel_fits():
    from PySide6 import QtWidgets

    from astro_vista.pairpanel import PairPanel

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = PairPanel()
    panel.set_mode(2)
    card = panel.cards["A"]
    assert "COS3X" not in card.keywords()  # elliprof's default: nothing written
    card.cos4x.setCurrentIndex(card.cos4x.findData(1))
    card.try_cos3x.setText("0, 1")
    words = card.keywords().split()
    assert "COS4X=1" in words and "TRYCOS3X=0,1" in words
    panel.parallel.setValue(4)
    assert "PARALLEL=4" in panel.command(1).split()
    panel.set_mode(0)
    assert "PARALLEL=4" in panel.command(1).split()
    card.set_values({"mode": "4m"})  # an older named mode: its COS3X and COS4X
    assert (card.cos3x.currentData(), card.cos4x.currentData()) == (0, 1)
    # parallel fits' calls interleave in the table: each row says which variant made it
    panel.add_iteration(dict(cycle="1", component="A", r1=20.0, passed=True, variant_number=3))
    assert panel.iterations.item(panel.iterations.rowCount() - 1, 0).text() == "variant 3: 1"
    app.processEvents()


def test_repeat_runs_change_r0_and_r1_and_are_averaged():
    from dataclasses import replace

    snap = iterative_snapshot()
    snap = replace(snap, settings=replace(snap.settings, runs=4, jitter=0.15, parallel=2))
    result = pair.run_search(snap, fitter=PureFitter())
    rows = result.repeats
    assert len(rows) == 4 and rows[0]["variant"] == "the fit kept"
    labels = [r["variant"] for r in rows[1:]]
    for label in labels:
        r0 = float(label.split("A R0 ")[1].split()[0])
        r1 = float(label.split(", ")[0].split(" R1 ")[1])
        assert 0.1 - 1e-6 <= abs(r0 / 2 - 1) <= 0.15 + 1e-6
        assert 0.1 - 1e-6 <= abs(r1 / 20 - 1) <= 0.15 + 1e-6
    e = result.ensemble
    assert e["repeats"] and e["members"] == 4
    assert np.allclose(e["A"] + e["B"], e["model"])
    assert "mean of 4 runs of 4" in pair.summary(result, result.snapshot)
    # the same seed, the same changes
    again = pair.run_search(snap, fitter=PureFitter())
    assert [r["variant"] for r in again.repeats] == [r["variant"] for r in rows]
    # with a search, its rows are no longer marked as in the mean: the repeats' mean replaced it
    searched = replace(snap, settings=replace(snap.settings, runs=3, try_fix=[0, 1]))
    assert any(r.get("mean") for r in pair._search(searched, None, None, None, PureFitter()).search)
    rows_ = pair.run_search(searched, fitter=PureFitter()).search
    assert len(rows_) == 2 and not any(r.get("mean") for r in rows_)
    # an R0 close to R1: a repeat's shrunk R1 never falls to R0 or below it
    close = replace(
        snap, components={**snap.components, "A": replace(snap.components["A"], r0=18.0)}
    )
    settings = replace(snap.settings, runs=30)
    for _label, sn in pair.jittered(close, settings):
        for c in sn.components.values():
            assert 0 < c.r0 < c.r1
    with pytest.raises(pair.PairError):
        pair.settings_from({"SKY": "1", "RUNS": "0"})
    with pytest.raises(pair.PairError):
        pair.settings_from({"SKY": "1", "HARMONICS": "yes"})


def test_every_harmonic_is_searched_with_harmonics_auto():
    from dataclasses import replace

    snap = harmonic_snapshot()
    snap = replace(snap, settings=replace(snap.settings, harmonic_search=True, parallel=4))
    f = PureFitter(share={1: 1.0, 0: 0.0, 2: 0.5, None: 0.5})
    result = pair.run_search(snap, fitter=f)
    tried = [r for r in result.search if r.get("harmonics") == "A"]
    assert len(tried) == len(pair.COS3X_VALUES) * len(pair.COS4X_VALUES) - 1
    assert pair.effective_harmonics(result.snapshot.components["A"])[1] == 1


def test_a_failed_innermost_isophote_starts_further_out():
    class InnerFitter(SteadyFitter):
        def __call__(self, data, x0, y0, r0, r1, nr, **kw):
            fit = super().__call__(data, x0, y0, r0, r1, nr, **kw)
            if x0 < 50 and r0 < 3:  # A's first isophotes cross while R0 is 2
                fit.profile.loc[1, "Rmaj"] = fit.profile.loc[0, "Rmaj"]
                fit.profile.loc[0, "ellip"] = 0.99
            return fit

    snap = iterative_snapshot()
    result = pair.run(snap, fitter=InnerFitter())
    assert result.complete and result.r0_raised["A"] == pytest.approx(3.0)
    assert "fitted from R0 3" in pair.summary(result, snap)


def test_a_mask_over_a_galaxys_centre_is_lifted_there():
    from astro_vista.commands import _uncover_centres

    masked = np.zeros((60, 120), bool)
    masked[20:40, 80:100] = True  # over B
    masked[5:10, 5:10] = True  # elsewhere
    notes = _uncover_centres(masked, {"A": {"X0": "30", "Y0": "30"}, "B": {"X0": "90", "Y0": "30"}})
    assert not masked[20:40, 80:100].any() and masked[5:10, 5:10].all()
    assert len(notes) == 1 and "B's centre" in notes[0]


def test_wings_beyond_a_mask_are_masked_and_specks_are_not():
    from dataclasses import replace

    snap = iterative_snapshot()
    rng = np.random.default_rng(3)
    residual = rng.normal(0, 1, snap.image.shape)
    yy, xx = np.mgrid[0:60, 0:120]
    star = np.hypot(xx - 60, yy - 50) < 4  # a masked star, with wings out to 9 px
    residual += 6 * np.exp(-np.hypot(xx - 60, yy - 50) / 4)
    residual[10, 100] += 30  # a speck, joined to no mask
    fit_mask = snap.fit_mask | star
    snap = replace(snap, fit_mask=fit_mask)
    first = pair.PairResult("stable_valid_pair", residual=residual)
    first.commits = {"A": object(), "B": object()}
    old = pair.WING_SEED
    pair.WING_SEED = 20
    try:
        out = pair.mask_wings(snap, first)
    finally:
        pair.WING_SEED = old
    added = out.fit_mask & ~fit_mask
    assert added.sum() > 50 and added[50, 66] and not added[10, 100]
    assert "WINGS:" in out.notes[-1]


# ---------------------------------------------------------------- Joe's mode, the pull check
class PullingFitter(SteadyFitter):
    """A's isophotes beyond ``start`` px move towards B when its centre is free."""

    def __init__(self, start=12.0, by=2.0, **kw):
        super().__init__(**kw)
        self.start, self.by = start, by

    def __call__(self, data, x0, y0, r0, r1, nr, **kw):
        fit = super().__call__(data, x0, y0, r0, r1, nr, **kw)
        if x0 < 50 and kw.get("fixctr") == 0:
            outer = fit.profile["Rmaj"] > self.start
            fit.profile.loc[outer, "x0"] = fit.profile.loc[outer, "x0"] + self.by
        return fit


def test_outer_pull_finds_where_the_ellipses_move_towards_the_other():
    import pandas as pd

    r = np.array([3, 5, 7, 10, 13, 17, 22, 28, 35.0])
    x = np.full(r.size, 30.0)
    x[r > 15] += np.array([0.2, 1.5, 2.5, 3.0])[: (r > 15).sum()]
    p = pd.DataFrame({"Rmaj": r, "x0": x, "y0": np.full(r.size, 30.0)})
    out = pair.outer_pull(p, (30.0, 30.0), (90.0, 30.0))
    assert out["from"] == 22 and out["r_ok"] == 17 and out["most"] == pytest.approx(3.0)
    # away from the other galaxy is not a pull
    away = pair.outer_pull(p, (30.0, 30.0), (-30.0, 30.0))
    assert away["from"] is None and away["r_ok"] == 35


def test_joes_mode_grows_each_galaxy_only_as_far_as_it_is_not_pulled():
    from dataclasses import replace

    snap = iterative_snapshot()
    snap = replace(
        snap, components={**snap.components, "A": replace(snap.components["A"], r1=30.0)}
    )
    snap = replace(snap, settings=replace(snap.settings, method="joe", cycles=8))
    f = PullingFitter(start=12.0)
    result = pair.run_search(snap, fitter=f)
    assert result.complete and result.joe["order"] == ["B", "A"]  # the smaller first
    assert result.commits["B"].r1_call == pytest.approx(20.0)  # B is never pulled: its R1
    assert result.commits["A"].r1_call <= 13.5  # A stops where its ellipses start to move (of 30)
    assert any("move towards B" in x for x in result.joe["steps"])
    # each galaxy's centre is left out while the other is fitted
    core_a = result.snapshot.cores["A"]
    assert core_a.shape == "circle" and core_a.a == pytest.approx(12.0)  # 0.2 of 60 px
    b_calls = [c for c in f.calls if c["component"] == "B"]
    assert all(c["kw"]["masked"][30, 30] for c in b_calls)
    # the models kept are fitted with the galaxy's own FIXCTR, the pull judged with free centres
    assert {c["kw"]["fixctr"] for c in f.calls} == {0, 1}
    text = pair.summary(result, result.snapshot)
    assert "Joe's mode: B first, then A" in text and "pull check of A" in text
    with pytest.raises(pair.PairError):
        pair.settings_from({"SKY": "1", "METHOD": "fast"})
    assert pair.settings_from({"SKY": "1", "METHOD": "joe"}).method == "joe"


def test_joes_mode_starts_with_the_other_galaxy_when_the_first_cannot_start():
    from dataclasses import replace

    snap = iterative_snapshot()
    snap = replace(
        snap, components={**snap.components, "A": replace(snap.components["A"], r1=30.0)}
    )
    snap = replace(snap, settings=replace(snap.settings, method="joe", cycles=2))
    f = SteadyFitter(fail={("B", k) for k in range(1, 13)})  # B's first fits all fail
    result = pair.run_joe(snap, fitter=f)
    assert result.joe["order"][0] == "A"
    assert any("could not be started" in x for x in result.joe["steps"])


def test_the_pull_check_reports_each_galaxy_and_changes_nothing():
    snap = iterative_snapshot()
    result = pair.run_search(snap, fitter=PullingFitter(start=12.0))
    assert set(result.pull) == {"A", "B"}
    assert result.pull["A"]["judged"] and result.pull["A"]["from"] > 12
    assert result.pull["B"]["from"] is None
    words = pair.pull_words("A", result.pull["A"], {"A": "A", "B": "B"})
    assert "move towards B from Rmaj" in words
    kept = pair.run_search(snap, fitter=PullingFitter(start=12.0))
    assert np.array_equal(kept.total, result.total)  # diagnostics only
    from dataclasses import replace

    off = replace(snap, settings=replace(snap.settings, pull_check=False))
    assert pair.run_search(off, fitter=PullingFitter()).pull == {}


def test_the_panel_writes_joes_mode():
    from PySide6 import QtWidgets

    from astro_vista.pairpanel import JOE, PairPanel

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = PairPanel()
    panel.set_mode(JOE)
    words = panel.command(1).split()
    assert "METHOD=JOE" in words and "TAIL=measured" in words and "TRYDAMP=1,0.5" not in words
    assert panel.joe_note.isVisibleTo(panel) and not panel.auto_note.isVisibleTo(panel)
    app.processEvents()


def test_joes_mode_settings_choose_the_order_growth_core_and_pull():
    from dataclasses import replace

    snap = iterative_snapshot()
    snap = replace(
        snap, components={**snap.components, "A": replace(snap.components["A"], r1=30.0)}
    )
    s = pair.settings_from(
        {
            "SKY": "100",
            "METHOD": "JOE",
            "FIRST": "A",
            "JOEGROW": "1.5",
            "JOECORE": "0",
            "PULLPX": "3",
            "PULLFRAC": "0",
            "PULLRUN": "1",
            "PULLR0": "0",
            "CYCLES": "6",
        }
    )
    snap = replace(snap, settings=replace(s, sky=100.0))
    f = PullingFitter(start=12.0, by=2.0)  # a 2 px pull: below PULLPX=3, so never a pull
    result = pair.run_search(snap, fitter=f)
    assert result.joe["order"] == ["A", "B"]  # FIRST=A, though A is the larger
    assert result.snapshot.cores["A"].shape == "none"  # JOECORE=0: no centre left out
    assert not any(c["kw"]["masked"].any() for c in f.calls)
    assert result.commits["A"].r1_call == pytest.approx(30.0)  # not pulled: it reached its R1
    a_calls = sorted({round(c["r1"], 2) for c in f.calls if c["component"] == "A"})
    assert 15.0 in a_calls  # START=AUTO 10 x JOEGROW 1.5
    assert result.pull["A"]["px"] == 3 and result.pull["A"]["from"] is None
    assert "R1 x 1.5 each cycle" in pair.summary(result, result.snapshot)
    for bad in (
        {"FIRST": "C"},
        {"JOEGROW": "1"},
        {"JOECORE": "2"},
        {"PULLRUN": "0"},
        {"JOECOREMIN": "40", "JOECOREMAX": "30"},
    ):
        with pytest.raises(pair.PairError):
            pair.settings_from({"SKY": "1", **bad})


def test_the_panel_writes_joes_settings():
    from PySide6 import QtWidgets

    from astro_vista.pairpanel import JOE, JOE_RUN, PairPanel

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = PairPanel()
    panel.set_mode(JOE)
    assert panel.joe_box.isVisibleTo(panel)
    assert panel.joe_keywords() == JOE_RUN  # Joe's defaults: nothing more written
    panel.joe_first.setCurrentIndex(panel.joe_first.findData("b"))
    panel.joe_grow.setValue(1.5)
    panel.pull_frac.setValue(5)
    panel.pull_run.setValue(3)
    panel.joe_fixctr.setCurrentIndex(panel.joe_fixctr.findData(0))
    panel.joe_tail.setCurrentIndex(panel.joe_tail.findData("cut"))
    panel.pull_check.setChecked(False)
    words = panel.command(1).split()
    for w in ("FIRST=B", "JOEGROW=1.5", "PULLFRAC=0.05", "PULLRUN=3", "TAIL=cut", "PULLCHECK=0"):
        assert w in words, w
    assert "FIXCTR=0" in words and "FIXCTR=1" not in words
    panel.joe_defaults()
    assert panel.joe_keywords() == JOE_RUN
    panel.set_mode(0)
    assert not panel.joe_box.isVisibleTo(panel)
    app.processEvents()


def test_joes_mode_uses_each_galaxys_r0_nr_first_r1_and_harmonics():
    from dataclasses import replace

    snap = iterative_snapshot()
    comps = {
        "A": replace(snap.components["A"], r0=4.0, r1=30.0, nr=8, cos4x=1, start=14.0),
        "B": replace(snap.components["B"], cos3x=0),
    }
    snap = replace(snap, components=comps, settings=replace(snap.settings, method="joe"))
    f = SteadyFitter()
    pair.run_joe(snap, fitter=f)
    a_calls = [c for c in f.calls if c["component"] == "A"]
    model_calls = [c for c in a_calls if c["kw"]["fixctr"] == 1]
    assert model_calls and all(c["kw"]["cos4x"] == 1 for c in a_calls)
    assert min(c["r1"] for c in a_calls) == pytest.approx(14.0)  # START: its first R1
    assert all(c["kw"]["cos3x"] == 0 for c in f.calls if c["component"] == "B")
    # NR=8 (not automatic) and R0=4 reach elliprof
    assert all(pair.nr_for(comps["A"], c["r1"]) == 8 for c in a_calls)


def test_the_panel_shows_and_writes_each_galaxys_settings_in_joes_mode():
    from PySide6 import QtWidgets

    from astro_vista.pairpanel import JOE, PairPanel

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = PairPanel()
    panel.set_mode(JOE)
    card = panel.cards["A"]
    for key in ("r0", "start", "nr", "harmonics"):
        assert card.form.isRowVisible(card._row_of[key]), key
    assert not card.try_cos3x.isVisibleTo(card)  # no search in Joe's mode
    card.r0.setValue(5)
    card.r1_auto.setChecked(False)
    card.r1.setValue(150)
    card.start_auto.setChecked(False)
    card.start.setValue(40)
    card.nr_auto.setChecked(False)
    card.nr.setValue(18)
    card.cos4x.setCurrentIndex(card.cos4x.findData(0))
    a = panel.command(1).split(" A: ")[1].split(" B: ")[0].split()
    for w in ("R0=5", "R1=150", "START=40", "NR=18", "COS4X=0", "FIXCTR=1"):
        assert w in a, w
    panel.set_mode(0)  # Automatic: those rows hidden again, the values to try back
    assert not card.form.isRowVisible(card._row_of["nr"]) and not card.try_cos3x.isHidden()
    app.processEvents()
