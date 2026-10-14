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


def test_strongly_correlated_noise_is_no_usable_sky():
    # at 0.5 sigma, noise correlated over a few pixels makes blobs as large as sources:
    # nearly everything is masked, and the tool says so instead of giving a level
    r = sky.measure(synthetic.blank(correlated=2.0, n_stars=10).data, "c")
    assert r.state != "usable_support" and r.reasons


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
    assert all(f"SKYFACTOR={kept['factor']:.10g}" in x for x in pair_lines)


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
    window.blink_pair()
    assert len(window.viewer.blink_choice()) == 4
    # the sky: its histogram, the user's decision, the factor and the residual sky
    assert panel.sky_hist.isVisibleTo(panel)
    panel.show_mask.setChecked(True)
    settle()
    assert window.session.masks["skymask"].visible
    panel._use_measured()
    assert panel.sky.value() == pytest.approx(panel._measurement.level, abs=1e-4)
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
    panel.cards["A"].set_values({"start": 60, "growth": 0, "growth_value": 1.5})
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
