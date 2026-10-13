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
