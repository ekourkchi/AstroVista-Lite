"""Tests of the demo: a synthetic galaxy fitted with ELLIPROF and detected with SExtractor."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest


from astro_vista.commands import CommandError, Session
from astro_vista.fitting import FitSettings, fit
from astro_vista.images import Image, read_fits, write_fits


def galaxy(nx=301, ny=281, x0=150.0, y0=140.0, ellip=0.3, angle=30.0, stars=True):
    """A Sersic-like elliptical galaxy on a flat sky, with noise and a few stars.  The centre
    is in the viewer's coordinates (the first pixel spans 0 to 1)."""
    rng = np.random.default_rng(1)
    jj, ii = np.mgrid[0:ny, 0:nx]
    dx, dy = ii + 0.5 - x0, jj + 0.5 - y0
    t = np.radians(angle)
    u = dx * np.cos(t) + dy * np.sin(t)
    v = (-dx * np.sin(t) + dy * np.cos(t)) / (1 - ellip)
    r = np.hypot(u, v)
    data = 1000.0 * np.exp(-7.67 * ((r / 40.0) ** 0.25 - 1)) + 100.0
    if stars:
        for sx, sy in ((40, 50), (250, 230), (60, 240), (260, 60)):
            data += 800.0 * np.exp(-((ii + 0.5 - sx) ** 2 + (jj + 0.5 - sy) ** 2) / (2 * 1.5**2))
    data += rng.normal(0, 2.0, data.shape)
    return Image("galaxy.fits", data.astype(np.float32))


def test_fits_round_trip(tmp_path):
    image = galaxy(stars=False)
    path = write_fits(image, tmp_path / "g.fits")
    back = read_fits(path)
    assert back.shape == image.shape
    assert np.array_equal(back.data, image.data)


def test_header_only_primary_is_skipped(tmp_path):
    from astropy.io import fits

    data = np.ones((20, 30), np.float32)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, name="SCI")]).writeto(tmp_path / "x.fits")
    image = read_fits(tmp_path / "x.fits")
    assert image.shape == (20, 30)
    assert image.name == "x.fits[1]"


def test_fit_recovers_the_galaxy():
    image = galaxy()
    result = fit(image, FitSettings(x0=150, y0=140, r0=4, r1=80, nr=12, niter=10, sky=100.0))
    p = result.profile
    assert len(p) == 12
    outer = p[p["Rmaj"] > 15]
    assert np.allclose(outer["x0"], 150.0, atol=0.3)
    assert np.allclose(outer["y0"], 140.0, atol=0.3)
    assert np.allclose(outer["ellip"], 0.3, atol=0.03)
    assert result.model.shape == image.shape
    # the ellipses are drawn where the galaxy is
    x, y, a, b, _t = result.ellipses[-1]
    assert abs(x - 150.0) < 0.5 and abs(y - 140.0) < 0.5
    assert abs(b / a - 0.7) < 0.03


def test_fit_with_a_mask_file(tmp_path):
    from astropy.io import fits

    image = galaxy()
    good = np.ones(image.shape, np.float32)
    good[:20, :] = 0
    fits.PrimaryHDU(good).writeto(tmp_path / "mask.fits")
    s = FitSettings(x0=150, y0=140, r0=4, r1=60, nr=8, sky=100.0, mask_file=tmp_path / "mask.fits")
    result = fit(image, s)
    assert result.masked[:20].all() and not result.masked[20:].any()
    assert np.isnan(result.residual[:20]).all()
    assert "--mask" in result.command


def test_bad_radii_are_refused():
    with pytest.raises(ValueError):
        fit(galaxy(), FitSettings(x0=150, y0=140, r0=50, r1=10))


def session_with_galaxy(tmp_path):
    s = Session()
    s.folder = tmp_path
    write_fits(galaxy(), tmp_path / "galaxy.fits")
    s.execute("rd 1 galaxy.fits")
    return s


def test_commands_buffers_and_arithmetic(tmp_path):
    s = session_with_galaxy(tmp_path)
    assert s.buffers[1].shape == (281, 301)
    s.execute("copy 1 2")
    s.execute("[3] = [1] - [2]  # zero")
    assert np.all(s.buffers[3].data == 0)
    s.execute("[4] = log10(abs([1]) + 1) * 2")
    assert np.allclose(s.buffers[4].data, 2 * np.log10(np.abs(s.buffers[1].data) + 1), rtol=1e-5)
    with pytest.raises(CommandError):
        s.execute("[5] = __import__('os')")
    s.execute("wd 3 zero.fits")
    assert (tmp_path / "zero.fits").exists()
    s.execute("close 3")
    assert 3 not in s.buffers
    assert "galaxy.fits" in s.execute("buffers").text
    with pytest.raises(CommandError, match="unknown command"):
        s.execute("frobnicate 1")
    s.execute("close all")
    assert not s.buffers


def test_rd_extension_by_name(tmp_path):
    from astropy.io import fits

    data = np.arange(600, dtype=np.float32).reshape(20, 30)
    hdus = [fits.PrimaryHDU(), fits.ImageHDU(data * 0, name="ERR"), fits.ImageHDU(data, name="SCI")]
    fits.HDUList(hdus).writeto(tmp_path / "x.fits")
    s = Session()
    s.folder = tmp_path
    s.execute("rd 1 x.fits[sci]")
    assert np.array_equal(s.buffers[1].data, data)
    s.execute("rd 2 x.fits[1]")
    assert not s.buffers[2].data.any()


def test_mask_commands(tmp_path):
    s = session_with_galaxy(tmp_path)
    s.execute("mask new m 1")
    s.execute("mask circle m 40.5 50.5 3")
    assert s.masks["m"].masked[50, 40] and s.masks["m"].count() == 29
    s.execute("mask circle m 40.5 50.5 3 ERASE")
    assert s.masks["m"].count() == 0
    s.execute("mask box m 0 0 10 5")
    assert s.masks["m"].count() == 50
    s.execute("mask save m m.dmask")
    s.execute("mask load back m.dmask 1")
    assert np.array_equal(s.masks["back"].masked, s.masks["m"].masked)
    s.execute("mask invert back")
    assert s.masks["back"].count() == 281 * 301 - 50
    stats = s.execute("stats 1 MASK=m").values
    assert stats["pixels"] == 281 * 301 - 50
    s.execute("mask delete back")
    assert "back" not in s.masks


def test_elliprof_command_with_mask_layers(tmp_path):
    s = session_with_galaxy(tmp_path)
    s.execute("mask new m 1")
    s.execute("mask box m 0 0 301 20")
    sky = s.execute("sky 1 MASK=m").values["sky"]
    assert 100.0 < sky < 200.0  # the sky plus the galaxy's faint outskirts
    reply = s.execute("elliprof 1 X0=150 Y0=140 R0=4 R1=60 NR=8 SKY=100 MASK=m TOMODEL=2 TORESID=3")
    assert "8 isophotes" in reply.text
    assert s.buffers[2].family == s.buffers[1].key
    assert np.isnan(s.buffers[3].data[:20]).all()
    assert s.fits[s.buffers[1].key].masked[:20].all()


def test_window_fits_and_detects(tmp_path):
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()

    def wait():
        deadline = QtCore.QDeadlineTimer(60000)
        while window.busy() and not deadline.hasExpired():
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)

    path = write_fits(galaxy(), tmp_path / "galaxy.fits")
    window.open_path(path)
    wait()
    image = window.viewer.image
    assert image is not None and window.shown == 1
    window.elliprof.set_values(dict(x0=150, y0=140, r0=4, r1=80, nr=12, sky=100.0))
    window.run_fit()
    wait()
    assert image.key in window.session.fits
    names = [b.name for b in window.session.buffers.values()]
    assert "galaxy.fits model" in names and "galaxy.fits residual" in names
    assert window.profile.table.rowCount() == 12
    assert "› elliprof 1 X0=150 Y0=140" in window.console.output.toPlainText()
    # a circle drawn with the mask tool goes on a new mask layer, ticked for the fit
    window.mode_actions["circle"].trigger()
    window.viewer.circle_drawn.emit(40.5, 50.5, 3.0, False)
    wait()
    assert window.session.masks["mask"].masked[50, 40]
    assert window.elliprof.ticked_layers() == ["mask"]
    window.mode_actions["pan"].trigger()
    if True:  # the Source finder is always there
        window.show_buffer(1)
        window.run_detection()
        wait()
        assert window.sources.current() is not None
        assert window.sources.table.rowCount() >= 5
        # the picker snaps to the source under the click
        window.viewer.picked.emit(151.0, 141.5)
        wait()
        assert (
            abs(window.elliprof.x0.value() - 150) < 1 and abs(window.elliprof.y0.value() - 140) < 1
        )
        # a fit with the sources masked
        window.mask_sources()
        wait()
        assert "sources" in window.elliprof.ticked_layers()
        window.run_fit()
        wait()
        fit_ = window.session.fits[image.key]
        assert fit_.masked[49, 39] and not fit_.masked[139, 149]
    window.close()


def test_mask_shapes_edits_and_undo(tmp_path):
    s = session_with_galaxy(tmp_path)
    s.execute("mask new m 1")
    s.execute("mask ellipse m 100 100 10 5 0")
    m = s.masks["m"].masked
    assert m[99, 109] and not m[99, 111] and m[104, 99] and not m[106, 99]
    s.execute("mask polygon m 200 200 220 200 210 220")
    assert s.masks["m"].masked[205, 209] and not s.masks["m"].masked[219, 201]
    s.execute("mask brush m 2 10 250 30 250")
    assert s.masks["m"].masked[249, 9:30].all()
    before = s.masks["m"].count()
    s.execute("mask unpatch m 100 100")  # the ellipse goes, the rest stays
    assert not s.masks["m"].masked[99, 99] and s.masks["m"].masked[205, 209]
    s.execute("undo")
    assert s.masks["m"].count() == before
    s.execute("redo")
    assert not s.masks["m"].masked[99, 99]
    s.execute("mask grow m 2")
    grown = s.masks["m"].count()
    s.execute("mask shrink m 2")
    assert s.masks["m"].count() < grown
    s.execute("mask new n 1")
    s.execute("mask box n 0 0 50 50")
    s.execute("mask union both m n")
    assert s.masks["both"].count() >= s.masks["n"].count()
    s.execute("mask intersect none m n")
    assert s.masks["none"].count() == 0
    s.execute("mask rename none empty")
    assert "empty" in s.masks and "none" not in s.masks
    s.execute("mask delete empty")
    s.execute("undo")
    assert "empty" in s.masks
    s.execute("mask range n 1 -1e30 1e30")
    assert s.masks["n"].count() == 281 * 301
    s.execute("mask tobuf n new")
    assert s.buffers[max(s.buffers)].data.sum() == 0
    with pytest.raises(CommandError):
        s.execute("mask unpatch m 5 5")


def test_where_conditions():
    from astro_vista.masks import where

    cols = {"MAG_AUTO": np.array([20.0, 23.0, 25.0]), "CLASS_STAR": np.array([0.9, 0.1, 0.95])}
    assert list(where(cols, "MAG_AUTO < 24")) == [True, True, False]
    assert list(where(cols, "MAG_AUTO < 24 and CLASS_STAR > 0.8")) == [True, False, False]
    assert list(where(cols, "MAG_AUTO > 24 or CLASS_STAR > 0.8")) == [True, False, True]
    with pytest.raises(ValueError):
        where(cols, "FOO < 1")


def test_scales():
    from astro_vista import scales

    data = np.linspace(0, 100, 9999).reshape(101, 99)
    lo, hi = scales.limits(data, "Min-max")
    assert (lo, hi) == (0.0, 100.0)
    t = np.linspace(0, 1, 11)
    for name in scales.STRETCHES:
        out = scales.stretch(t, name)
        assert out[0] == pytest.approx(0, abs=1e-6) and out[-1] == pytest.approx(1, abs=1e-3)
        assert np.all(np.diff(out) >= -1e-9), name  # every stretch keeps the order
    table = scales.colour_table("Heat", contrast=2.0, bias=0.5)
    assert table.shape == (256, 3) and table.dtype == np.uint8
    assert tuple(table[0]) == (0, 0, 0) and tuple(table[-1]) == (255, 255, 255)
    ticks = scales.colour_bar_ticks(0, 100, "Log", 1000)
    assert ticks[0][1] == pytest.approx(0) and ticks[-1][1] == pytest.approx(100)


def test_shell_commands(tmp_path):
    s = Session()
    s.folder = tmp_path
    (tmp_path / "a.txt").write_text("hello\n")
    assert s.is_shell("ls -l") and s.is_shell("!rd") and not s.is_shell("rd 1 x.fits")
    assert "a.txt" in s.execute("ls").text
    assert "a.txt" in s.execute("ll").text
    assert s.execute("!echo $((2 + 3))").text == "5"
    assert s.execute("pwd").text == str(tmp_path)
    assert "(exit 1)" in s.execute("false").text
    with pytest.raises(CommandError, match="nor a program"):
        s.execute("no_such_program_here_xyz")


def test_any_file_name_and_bitmaps(tmp_path):
    from astro_vista.app import EXAMPLE_FILES, example_folder

    image = example_folder() / EXAMPLE_FILES[0]
    mask = example_folder() / EXAMPLE_FILES[1]
    renamed = tmp_path / "galaxy.image"
    renamed.write_bytes(image.read_bytes())  # gzip-compressed FITS with another suffix
    s = Session()
    s.folder = tmp_path
    s.execute(f"rd 1 {renamed}")
    assert s.buffers[1].shape == (1022, 1025)
    s.execute(f"rd 2 {mask}")  # a legacy bitmap as an image: 1 = good
    assert set(np.unique(s.buffers[2].data)) == {0.0, 1.0}
    s.execute(f"mask load m {mask} 1")
    assert s.masks["m"].count() == 103402
    (tmp_path / "notes.txt").write_text("not FITS")
    with pytest.raises(CommandError, match="not a FITS file"):
        s.execute("rd 3 notes.txt")


def test_window_example_picking_and_terminal(tmp_path):
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()

    def wait():
        deadline = QtCore.QDeadlineTimer(60000)
        while window.busy() and not deadline.hasExpired():
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)

    window.open_example()
    wait()
    assert window.shown == 1 and "u12517j" in window.session.masks
    assert window.elliprof.ticked_layers() == ["u12517j"]
    # R1 picked on the image, 100 pixels from the centre
    window.pick_radius("r1")
    window.viewer.picked.emit(567.0 + 60.0, 562.0 + 80.0)
    assert window.elliprof.r1.value() == pytest.approx(100.0)
    assert window.viewer.mode == "pan"
    # the crosshair tool puts a crosshair on the image and ticks its layer
    window.set_mode("crosshair")
    assert window.viewer.crosshair_at is not None and window.layers.checked("crosshair")
    # a Ctrl-click removes the masked patch under the cursor
    m = window.session.masks["u12517j"].masked
    j, i = np.argwhere(m)[len(np.argwhere(m)) // 2]
    before = m.sum()
    window.viewer.unpatch.emit(i + 0.5, j + 0.5)
    wait()
    assert window.session.masks["u12517j"].masked.sum() < before
    # the Terminal button minimises and opens the terminal
    window.show()
    window.toggle_terminal()
    assert window.docks["Terminal"].folded
    window.toggle_terminal()
    assert not window.docks["Terminal"].folded
    window.run_command("echo hi")
    wait()
    assert window.console.output.toPlainText().rstrip().endswith("hi")
    window.close()


def test_file_names():
    from astro_vista.app import file_stem

    assert file_stem("u12517j.fits.gz residual") == "u12517j_residual"
    assert file_stem("u12517j.fits") == "u12517j"
    assert file_stem("x.fits[2]") == "x_2"


def test_crosshair_follows_the_mouse(tmp_path):
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    window.open_example()
    deadline = QtCore.QDeadlineTimer(30000)
    while window.busy() and not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    window.set_mode("crosshair")
    v = window.viewer
    v.fit_view()
    app.processEvents()
    scene_point = v.group.mapToScene(QtCore.QPointF(300.5, 400.5))
    v._mouse_moved(scene_point)
    assert v.crosshair_at == pytest.approx((300.5, 400.5), abs=1e-6)
    v._mouse_moved(v.group.mapToScene(QtCore.QPointF(10.5, 20.5)))  # a click does not stop it
    assert v.crosshair_at == pytest.approx((10.5, 20.5), abs=1e-6)
    window.set_mode("pan")  # another tool: the crosshair stays where it was
    v._mouse_moved(v.group.mapToScene(QtCore.QPointF(500.5, 500.5)))
    assert v.crosshair_at == pytest.approx((10.5, 20.5), abs=1e-6)
    window.close()


def test_panels_fold_and_the_layout_is_remembered():
    from PySide6 import QtWidgets

    from astro_vista import guide
    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    tools = window.docks["Tools"]
    tools.set_folded(True)
    app.processEvents()
    assert tools.folded and not tools.isVisible()  # folded out of sight
    tools.set_folded(False)
    app.processEvents()
    tools.toggle_float()  # popped out, for another monitor
    app.processEvents()
    assert tools.isFloating() and not tools.folded
    tools.toggle_float()  # and back in its place
    app.processEvents()
    assert not tools.isFloating()
    tools.set_folded(True)
    app.processEvents()
    # the command line: inside the terminal while it is open, in the bottom bar when folded
    results = window.docks["Terminal"]
    row = window.console.input_row
    assert row.parent() is window.console.surface
    results.set_folded(True)
    app.processEvents()
    assert results.folded and not window.console.isVisible()  # out of sight when minimised
    assert row.parent() is not window.console.surface and row.isVisible()  # in the bottom bar
    window.toggle_terminal()  # the Terminal button opens it again
    app.processEvents()
    assert (
        not results.folded and window.console.isVisible() and row.parent() is window.console.surface
    )
    results.set_folded(True)
    results.toggle_float()  # popped out: the command line goes with the terminal
    app.processEvents()
    assert results.isFloating() and row.parent() is window.console.surface
    results.toggle_float()
    app.processEvents()
    assert not results.isFloating()
    assert window.finder.command(1) == "sources 1"
    window.close()  # saves the layout
    again = MainWindow()
    again.show()
    for _ in range(5):
        app.processEvents()
    # every panel opens unfolded, whatever was folded when the window closed
    assert all(not d.folded and d.isVisible() for d in again.docks.values())
    assert again.docks["Tools"].widget().isVisible()
    again.reset_layout()
    again.close()
    # every guide topic opens
    for key in guide.TOPICS:
        guide.show(key)
        assert guide._window.text.toPlainText().strip()
    guide._window.close()


def test_the_example_session_runs_from_anywhere(tmp_path):
    from pathlib import Path

    session = Path(__file__).resolve().parent.parent / "examples" / "u12517" / "session.txt"
    s = Session()
    s.folder = tmp_path  # not the example's folder
    reply = s.execute(f"run {session}")
    assert "23 isophotes" in reply.text
    assert s.masks["u12517j"].count() == 103402 and s.masks["sources"].count() > 0
    assert s.folder == tmp_path


def test_compass_is_square_after_north_up():
    from PySide6 import QtWidgets

    from astro_vista.app import EXAMPLE_FILES, example_folder
    from astro_vista.viewer import Viewer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    v = Viewer()
    v.resize(800, 600)
    v.show()
    v.set_image(read_fits(example_folder() / EXAMPLE_FILES[0]))
    assert v.north_up()
    v.show_compass(True)
    app.processEvents()
    n, e = v.compass.north, v.compass.east
    assert abs(float(np.dot(n, e))) < 1e-9  # a right angle
    assert n[1] < -0.999  # North up (screen y points down)
    assert e[0] < -0.999  # East left
    v.close()


def test_crosshair_stays_on_the_image_and_the_view_pans():
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    window.open_example()
    deadline = QtCore.QDeadlineTimer(30000)
    while window.busy() and not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    window.set_mode("crosshair")
    v = window.viewer
    v.fit_view()
    app.processEvents()
    v._mouse_moved(v.group.mapToScene(QtCore.QPointF(-10.0, 300.0)))  # just left of the image
    assert v.crosshair_at == pytest.approx((0.0, 300.0))
    assert v.view.state["mouseEnabled"] == [True, True]  # dragging pans in this mode
    window.close()


def test_buffer_as_a_mask_in_the_fit(tmp_path):
    s = session_with_galaxy(tmp_path)
    good = np.ones((281, 301), np.float32)
    good[:30] = 0  # masked
    s.buffers[7] = s.buffers[1].derived("my mask", good)
    s.execute("elliprof 1 X0=150 Y0=140 R0=4 R1=60 NR=8 SKY=100 MASKBUF=7")
    assert s.fits[s.buffers[1].key].masked[:30].all()
    assert not s.fits[s.buffers[1].key].masked[30:].any()


def test_window_eraser_blink_colours_and_welcome():
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    assert window.viewer.welcome.isVisible()  # before an image is opened

    def wait():
        deadline = QtCore.QDeadlineTimer(30000)
        while window.busy() and not deadline.hasExpired():
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)

    window.open_example()
    wait()
    app.processEvents()
    assert not window.viewer.welcome.isVisible()
    assert window.viewer._compass_on  # always shown
    assert window.viewer.centre.opts["size"] <= 10  # a small cross
    # the eraser unmasks along its path
    m = window.session.masks["u12517j"]
    j, i = np.argwhere(m.masked)[len(np.argwhere(m.masked)) // 2]
    window.set_mode("eraser")
    window.viewer.brush_drawn.emit([(i + 0.5, j + 0.5)], 3.0, True)
    wait()
    assert not window.session.masks["u12517j"].masked[j, i]
    # the arrow keys blink between buffers
    window.run_command("copy 1 2")
    wait()
    window.show_buffer(1)
    window._blink_step(1)
    assert window.shown == 2
    window._blink_step(1)
    assert window.shown == 1
    # the colour box changes a layer's colour, the checkbox does not
    item = window.layers.overlays["isophotes"]
    assert item.checkState(1) == QtCore.Qt.Checked and not item.icon(0).isNull()
    # buffers of the image's size can mask the fit
    window.show_buffer(1)
    texts = [window.elliprof.layers.item(k).text() for k in range(window.elliprof.layers.count())]
    assert any(t.startswith("buffer 2") for t in texts)
    window.close()


def test_loading_a_mask_into_the_layers_panel():
    from pathlib import Path

    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    window.open_example()

    def wait():
        deadline = QtCore.QDeadlineTimer(30000)
        while window.busy() and not deadline.hasExpired():
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)

    wait()
    dmask = Path(__file__).resolve().parent.parent / "examples" / "u12517" / "u12517j.dmask"
    window.layers.load_file(str(dmask))  # as Load mask… or a drop on the panel does
    wait()
    names = [
        window.layers.mask_root.child(k).text(1)
        for k in range(window.layers.mask_root.childCount())
    ]
    assert len([n for n in names if n.startswith("u12517j")]) == 1  # loaded again: replaced
    assert "u12517j" in window.elliprof.ticked_layers()
    # a mask layer from a buffer is ticked for the fit too
    window._mask_command("mask frombuf frombuf1 1")
    wait()
    assert "frombuf1" in window.session.masks and "frombuf1" in window.elliprof.ticked_layers()
    window.close()


def test_a_loaded_mask_is_a_buffer_too(tmp_path):
    from astro_vista.app import EXAMPLE_FILES, example_folder

    s = Session()
    s.folder = tmp_path
    s.execute(f"rd 1 {example_folder() / EXAMPLE_FILES[0]}")
    reply = s.execute(f"mask load m {example_folder() / EXAMPLE_FILES[1]} 1")
    assert "buffer 2" in reply.text
    assert s.buffers[2].name == "mask m" and (s.buffers[2].data == 0).sum() == 103402
    s.execute("mask circle m 100 100 5")  # the buffer follows the layer
    assert s.buffers[2].data[99, 99] == 0
    s.execute("undo")
    assert (s.buffers[2].data == 0).sum() == 103402
    s.execute("copy 1 2")  # something else in its place: the mask gets a new buffer
    s.execute("mask circle m 100 100 5")
    assert s.buffers[2].name.endswith("copy")
    n, _image = s.mask_buffers["m"]
    assert n != 2 and s.buffers[n].data[99, 99] == 0
    # the fit can use the buffer as a mask
    s.execute(f"elliprof 1 X0=567 Y0=562 R0=9 R1=347 NR=23 SKY=3246 MASKBUF={n}")
    assert s.fits[s.buffers[1].key].masked.sum() >= 103402


def test_tab_completion(tmp_path):
    from astro_vista.commands import completions

    s = session_with_galaxy(tmp_path)
    s.execute("mask new stars 1")
    (tmp_path / "residual.fits").write_text("")
    (tmp_path / "folder").mkdir()
    assert "elliprof" in completions(s, "ell")[1]
    assert completions(s, "mask gr")[1] == ["grow"]
    assert completions(s, "mask grow st")[1] == ["stars"]
    start, found = completions(s, "elliprof 1 MASK=st")
    assert found == ["stars"] and "elliprof 1 MASK=st"[start:] == "st"
    assert "NITER=" in completions(s, "elliprof 1 X0=1 NI")[1]
    assert "DETECT_THRESH=" in completions(s, "sources 1 DET")[1]
    assert completions(s, "wd 1 res")[1] == ["residual.fits"]
    assert completions(s, "cd fol")[1] == ["folder/"]
    assert "1" in completions(s, "tv ")[1]
    assert completions(s, "ls resi")[1] == ["residual.fits"]  # a shell command's argument


def test_contour_tool_and_blink_checkbox():
    from PySide6 import QtCore, QtGui, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    window.open_example()
    deadline = QtCore.QDeadlineTimer(30000)
    while window.busy() and not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    panel = window.contour_panel
    panel.count.setValue(5)
    panel.scale.setCurrentText("Log")
    panel.limit_mode.setCurrentText("99%")
    panel.generate_requested()
    levels = panel.values()
    assert len(levels) == 5 and levels == sorted(levels)
    assert levels[1] - levels[0] < levels[-1] - levels[-2]  # Log: closer at the faint end
    window._apply_contours()
    assert window.layers.checked("contours")
    assert len(window.viewer.contours.getData()[0]) > 0
    window._clear_contours()
    assert (
        window.viewer.contours.getData()[0] is None or len(window.viewer.contours.getData()[0]) == 0
    )
    # the arrow keys blink only while Blink is ticked
    window.run_command("copy 1 7")
    deadline = QtCore.QDeadlineTimer(30000)
    while window.busy() and not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    window.show_buffer(1)
    right = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
    window.viewer._key(right)
    assert window.shown == 1
    window.viewer.blink_box.setChecked(True)
    for action in window.viewer.blink_menu.actions():  # blink between 1 and 7 only
        action.setChecked(action.data() in (1, 7))
    window.viewer._key(right)
    assert window.shown == 7
    window.viewer._key(right)
    assert window.shown == 1
    window.close()


def test_quick_inspector_is_the_first_tool():
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    window.open_example()
    deadline = QtCore.QDeadlineTimer(30000)
    while window.busy() and not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    assert window.side.tabText(0) == "Inspector"
    assert window.tools.tabText(0) == "ELLIPROF"
    window.side.setCurrentIndex(0)
    window._inspect_at = (567.5, 562.5)
    window._inspect()
    f = window.quick.fields
    assert f["pixel"].text() == "568, 563"
    assert float(f["value"].text()) == pytest.approx(
        float(window.viewer.image.data[562, 567]), rel=1e-5
    )
    assert f["ra"].text().startswith("23:19:")
    assert window.quick.magnifier.pixmap() is not None
    window.close()


def test_start_defaults_and_north_up_toggle():
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    assert not window.viewer.histogram_box.isChecked() and not window.viewer.histogram.isVisible()
    assert window.viewer.stretch_box.currentText() == "Log"
    window.open_example()
    deadline = QtCore.QDeadlineTimer(30000)
    while window.busy() and not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    assert not window.a_north.isChecked()
    window.a_north.trigger()  # on
    assert window.a_north.isChecked() and window.viewer.rotation != 0
    window.a_north.trigger()  # off again: the image as it is
    assert not window.a_north.isChecked() and window.viewer.rotation == 0
    # the inspector follows the mouse
    window.side.setCurrentWidget(window.inspector)
    window._show_position(100.5, 200.5)
    window.inspect_timer.timeout.emit()
    assert window.quick.fields["pixel"].text() == "101, 201"
    window.close()


def test_source_finder_finds_the_galaxy_and_the_stars(tmp_path):
    s = session_with_galaxy(tmp_path)
    reply = s.execute("sources 1 TOSEG=5")
    run = s.runs[-1]
    assert "sources" in reply.text and len(run) >= 5
    centre = run.nearest(150.0, 140.0)
    x, y, _a, _b, _t = run.ellipses()[centre]
    assert abs(x - 150.0) < 1.0 and abs(y - 140.0) < 1.0
    for sx, sy in ((40, 50), (250, 230), (60, 240), (260, 60)):
        k = run.nearest(sx, sy)
        assert np.hypot(run.ellipses()[k][0] - sx, run.ellipses()[k][1] - sy) < 1.0
    assert run.columns["KRON_RADIUS"].min() >= 3.5  # as SExtractor reports it
    assert s.buffers[5].shape == s.buffers[1].shape  # the segmentation
    # and its inverse as a mask layer: every source's pixels masked, the sky left
    layer = s.masks[f"seg{run.number}"]
    assert np.array_equal(layer.masked, s.buffers[5].data > 0)
    assert layer.masked[140, 150] and not layer.masked[5, 5]
    s.execute("sources 1 TOSEG=6 SEGMASK=0")
    assert f"seg{s.runs[-1].number}" not in s.masks
    galaxy_row = run.nearest(150.0, 140.0)
    number = int(run.columns["NUMBER"][galaxy_row])
    values = s.execute(f"centre 1 SOURCE={number}").values
    assert abs(values["x0"] - 150.0) < 1.0 and abs(values["y0"] - 140.0) < 1.0
    s.execute("mask new src 1")
    s.execute("mask sources src SCALE=1.5 X0=150 Y0=140")
    m = s.masks["src"].masked
    assert m[49, 39] and m[229, 249] and not m[139, 149]
    s.execute("mask new stars 1")
    s.execute("mask sources stars WHERE='ISOAREA_IMAGE < 200' X0=150 Y0=140")
    assert s.masks["stars"].masked[49, 39] and not s.masks["stars"].masked[139, 149]
    s.execute("mask new seg 1")
    s.execute("mask segments seg GROW=2 X0=150 Y0=140")
    assert s.masks["seg"].masked[49, 39] and not s.masks["seg"].masked[139, 149]
    # the segments replace what the layer masked; ADD adds them
    s.execute("mask fill seg")
    s.execute("mask segments seg X0=150 Y0=140")
    seg = s.runs[-1].segmentation
    assert np.array_equal(s.masks["seg"].masked[seg == 0], np.zeros((seg == 0).sum(), bool))
    s.execute("mask fill seg")
    s.execute("mask segments seg X0=150 Y0=140 ADD")
    assert s.masks["seg"].masked.all()
    with pytest.raises(CommandError, match="unknown setting"):
        s.execute("sources 1 DETECT_THRESHOLD=3")


def test_image_commands(tmp_path):
    s = session_with_galaxy(tmp_path)
    before = s.buffers[1].data.copy()
    s.execute("ac 1 10")
    assert np.allclose(s.buffers[1].data, before + 10)
    s.execute("sc 1 10")
    s.execute("copy 1 2")
    s.execute("si 2 1")
    assert np.allclose(s.buffers[2].data, 0)
    reply = s.execute("abx 1")
    assert "max" in reply.text and "column 149, row 139" in reply.text  # the galaxy's centre
    s.execute("max 1")
    assert (s.variables["maxc"], s.variables["maxr"]) == (149, 139)
    with pytest.raises(CommandError, match="unknown command"):
        s.execute("box 1 NC=3 NR=3")  # no boxes in the demo
    s.execute("set xc=150 yc=140 r=2*30")
    assert s.variables["r"] == 60
    s.execute("elliprof 1 X0=xc Y0=yc R0=4 R1=r NR=8 SKY=100")
    assert s.fits[s.buffers[1].key].profile["Rmaj"].max() == pytest.approx(60, rel=1e-3)
    s.execute("copy 1 4")
    s.execute("shift 4 DX=2 DY=-3 ORDER=1")
    s.execute("max 4")
    assert (s.variables["maxc"], s.variables["maxr"]) == (151, 136)
    s.execute("rotate 4 ANGLE=90")
    s.execute("flip 4 X")
    assert s.buffers[4].shape == (301, 281)  # turned by 90 degrees
    s.execute("bin 4 2")
    assert s.buffers[4].shape == (150, 140)
    s.execute("copy 1 5")
    s.execute("gauss 5 X=40.5 Y=40.5 FWHM=1.5 PEAK=1e6")
    s.execute("zap 5 SIG=5 SIZE=5")
    assert s.buffers[5].data[40, 40] < 1e5  # the hot spot is gone
    s.execute("smooth 5 FWHM=3")
    s.execute("clip 5 MAX=500")
    assert np.nanmax(s.buffers[5].data) <= 500
    s.execute("fft 1 OUT=6")
    assert s.buffers[6].shape == s.buffers[1].shape
    s.execute("mash 1 ROWS OUT=7")
    assert s.buffers[7].shape == (1, 301)
    s.execute("ch 1 OBJECT=galaxy")
    assert "galaxy" in s.execute("chead 1 OBJECT").text
    assert "xc = 150" in s.execute("typ xc").text
    for name in ("log", "sqrt", "round", "inverse", "ln"):
        s.execute("copy 1 8")
        s.execute(f"{name} 8")
    assert "DX=" in s.execute("help shift").text


def test_source_finder_settings(tmp_path):
    s = session_with_galaxy(tmp_path)
    s.execute("sources 1 PHOT_APERTURES=4,8 PHOT_FLUXFRAC=0.9 FILTER_FWHM=2.5 FILTER_TYPE=conv")
    run = s.runs[-1]
    star = run.nearest(40.0, 50.0)
    c = run.columns
    assert {"MAG_APER", "MAG_APER_2", "FLUX_APER_2"} <= set(c)
    assert c["FLUX_APER_2"][star] > c["FLUX_APER"][star]  # the larger aperture holds more
    s.execute("sources 1 PHOT_FLUXFRAC=0.5")
    half = s.runs[-1].columns["FLUX_RADIUS"][s.runs[-1].nearest(40.0, 50.0)]
    assert c["FLUX_RADIUS"][star] > half  # 90% of the light lies further out than 50%
    s.execute("sources 1 THRESH_TYPE=ABSOLUTE DETECT_THRESH=400")
    bright = len(s.runs[-1])
    s.execute("sources 1 THRESH_TYPE=ABSOLUTE DETECT_THRESH=40")
    assert len(s.runs[-1]) >= bright  # a lower threshold finds as many or more
    s.execute("sources 1 BACK_TYPE=MANUAL BACK_VALUE=100")
    assert np.allclose(s.runs[-1].columns["BACKGROUND"], 100)
    s.execute("mask new hide 1")
    s.execute("mask circle hide 40.5 50.5 8")
    s.execute("sources 1 MASK=hide")
    found = s.runs[-1]
    k = found.nearest(40.0, 50.0)
    assert np.hypot(found.ellipses()[k][0] - 40, found.ellipses()[k][1] - 50) > 3  # left out
    with pytest.raises(CommandError, match="PHOT_AUTOPARAMS"):
        s.execute("sources 1 PHOT_AUTOPARAMS=2.5")


def test_mac_app_name():
    import sys

    from astro_vista.app import name_the_mac_app

    name_the_mac_app("AstroVista")  # does nothing elsewhere; never fails
    if sys.platform == "darwin":
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        call = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(send)
        call1 = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        )(send)
        callstr = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p
        )(send)
        utf8 = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)(send)
        key = callstr(
            objc.objc_getClass(b"NSString"),
            objc.sel_registerName(b"stringWithUTF8String:"),
            b"CFBundleName",
        )
        bundle = call(objc.objc_getClass(b"NSBundle"), objc.sel_registerName(b"mainBundle"))
        info = call(bundle, objc.sel_registerName(b"infoDictionary"))
        value = call1(info, objc.sel_registerName(b"objectForKey:"), key)
        assert utf8(value, objc.sel_registerName(b"UTF8String")) == b"AstroVista"


def test_every_question_mark_explains_something():
    from PySide6 import QtWidgets

    from astro_vista import detection, guide
    from astro_vista.app import MainWindow

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    buttons = [b for b in window.findChildren(QtWidgets.QToolButton) if b.text() == "?"]
    assert len(buttons) >= 40  # beside the settings of every tool, and above the image
    for b in buttons[:60]:
        b.click()  # opens the bubble or the guide; must not fail
        QtWidgets.QWhatsThis.hideText()
    if guide._window is not None:
        guide._window.close()
    assert all(key in guide.TIPS for key in detection.DEFAULTS)
    window.close()


def test_arithmetic_like_the_main_astrovista(tmp_path):
    s = session_with_galaxy(tmp_path)
    s.execute("copy 1 2")
    s.execute("ac 2 50")
    s.execute("[3] = [2] - [1]")
    assert np.allclose(s.buffers[3].data, 50, atol=0.5)  # 32-bit floats at the bright centre
    s.execute("B4 = ([1] - 100) ^ 2 / 4")
    assert np.allclose(s.buffers[4].data, (s.buffers[1].data - 100.0) ** 2 / 4, rtol=1e-5)
    s.execute("[5] = median([1], [2], [3])")
    stack = np.stack([s.buffers[k].data for k in (1, 2, 3)])
    assert np.allclose(s.buffers[5].data, np.median(stack, axis=0), rtol=1e-5)
    s.execute("[6] = [1] - median([1])")
    assert abs(np.median(s.buffers[6].data)) < 1e-3
    s.buffers[7] = s.buffers[1].derived("zeros", np.zeros((281, 301)))
    s.execute("[8] = div([1], [7], 0)")
    assert not s.buffers[8].data.any()
    s.execute("[8] = [1] / [7]")
    assert np.isinf(s.buffers[8].data).any() or np.isnan(s.buffers[8].data).any()
    s.execute("mask new m 1")
    s.execute("mask box m 0 0 10 10")
    s.execute("[9] = [1] * m")
    assert (
        np.isnan(s.buffers[9].data[:10, :10]).all()
        and np.isfinite(s.buffers[9].data[20:, 20:]).all()
    )
    s.execute("[10] = good(m) + 2 * masked(m)")
    assert s.buffers[10].data[0, 0] == 2 and s.buffers[10].data[50, 50] == 1
    s.execute("[11] = where([1] > 1000, 1, 0)")
    assert set(np.unique(s.buffers[11].data)) <= {0.0, 1.0}
    s.execute("set k=3")
    s.execute("[12] = [1] * k")
    assert np.allclose(s.buffers[12].data, 3 * s.buffers[1].data, rtol=1e-5)
    s.execute("calc 13 EXPR='[1] + 1'")
    assert np.allclose(s.buffers[13].data, s.buffers[1].data + 1, rtol=1e-6)
    with pytest.raises(CommandError):
        s.execute("[14] = __import__('os').getcwd()")
    with pytest.raises(CommandError, match="unknown function"):
        s.execute("[14] = frobnicate([1])")
    assert "median" in s.execute("help calc").text


def test_same_scale_blinking_and_the_wheel_guard():
    from PySide6 import QtCore, QtGui, QtWidgets

    from astro_vista import guide
    from astro_vista.app import MainWindow, WheelGuard

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    window = MainWindow()
    window.show()
    window.open_example()
    deadline = QtCore.QDeadlineTimer(30000)
    while window.busy() and not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    window.run_command("[2] = [1] * 2")
    deadline = QtCore.QDeadlineTimer(30000)
    while window.busy() and not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    window.show_buffer(1)
    limits = window.viewer._limits
    for action in window.viewer.blink_menu.actions():
        action.setChecked(action.data() in (1, 2))
    window.viewer.same_scale.setChecked(True)
    window._blink_step(1)
    assert window.shown == 2 and window.viewer._limits == limits  # the same scale
    window.viewer.same_scale.setChecked(False)
    window._blink_step(1)
    window._blink_step(1)
    assert window.shown == 2 and window.viewer._limits != limits  # its own scale
    # the wheel over a setting does not change it
    box = window.elliprof.nr
    before = box.value()
    wheel = QtGui.QWheelEvent(
        QtCore.QPointF(5, 5),
        QtCore.QPointF(box.mapToGlobal(QtCore.QPoint(5, 5))),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, 120),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.NoScrollPhase,
        False,
    )
    QtWidgets.QApplication.sendEvent(box, wheel)
    assert box.value() == before
    app.removeEventFilter(guard)
    assert window.windowTitle() == "AstroVista Lite"
    assert list(guide.TOPICS)[:2] == ["overview", "lesson"] and "credits" in guide.TOPICS
    assert "github.com/sep-developers/sep" in guide.TOPICS["credits"][1]
    window.close()


def test_special_values_never_give_infinity(tmp_path):
    import warnings

    from astro_vista.images import Image

    s = Session()
    s.folder = tmp_path
    s.buffers[1] = Image("a", np.array([[1, 2], [np.nan, 4]], np.float32))
    s.buffers[2] = Image("z", np.array([[0, 1], [0, np.nan]], np.float32))
    s.buffers[3] = Image("n", np.full((2, 2), np.nan, np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning may escape
        s.execute("[4] = [1] / [2]")
        s.execute("[5] = mean([1], [2])")
        s.execute("[6] = sum([3], [3])")
        s.execute("[7] = log10([2])")
        s.execute("copy 1 8")
        s.execute("di 8 2")
        s.execute("copy 2 9")
        s.execute("inverse 9")
    for n in (4, 5, 6, 7, 8, 9):
        assert not np.isinf(s.buffers[n].data).any()
    assert np.isnan(s.buffers[4].data[0, 0]) and s.buffers[4].data[0, 1] == 2
    assert s.buffers[5].data.tolist() == [[0.5, 1.5], [0.0, 4.0]]  # NaN left out
    assert np.isnan(s.buffers[6].data).all()  # all NaN: NaN, not 0


def test_tags_and_how_buffers_were_made(tmp_path):
    s = session_with_galaxy(tmp_path)
    s.execute("copy 1 2")
    reply = s.execute('[3] = [1] - [2] tag="difference"')
    assert s.buffers[3].tag == "difference" and s.buffers[3].label == "difference"
    assert s.buffers[3].note == "[3] = [1] - [2]" and "tagged 3" in reply.text
    assert any("[3] = [1] - [2]" in h for h in s.buffers[3].header.get("HISTORY", []))
    s.execute("sources 1 TAG=found TOSEG=4 DETECT_THRESH=4")  # TAG= anywhere on the line
    assert s.buffers[4].tag == "found" and s.buffers[4].note.startswith("sources 1")
    s.execute("elliprof 1 X0=150 Y0=140 R0=4 R1=60 NR=8 SKY=100 TOMODEL=5 TORESID=6 tag=fit")
    assert s.buffers[5].tag == "fit" and s.buffers[6].tag == "fit"
    s.execute('tag 1 "the galaxy"')
    assert s.buffers[1].label == "the galaxy" and s.buffers[1].note == "rd 1 galaxy.fits"
    assert "difference" in s.execute("buffers").text


def test_panels_stay_in_their_place_and_blink_shows_tags():
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtCore import Qt

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    places = {
        "Buffers and layers": Qt.LeftDockWidgetArea,
        "Tools": Qt.RightDockWidgetArea,
        "Terminal": Qt.BottomDockWidgetArea,
    }
    for title, area in places.items():
        assert window.docks[title].allowedAreas() == area
    menus = [a.text().replace("&", "").replace("\u200b", "") for a in window.menuBar().actions()]
    assert menus == ["File", "Edit", "View", "Tools", "Masks", "Help"]  # Edit hides a space
    window.open_example()
    deadline = QtCore.QDeadlineTimer(30000)
    while window.busy() and not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    window.run_command('[3] = [1] * 2 tag="twice"')
    deadline = QtCore.QDeadlineTimer(30000)
    while window.busy() and not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    action = next(a for a in window.viewer.blink_menu.actions() if a.data() == 3)
    assert action.text() == "3  twice" and "[3] = [1] * 2" in action.toolTip()
    window.docks["Tools"].toggle_float()
    window.docks["Terminal"].set_folded(True)
    window.reset_layout()  # View ▸ Reset the layout
    app.processEvents()
    assert not window.docks["Tools"].isFloating() and not window.docks["Terminal"].folded
    window.close()


def test_right_panel_tabs_and_upright_crosshair():
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    assert [window.side.tabText(k) for k in range(window.side.count())] == [
        "Inspector",
        "Tools",
        "Results",
        "Header",
    ]
    assert [window.results.tabText(k) for k in range(window.results.count())] == [
        "Profile",
        "Sources",
        "Plots",
        "Pair",
    ]
    assert window.docks["Terminal"].widget() is window.console  # the terminal alone
    window.open_example()
    deadline = QtCore.QDeadlineTimer(30000)
    while window.busy() and not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    window.north_up()
    window.viewer.set_crosshair(500.0, 400.0)
    assert window.viewer.cross_v.angle == 90 and window.viewer.cross_h.angle == 0  # upright
    p = window.viewer.group.mapToParent(QtCore.QPointF(500.0, 400.0))
    assert window.viewer.cross_v.value() == pytest.approx(p.x())
    assert window.viewer.cross_h.value() == pytest.approx(p.y())
    window.show_result(window.profile)
    assert window.side.currentWidget() is window.results
    window.close()


def test_documentation_search_and_one_plot_area():
    from PySide6 import QtWidgets

    from astro_vista import guide
    from astro_vista.panels import ProfilePanel

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    guide.show("overview")
    window = guide._window
    window.search.setText("blink")
    shown = [
        window.topics.item(k).data(0x0100)
        for k in range(window.topics.count())
        if not window.topics.item(k).isHidden()
    ]
    assert "display" in shown and "credits" not in shown
    window.search.setText("")
    assert all(not window.topics.item(k).isHidden() for k in range(window.topics.count()))
    window.close()
    panel = ProfilePanel()
    assert panel.stack.count() == 2 and panel.stack.currentWidget() is panel.plot
    panel.as_table.setChecked(True)
    assert panel.stack.currentWidget() is panel.table


def test_listing_colours_links_and_fitsinfo(tmp_path):
    from astropy.io import fits

    (tmp_path / "data").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "plot.png").write_bytes(b"x")
    data = np.ones((10, 12), np.float32)
    hdus = [
        fits.PrimaryHDU(),
        fits.ImageHDU(data, name="SCI", ver=1),
        fits.ImageHDU(data * 2, name="SCI", ver=2),
        fits.BinTableHDU.from_columns(
            [fits.Column(name="a", format="E", array=np.arange(3))], name="CAT"
        ),
    ]
    fits.HDUList(hdus).writeto(tmp_path / "frame.fits")
    s = Session()
    s.folder = tmp_path
    reply = s.execute("ls")
    assert (
        f"cd:{tmp_path / 'data'}" in reply.html and f"open:{tmp_path / 'frame.fits'}" in reply.html
    )
    assert "#c678dd" in reply.html and "#e5c07b" in reply.html  # a picture, a text file
    long = s.execute("ls -l")
    assert f"open:{tmp_path / 'frame.fits'}" in long.html
    info = s.execute("fitsinfo frame.fits").text
    assert "[1] SCI,1" in info and "[2] SCI,2" in info and "table, 3 rows" in info
    s.execute("rd 1 frame.fits[sci,2]")
    assert np.all(s.buffers[1].data == 2)
    s.execute("rd 2 frame.fits[SCI]")
    assert np.all(s.buffers[2].data == 1)
    with pytest.raises(CommandError, match="fitsinfo lists them"):
        s.execute("rd 3 frame.fits[err]")


def test_open_recent_and_the_r_quarter_law(tmp_path):
    import pandas as pd
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow
    from astro_vista.fitting import FitResult

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    path = write_fits(galaxy(), tmp_path / "g.fits")
    window.open_path(path)
    deadline = QtCore.QDeadlineTimer(30000)
    while window.busy() and not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    assert window._recent()[0] == str(path)
    window._fill_recent()
    assert any(a.text() == "g.fits" for a in window.recent_menu.actions())
    r = np.geomspace(2, 200, 20)
    i0 = 500 * np.exp(-7.669 * ((r / 40.0) ** 0.25 - 1))
    profile = pd.DataFrame(
        {
            "Rmaj": r,
            "I0": i0,
            "ellip": 0.2,
            "alpha": 0.0,
            "x0": 0.0,
            "y0": 0.0,
            "I3": 0.0,
            "A3": 0.0,
            "I4": 0.0,
            "A4": 0.0,
            "slope": 0.0,
        }
    )
    window.profile.set_result(FitResult(profile=profile, model=None, residual=None, masked=None))
    window.profile.devauc.setChecked(True)
    assert "Re = 40.0 px" in window.profile.fit_note.text()
    window.close()


def test_menus_tools_masks_help_and_tips():
    from PySide6 import QtWidgets

    from astro_vista import guide
    from astro_vista.app import MainWindow, remove_mac_services

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()

    def menu(title):
        for action in window.menuBar().actions():
            if action.text().replace("&", "").replace("\u200b", "") == title:
                return action.menu()

    def texts(m):
        return [a.text().replace("&", "") for a in m.actions() if not a.isSeparator()]

    assert "Import a mask (any file name)…" not in texts(menu("File"))
    assert texts(menu("Tools")) == [
        "ELLIPROF",
        "Source finder",
        "Display",
        "Sky plane",
        "Contours",
    ]
    masks = texts(menu("Masks"))
    assert (
        masks[:7]
        == [
            "Unmask",
            "Mask circle",
            "Mask ellipse",
            "Mask box",
            "Mask polygon",
            "Mask brush",
            "Mask eraser",
        ]
        and "More" in masks
    )
    helps = texts(menu("Help"))
    assert {"Tips", "FAQ", "Troubleshooting", "List the commands"} <= set(helps)
    assert "First lesson" not in helps and "lesson" in guide.TOPICS
    assert window.a_fit.text() == "Elliprof" and not window.a_fit.icon().isNull()
    assert "sources N" in guide.TOPICS["allcommands"][1] and guide.TIP_LIST
    remove_mac_services()  # never fails
    window.close()


def test_top_bar_buttons_open_the_tools():
    from PySide6 import QtWidgets

    from astro_vista.app import MainWindow

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.side.setCurrentWidget(window.results)
    window.tb_finder.trigger()  # opens the Source finder, runs nothing
    assert window.side.currentWidget() is window.tools
    assert window.tools.currentWidget().widget() is window.finder
    assert not window.busy() and not window.session.runs
    window.tb_display.trigger()  # Display and Contours open their panels too
    assert window.tools.currentWidget().widget() is window.display
    window.tb_contours.trigger()
    assert window.tools.currentWidget().widget() is window.contour_panel
    bar = next(t for t in window.findChildren(QtWidgets.QToolBar) if t.objectName() == "Run")
    assert bar.actions() == [
        window.tb_elliprof,
        window.tb_finder,
        window.tb_display,
        window.tb_contours,
    ]
    window.tb_elliprof.trigger()
    assert window.tools.currentWidget().widget() is window.elliprof
    assert not window.session.fits
    window.close()


def test_a_streak_along_the_columns_is_measured():
    # a one-pixel vertical streak has theta = pi/2 in float32, a little more than pi/2:
    # the source finder must still measure it (it once failed: invalid aperture parameters)
    from astro_vista import detection

    data = np.random.default_rng(1).normal(0, 1, (200, 200))
    data[40:120, 100] += 50
    image = Image("streak", data.astype(np.float32))
    run = detection.detect(image, {"THRESH_TYPE": "ABSOLUTE", "DETECT_THRESH": "5", "FILTER": "N"})
    assert len(run) >= 1 and np.isfinite(run.columns["FLUX_AUTO"]).all()


def test_results_go_into_the_buffers_chosen(tmp_path):
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()

    def wait():
        deadline = QtCore.QDeadlineTimer(60000)
        while window.busy() and not deadline.hasExpired():
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)

    window.open_path(write_fits(galaxy(), tmp_path / "galaxy.fits"))
    wait()
    choice = window.elliprof.model_buffer
    # only the free buffers are offered (buffer 1 holds the image), then all on request
    offered = [choice.itemData(k) for k in range(choice.count())]
    assert 1 not in offered and 2 in offered and choice.SHOW_ALL in offered
    choice.activated.emit(offered.index(choice.SHOW_ALL))
    assert 1 in [choice.itemData(k) for k in range(choice.count())]
    choice.hidePopup()
    window.elliprof.set_values(dict(x0=150, y0=140, r0=4, r1=80, nr=6, sky=100.0))
    window.elliprof.model_buffer.set_value(7)
    window.elliprof.residual_buffer.set_value(9)
    window.run_fit()
    wait()
    assert window.session.buffers[7].name.endswith("model")
    assert window.session.buffers[9].name.endswith("residual")
    # the buffer chosen stays chosen once it is filled
    assert window.elliprof.model_buffer.value() == 7
    window.finder.add_segmentation.setChecked(True)
    window.finder.segmentation_buffer.set_value(12)
    window.show_buffer(1)
    window.run_detection()
    wait()
    assert 12 in window.session.buffers
    window.close()


def test_the_display_bar_folds_away_and_comes_back():
    from PySide6 import QtWidgets

    from astro_vista.app import MainWindow

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    viewer = window.viewer
    assert not hasattr(viewer, "title")  # no header line above the image any more
    viewer.fold_button.click()
    assert viewer.bar_folded and viewer.bar_holder.isHidden()
    assert not window.a_display_bar.isChecked()
    window.a_display_bar.trigger()  # View ▸ Display settings bar
    assert not viewer.bar_folded and not viewer.bar_holder.isHidden()
    viewer.set_bar_folded(True)
    window.reset_layout()
    assert not viewer.bar_folded
    window.close()


def test_the_mask_buttons_are_in_the_right_click_menu():
    from PySide6 import QtWidgets

    from astro_vista.app import MainWindow

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    layers = window.layers
    assert not layers.findChildren(QtWidgets.QPushButton)  # no buttons in the panel
    menu = QtWidgets.QMenu()
    layers._general_actions(menu)
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert texts == [
        "New mask…",
        "Load mask…",
        "Mask from a buffer…",
        "Undo the last mask edit",
        "Redo",
    ]
    window.close()


def test_every_question_mark_has_a_write_up_in_the_documentation():
    from PySide6 import QtWidgets

    from astro_vista import guide
    from astro_vista.app import MainWindow

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    # every ? bubble made in the window has its entry on the settings page...
    page = guide.TOPICS["settings"][1]
    assert guide.MADE_TIPS and guide.MADE_TIPS <= set(guide.TIPS)
    for key in guide.MADE_TIPS:
        assert f"<a name='{key}'></a>" in page, key
    # ...every ? that opens a page opens one that exists
    marks = [b for b in window.findChildren(QtWidgets.QToolButton) if b.text().startswith("?")]
    assert len(marks) > 40
    for b in marks:
        assert not b.toolTip()  # no hover text: a click opens the documentation
        topic, setting = b.property("help_topic"), b.property("help_setting")
        assert topic in guide.TOPICS or setting in guide.TIPS, b.text()
    # a setting's ? opens the documentation at its entry
    next(b for b in marks if b.property("help_setting") == "DETECT_THRESH").click()
    assert guide._window.isVisible()
    assert guide._window.topics.currentItem().data(0x0100) == "settings"
    guide._window.close()
    # a tool's ? opens its page
    next(b for b in marks if b.property("help_topic") == "elliprof").click()
    assert guide._window.topics.currentItem().data(0x0100) == "elliprof"
    guide._window.close()
    window.close()


def test_the_arithmetic_page_examples_all_run():
    # every formula shown on Help ▸ Documentation ▸ Image arithmetic works as written
    import html
    import re

    from astro_vista import guide

    s = Session()
    rng = np.random.default_rng(3)
    for n in (1, 2, 3, 5, 8):
        s.buffers[n] = Image(f"image {n}", rng.uniform(1, 2000, (20, 20)).astype(np.float32))
    s.execute("mask new mymask 1")
    page = html.unescape(guide.TOPICS["arithmetic"][1])
    lines = []
    for block in re.findall(r"<pre>(.*?)</pre>", page, re.S):
        for line in block.splitlines():
            line = line.split("#")[0].strip()
            if line:
                lines.append(line)
    assert len(lines) >= 12
    for line in lines:
        s.execute(line)  # raises on any formula that does not work
    page = guide.TOPICS["faq"][1] + guide.TOPICS["credits"][1]
    assert "github.com/ekourkchi/AstroVista-Lite" in page and "Linux and macOS" in page


def test_blinking_zoomed_in_locks_on_the_wcs_or_the_image():
    from astropy.io import fits
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    app.processEvents()

    def sky(n, scale):  # a tangent-plane WCS, the same sky at the middle
        h = fits.Header()
        h.update(CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CRVAL1=150.0, CRVAL2=20.0)
        h.update(CRPIX1=n / 2 + 0.5, CRPIX2=n / 2 + 0.5)
        h.update(CDELT1=-scale / 3600, CDELT2=scale / 3600)
        return h

    rng = np.random.default_rng(2)
    s = window.session
    s.buffers[1] = Image("fine", rng.normal(0, 1, (300, 300)).astype(np.float32), sky(300, 1.0))
    s.buffers[2] = Image("coarse", rng.normal(0, 1, (150, 150)).astype(np.float32), sky(150, 2.0))
    s.buffers[3] = Image("no wcs", rng.normal(0, 1, (300, 300)).astype(np.float32))
    window._refresh_buffers()
    window.show_buffer(1)
    viewer = window.viewer

    def middle():
        p = viewer.group.mapFromParent(viewer.view.viewRect().center())
        return p.x(), p.y()

    # the whole image in view: the next one is shown whole
    assert viewer.view_lock() is None
    # zoomed in on (200, 100) of the fine image
    viewer.centre_on(200.0, 100.0)
    viewer.zoom(4)
    viewer.centre_on(200.0, 100.0)
    app.processEvents()
    zoom = viewer.zoom_factor()
    viewer.blink_box.setChecked(True)
    window._blink_step(+1)  # to the coarse image: the same sky, pixels twice as large
    assert window.shown == 2 and "locked on the WCS" in window.statusBar().currentMessage()
    x, y = middle()
    assert abs(x - 100.0) < 0.5 and abs(y - 50.0) < 0.5
    assert abs(viewer.zoom_factor() / zoom - 2.0) < 0.05
    window._blink_step(+1)  # to the image without a WCS: the same pixel, the same zoom
    assert window.shown == 3 and "locked on the image" in window.statusBar().currentMessage()
    x, y = middle()
    assert abs(x - 100.0) < 0.5 and abs(y - 50.0) < 0.5
    window.close()


def test_ellipse_box_and_polygon_masks_rotate_before_they_are_applied():
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    s = window.session
    s.buffers[1] = Image("flat", np.zeros((200, 200), np.float32))
    window._refresh_buffers()
    window.show_buffer(1)
    app.processEvents()
    viewer = window.viewer

    def wait():
        deadline = QtCore.QDeadlineTimer(20000)
        while window.busy() and not deadline.hasExpired():
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)

    def mouse(kind, x, y, button=Qt.LeftButton, buttons=Qt.LeftButton, keys=Qt.NoModifier):
        """A mouse event on the canvas at the image point (x, y)."""
        scene = viewer.group.mapToScene(QtCore.QPointF(x, y))
        p = QtCore.QPointF(viewer.canvas.mapFromScene(scene))
        event = QtGui.QMouseEvent(kind, p, p, button, buttons, keys)
        viewer.eventFilter(viewer.canvas.viewport(), event)

    def key(k, text=""):
        viewer.eventFilter(
            viewer.canvas, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, k, Qt.NoModifier, text)
        )

    window.set_mode("box")
    mask = lambda: s.masks[window.active_mask].masked  # noqa: E731
    # a long, thin box, drawn: it waits, unapplied, with its handle above it
    mouse(QtCore.QEvent.MouseButtonPress, 60, 95)
    mouse(QtCore.QEvent.MouseMove, 140, 105, Qt.NoButton)
    mouse(QtCore.QEvent.MouseButtonRelease, 140, 105, buttons=Qt.NoButton)
    assert viewer.pending()["kind"] == "box" and not mask().any()
    # the handle dragged to the left of the centre turns it by 90° (Shift: 15° steps)
    hx, hy, *_ = viewer._handle(viewer.pending())
    mouse(QtCore.QEvent.MouseButtonPress, hx, hy)
    mouse(QtCore.QEvent.MouseMove, 48, 101, Qt.NoButton, keys=Qt.ShiftModifier)
    mouse(QtCore.QEvent.MouseButtonRelease, 50, 100, buttons=Qt.NoButton)
    assert viewer.pending()["angle"] == 90 and window.angle.value() == 90
    key(Qt.Key_Return)  # applied: now upright, as a box command
    wait()
    assert viewer.pending() is None
    assert mask()[130, 100] and not mask()[100, 130]
    assert "mask box" in window.console.output.toPlainText()

    # an ellipse turned with the keys, then cancelled: the mask does not change
    before = mask().copy()
    viewer._start_pending("ellipse", False, 100.0, 100.0, 0.0, a=30.0, b=5.0)
    key(Qt.Key_BracketLeft, "[")
    assert abs(viewer.pending()["angle"] - 5) < 1e-6
    window.angle.setValue(45)  # the angle box turns it too
    assert abs(viewer.pending()["angle"] - 45) < 1e-6
    key(Qt.Key_Escape)
    assert viewer.pending() is None and (mask() == before).all()

    # a turned box becomes a polygon; any command applies a waiting shape first
    window.run_command("mask clear " + window.active_mask)
    wait()
    viewer._start_pending(
        "box", False, 100.0, 100.0, 0.0, points=[(-40, -3), (40, -3), (40, 3), (-40, 3)]
    )
    viewer.set_pending_angle(45)
    window.run_command("stats 1")
    wait()
    assert viewer.pending() is None and "mask polygon" in window.console.output.toPlainText()
    assert mask()[120, 120] and not mask()[100, 130]  # along the diagonal only
    window.close()


def test_a_waiting_shape_moves_and_resizes_before_it_is_applied():
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    s = window.session
    s.buffers[1] = Image("flat", np.zeros((200, 200), np.float32))
    window._refresh_buffers()
    window.show_buffer(1)
    app.processEvents()
    viewer = window.viewer

    def drag(x0, y0, x1, y1):
        """A left-button drag on the canvas, between image points."""
        for kind, x, y, buttons in (
            (QtCore.QEvent.MouseButtonPress, x0, y0, Qt.LeftButton),
            (QtCore.QEvent.MouseMove, x1, y1, Qt.LeftButton),
            (QtCore.QEvent.MouseButtonRelease, x1, y1, Qt.NoButton),
        ):
            scene = viewer.group.mapToScene(QtCore.QPointF(x, y))
            p = QtCore.QPointF(viewer.canvas.mapFromScene(scene))
            event = QtGui.QMouseEvent(kind, p, p, Qt.LeftButton, buttons, Qt.NoModifier)
            viewer.eventFilter(viewer.canvas.viewport(), event)

    window.set_mode("box")
    viewer._start_pending(
        "box", False, 60.0, 60.0, 0.0, points=[(-10, -5), (10, -5), (10, 5), (-10, 5)]
    )
    drag(62, 61, 102, 101)  # from inside: moved by (40, 40)
    d = viewer.pending()
    assert abs(d["cx"] - 100) < 0.5 and abs(d["cy"] - 100) < 0.5
    # the top-right corner dragged out; the bottom-left one stays where it was
    drag(110, 105, 130, 125)
    d = viewer.pending()
    xs = [p[0] for p in viewer._turned(d, d["points"])]
    ys = [p[1] for p in viewer._turned(d, d["points"])]
    assert abs(min(xs) - 90) < 0.5 and abs(min(ys) - 95) < 0.5
    assert abs(max(xs) - 130) < 0.5 and abs(max(ys) - 125) < 0.5
    viewer.commit_pending()
    deadline = QtCore.QDeadlineTimer(20000)
    while window.busy() and not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.AllEvents, 20)
    m = s.masks[window.active_mask].masked
    assert m[120, 125] and m[96, 91] and not m[60, 60]

    # an ellipse's axes, and a polygon's corner
    window.set_mode("ellipse")
    viewer._start_pending("ellipse", False, 100.0, 100.0, 0.0, a=20.0, b=10.0)
    drag(120, 100, 140, 100)  # the end of the long axis
    drag(100, 110, 100, 105)  # the end of the short axis
    assert abs(viewer.pending()["a"] - 40) < 0.5 and abs(viewer.pending()["b"] - 5) < 0.5
    viewer.cancel_pending()
    window.set_mode("polygon")
    viewer._start_pending(
        "polygon", False, 100.0, 100.0, 0.0, points=[(-10, -10), (10, -10), (0, 10)]
    )
    drag(100, 110, 100, 150)
    assert abs(viewer.pending()["points"][2][1] - 50) < 0.5
    viewer.cancel_pending()
    window.close()


def test_the_command_line_loads_images_and_masks(tmp_path):
    from PySide6 import QtCore, QtWidgets

    from astro_vista import app as app_module
    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    a = write_fits(galaxy(), tmp_path / "a.fits")
    b = write_fits(galaxy(stars=False), tmp_path / "b.fits")
    mask = np.ones((281, 301), np.float32)
    mask[:10] = 0
    m = write_fits(Image("m", mask), tmp_path / "edge_mask.fits")
    args = app_module.make_parser().parse_args(["-i", str(a), str(b), "-m", str(m)])
    window = MainWindow()
    assert window.side.currentWidget() is window.inspector  # the tab at the start
    app_module.open_arguments(window, args)
    deadline = QtCore.QDeadlineTimer(20000)
    while window.busy() and not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.AllEvents, 20)
    s = window.session
    assert s.buffers[1].name.startswith("a.fits") and s.buffers[2].name.startswith("b.fits")
    assert "edge_mask" in s.masks and s.masks["edge_mask"].masked[:10].all()
    assert window.elliprof.ticked_layers() == ["edge_mask"]
    assert window.side.currentWidget() is window.inspector
    window.close()


def test_the_unmask_toggle_and_the_tip_under_the_layers():
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    s = window.session
    s.buffers[1] = Image("flat", np.zeros((200, 200), np.float32))
    window._refresh_buffers()
    window.show_buffer(1)
    app.processEvents()
    viewer = window.viewer

    def click(x, y, keys=Qt.NoModifier):
        for kind, buttons in (
            (QtCore.QEvent.MouseButtonPress, Qt.LeftButton),
            (QtCore.QEvent.MouseButtonRelease, Qt.NoButton),
        ):
            scene = viewer.group.mapToScene(QtCore.QPointF(x, y))
            p = QtCore.QPointF(viewer.canvas.mapFromScene(scene))
            event = QtGui.QMouseEvent(kind, p, p, Qt.LeftButton, buttons, keys)
            viewer.eventFilter(viewer.canvas.viewport(), event)
        deadline = QtCore.QDeadlineTimer(20000)
        while window.busy() and not deadline.hasExpired():
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)

    window.set_mode("circle")
    mask = lambda: s.masks[window.active_mask].masked  # noqa: E731
    window.run_command(f"mask fill {window.active_mask}")
    click(100, 100)  # Unmask off: masks (already masked)
    assert mask()[100, 100]
    # the toggle sits just before the circle tool in the tool bar
    bar = next(t for t in window.findChildren(QtWidgets.QToolBar) if t.objectName() == "Tools")
    acts = bar.actions()
    assert acts.index(window.a_unmask) + 1 == acts.index(window.mode_actions["circle"])
    window.a_unmask.trigger()  # on: a click unmasks
    assert viewer.unmask
    click(100, 100)
    assert not mask()[100, 100]
    click(150, 150, Qt.ShiftModifier)  # Shift does the opposite: here it masks
    assert mask()[150, 150]
    click(50, 50)
    assert not mask()[50, 50]
    window.a_unmask.trigger()  # off again
    assert not viewer.unmask
    # the tip with its bulb, under the opacity slider
    assert "Alt" in window.layers.tip_text.text() and window.layers.tip_box.isVisible()
    window.close()


def test_aligned_csv_png_figures_bulb_and_tool_dividers(tmp_path):
    from PySide6 import QtCore, QtWidgets

    from astro_vista import exports
    from astro_vista.app import MainWindow

    # the CSV: every line as long as the others (the commas line up), read back as numbers
    columns = {
        "NUMBER": np.arange(1, 4),
        "X_IMAGE": np.array([1.5, 1234.25, 99.0]),
        "MAG_AUTO": np.array([-12.3456, np.nan, -9.1]),
        "FLUX_AUTO": np.array([1e3, 2.5e5, 7.0]),
    }
    path = tmp_path / "s.csv"
    exports.aligned_csv(path, columns)
    lines = path.read_text().splitlines()
    assert len({len(line) for line in lines}) == 1 and len(lines) == 4
    from astropy.table import Table

    table = Table.read(path, format="ascii.csv")
    assert table.colnames == list(columns) and abs(table["X_IMAGE"][1] - 1234.25) < 1e-9

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()

    def wait():
        deadline = QtCore.QDeadlineTimer(60000)
        while window.busy() and not deadline.hasExpired():
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)

    window.open_path(write_fits(galaxy(), tmp_path / "galaxy.fits"))
    wait()
    window.elliprof.set_values(dict(x0=150, y0=140, r0=4, r1=80, nr=10, sky=100.0))
    window.run_fit()
    window.run_detection()
    wait()
    window.profile.devauc.setChecked(True)
    window.profile.write_png(tmp_path / "p.png", "galaxy")
    window.plots.mark(0)
    window.plots.write_png(tmp_path / "q.png", "galaxy")
    for name in ("p.png", "q.png"):
        assert (tmp_path / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # the tip starts folded to its bulb; a click unfolds it, another folds it
    layers = window.layers
    assert layers.tip_text.isHidden()
    layers.bulb.click()
    assert not layers.tip_text.isHidden()
    layers.bulb.click()
    assert layers.tip_text.isHidden()
    # the tool bar's groups are divided: looking | masking | radius and angle
    bar = next(t for t in window.findChildren(QtWidgets.QToolBar) if t.objectName() == "Tools")
    acts = bar.actions()
    crosshair = acts.index(window.mode_actions["crosshair"])
    eraser = acts.index(window.mode_actions["eraser"])
    assert acts[crosshair + 1].isSeparator() and acts[crosshair + 2] is window.a_unmask
    assert acts[eraser + 1].isSeparator()
    window.close()


def test_profile_table_saves_as_csv_and_terminal_font_size(tmp_path):
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()

    def wait():
        deadline = QtCore.QDeadlineTimer(60000)
        while window.busy() and not deadline.hasExpired():
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)

    window.open_path(write_fits(galaxy(), tmp_path / "galaxy.fits"))
    wait()  # the image is in before the fit
    window.elliprof.set_values(dict(x0=150, y0=140, r0=4, r1=80, nr=10, sky=100.0))
    window.run_fit()
    wait()
    profile = window.profile
    # the plot shown: PNG; the table shown: CSV
    assert profile.save.text() == "Save PNG…"
    profile.as_table.setChecked(True)
    assert profile.save.text() == "Save CSV…"
    got = []
    profile.save_csv.connect(lambda: got.append("csv"))
    profile.save_png.connect(lambda: got.append("png"))
    window.profile.save_csv.disconnect(window.export_profile)  # no file dialog in the test
    profile.save.click()
    assert got == ["csv"]
    path = tmp_path / "profile.csv"
    profile.write_csv(path)
    lines = path.read_text().splitlines()
    assert len(lines) == 11 and len({len(line) for line in lines}) == 1
    assert lines[0].split(",")[0].strip() == "Rmaj"
    # the terminal's text size: larger, smaller, back to the system's
    start = window.console.font_size()
    window.a_font_larger.trigger()
    window.a_font_larger.trigger()
    assert window.console.font_size() == start + 2
    assert window.console.line.font().pointSizeF() == start + 2
    window.a_font_smaller.trigger()
    assert window.console.font_size() == start + 1
    window.a_font_default.trigger()
    assert window.console.font_size() == start
    window.close()


def test_closing_the_main_window_closes_the_rest():
    from PySide6 import QtWidgets

    from astro_vista import guide
    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    guide.show("overview")
    docs = guide._window
    assert docs.isVisible()
    tools = window.docks["Tools"]
    tools.toggle_float()  # a panel popped out as a window of its own
    assert tools.isFloating()
    other = QtWidgets.QDialog()  # any other window
    other.show()
    window.close()
    app.processEvents()
    import shiboken6

    assert guide._window is None  # the documentation is closed and deleted
    assert not shiboken6.isValid(docs) or not docs.isVisible()
    assert not tools.isFloating()
    assert not other.isVisible()


def test_every_command_on_the_website_runs(tmp_path):
    # the code shown on the website (docs/*.html) works as written: each command in the
    # pages' code blocks runs, in page order, on copies of the example image
    import html
    import re
    import shutil

    from astropy.io import fits

    from astro_vista.app import EXAMPLE_FILES, example_folder

    docs = Path(__file__).resolve().parent.parent / "docs"
    names = (
        "index.html",
        "tutorial.html",
        "commands.html",
        "terminal.html",
        "reference.html",
        "pairs.html",  # last: it reads the pair into buffer 1
    )
    pages = [docs / name for name in names]
    assert all(page.exists() for page in pages)
    # the files the pages read: the example under the names they use
    data = fits.getdata(example_folder() / EXAMPLE_FILES[0]).astype(np.float32)
    shutil.copy(example_folder() / EXAMPLE_FILES[1], tmp_path / "u12517j.dmask")
    for name in ("u12517j.fits", "galaxy.fits", "exp1.fits", "exp2.fits", "exp3.fits"):
        fits.writeto(tmp_path / name, data)
    fits.HDUList(
        [fits.PrimaryHDU()] + [fits.ImageHDU(data, name="SCI", ver=v) for v in (1, 2)]
    ).writeto(tmp_path / "frame.fits")
    fits.writeto(tmp_path / "edges.fits", np.ones(data.shape, np.uint8))
    (tmp_path / "session.txt").write_text("stats 1\n")
    pair = fits.getdata(example_folder() / "pair.fits.gz")
    fits.writeto(tmp_path / "pair.fits", pair)
    shutil.copy(example_folder() / "pair.dmask", tmp_path / "pair.dmask")
    s = Session()
    s.folder = tmp_path
    s.execute("rd 1 galaxy.fits")
    s.execute("mask load u12517j u12517j.dmask 1")  # as File > Open example does
    s.execute("mask new mymask 1")
    for n in (2, 3, 5, 8):
        s.execute(f"copy 1 {n}")
    shell = ("pip ", "conda ", "astrovista", "python", "source ", "sudo ", "Kourkchi")
    ran = 0
    for page in pages:
        for block in re.findall(r"<pre><code>(.*?)</code></pre>", page.read_text(), re.S):
            text = html.unescape(re.sub(r"<[^>]+>", "", block))
            for line in text.splitlines():
                line = re.sub(r"\s+#.*$", "", line).strip()
                if not line or line.startswith("#") or line.startswith(shell):
                    continue
                s.execute(line)  # raises on any command that does not work
                ran += 1
    assert ran >= 180


def test_the_reference_covers_every_command_and_the_website_matches_it():
    # every command and mask action the terminal knows is in the reference, and the
    # website's generated pages are what docs/build.py makes from it now
    import importlib.util

    from astro_vista import guide, reference
    from astro_vista.commands import ALIASES, MASK_ACTIONS, Session

    verbs = {n[2:] for n in dir(Session) if n.startswith("c_")} - {"arithmetic"}
    documented = reference.documented_names()
    assert not verbs - documented, sorted(verbs - documented)
    assert not {f"mask {a}" for a in MASK_ACTIONS} - documented
    assert all(alias in reference.TERMINAL_HTML for alias in ALIASES)
    assert all(c.syntax and c.what and c.example for c in reference.commands())
    # the documentation window has both pages, and the credits name John Tonry
    assert guide.TOPICS["terminal"][1] == reference.TERMINAL_HTML
    assert "elliprof N X0=" in guide.TOPICS["allcommands"][1]
    assert "John Tonry" in guide.TOPICS["credits"][1]
    docs = Path(__file__).resolve().parent.parent / "docs"
    spec = importlib.util.spec_from_file_location("docs_build", docs / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    for name, text in build.pages().items():
        assert (docs / name).read_text() == text, f"run python docs/build.py ({name})"


def test_variable_names_in_any_case():
    s = Session()
    s.buffers[1] = Image("a", np.ones((10, 10), np.float32))
    s.execute("set XC=5 rc=2")
    assert s.variables == {"xc": 5.0, "rc": 2.0}
    s.execute("gauss 1 X=XC+0.5 Y=Rc FWHM=2 PEAK=1")  # keyword values, an expression too
    s.execute("[2] = [1] * xc + RC")  # formulas
    assert abs(s.buffers[2].data[0, 0] - (s.buffers[1].data[0, 0] * 5 + 2)) < 1e-5
    assert "column 5" in s.execute("max 1").text
    assert "5" in s.execute("print 1 XC 3 SIZE=1").text.splitlines()[0]  # numbers in commands


def test_the_median_filter_leaves_nan_out_quickly():
    # filt on an image with NaN pixels: the same as SciPy's per-pixel nanmedian filter,
    # boxes that are all NaN stay NaN, and a 1000 x 1000 image takes well under a second
    import time
    import warnings

    from scipy import ndimage

    from astro_vista.imageops import nan_median_filter

    rng = np.random.default_rng(1)
    a = rng.normal(size=(30, 41))
    a[rng.random(a.shape) < 0.3] = np.nan
    a[:5, :5] = np.nan
    for k in (2, 3, 5):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = ndimage.generic_filter(a, np.nanmedian, size=k)
        got = nan_median_filter(a, k, budget=4000)  # in bands of a few rows
        assert np.array_equal(np.isnan(got), np.isnan(expected))
        assert np.allclose(got, expected, equal_nan=True)
    s = Session()
    big = rng.normal(size=(1000, 1000)).astype(np.float32)
    big[::40, ::40] = np.nan
    s.buffers[1] = Image("big", big)
    start = time.perf_counter()
    s.execute("filt 1 MEDIAN=5 OUT=2")
    assert time.perf_counter() - start < 3.0
    assert np.isfinite(s.buffers[2].data).all()  # every 5 x 5 box has finite pixels


def test_a_mask_saved_as_a_legacy_bitmap_gets_the_dmask_ending(tmp_path):
    # the type chosen in the save dialog decides the ending, which decides the format
    from astro_vista.panels import mask_file_name

    bitmap, fits_ = "Legacy bitmap (*.dmask)", "FITS (*.fits)"
    assert mask_file_name("/d/edges_mask.fits", bitmap) == "/d/edges_mask.dmask"
    assert mask_file_name("/d/edges", bitmap) == "/d/edges.dmask"
    assert mask_file_name("/d/edges.dmask", fits_) == "/d/edges.fits"
    assert mask_file_name("/d.v1/edges", bitmap) == "/d.v1/edges.dmask"
    assert mask_file_name("/d/edges_mask.fits", fits_) == "/d/edges_mask.fits"
    s = Session()
    s.folder = tmp_path
    s.buffers[1] = Image("a", np.ones((20, 30), np.float32))
    s.execute("mask new edges 1")
    s.execute("mask box edges 0 0 5 5")
    s.execute(f"mask save edges {mask_file_name(str(tmp_path / 'edges_mask.fits'), bitmap)}")
    from astro_vista.images import is_bitmap_mask

    assert is_bitmap_mask(tmp_path / "edges_mask.dmask")
    s.execute("mask load back edges_mask.dmask 1")
    assert (s.masks["back"].masked == s.masks["edges"].masked).all()


def test_masking_the_sources_before_a_fit_keeps_the_galaxy(tmp_path):
    # with no ELLIPROF centre yet, mask sources keeps the largest source (the galaxy)
    s = Session()
    s.folder = tmp_path
    write_fits(galaxy(), tmp_path / "galaxy.fits")
    s.execute("rd 1 galaxy.fits")
    s.execute("sources 1")
    s.execute("mask new found 1")
    reply = s.execute("mask sources found SCALE=1.5")
    assert "the largest" in reply.text
    assert not s.masks["found"].masked[140, 150]  # the galaxy's centre stays unmasked
    s.execute("mask clear found")
    s.execute("mask sources found KEEP=none")  # unless asked to mask everything
    assert s.masks["found"].masked[140, 150]


def test_the_session_is_kept_when_buffers_close_and_comes_back(tmp_path):
    s = session_with_galaxy(tmp_path)
    s.execute("copy 1 2 tag=mine")
    s.buffers[2].about = "model #3: its parameters"
    s.execute("mask new stars 1")
    s.execute("mask circle stars 40 50 6")
    s.execute("mask role stars fit")
    folder = tmp_path / ".temp"
    s.store_folder = folder
    before = {n: b.data.copy() for n, b in s.buffers.items()}
    reply = s.execute("close all")
    assert "saved 2 buffers and 1 mask layers" in reply.text and not s.buffers
    assert "2 buffers and 1 mask layers" in s.execute("session").text
    # closing with nothing left never replaces the saved session with an empty one
    s.execute("mask delete stars")
    assert s.keep_session() == ""
    text = s.execute("session restore").text
    assert text.startswith("restored the session") and sorted(s.buffers) == [1, 2]
    for n, data in before.items():
        assert np.array_equal(s.buffers[n].data, data)
    b = s.buffers[2]
    assert b.tag == "mine" and b.about == "model #3: its parameters" and b.note
    m = s.masks["stars"]
    assert m.role == "fit" and m.masked[50, 40] and m.count() > 50
    # a second save keeps the first beside it; restoring into taken numbers uses free ones
    s.execute("session save")
    assert (tmp_path / ".temp.previous" / "session.json").exists()
    s.execute("session restore")
    assert len(s.buffers) == 4 and "stars_restored" in s.masks
    with pytest.raises(CommandError, match="no saved session"):
        s.execute(f"session restore {tmp_path / 'nothing'}")


def test_the_window_keeps_the_session_and_offers_it_at_the_start(tmp_path, monkeypatch):
    from PySide6 import QtCore, QtWidgets

    from astro_vista import app as app_module
    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app_module.settings().setValue("session/folder", str(tmp_path / "kept"))
    try:
        window = MainWindow()

        def wait(w):
            deadline = QtCore.QDeadlineTimer(60000)
            while w.busy() and not deadline.hasExpired():
                app.processEvents(QtCore.QEventLoop.AllEvents, 50)

        assert window.session.store_folder == tmp_path / "kept"
        window.open_path(write_fits(galaxy(), tmp_path / "galaxy.fits"))
        window.run_command("mask new edge 1")
        window.run_command("mask box edge 0 0 20 20")
        wait(window)
        window.close()  # by accident: the work is kept
        assert (tmp_path / "kept" / "session.json").exists()
        again = MainWindow()
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.Yes
        )
        again.offer_session()
        wait(again)
        assert sorted(again.session.buffers) == [1] and "edge" in again.session.masks
        assert again.viewer.image is not None and again.buffer_list.count() == 1
        again.close()
    finally:
        app_module.settings().remove("session/folder")


def test_a_click_inside_a_source_takes_its_centre_and_the_centres_show_as_crosses(tmp_path):
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()

    def wait():
        deadline = QtCore.QDeadlineTimer(60000)
        while window.busy() and not deadline.hasExpired():
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)

    window.open_path(write_fits(galaxy(), tmp_path / "galaxy.fits"))
    wait()
    window.run_command("sources 1")
    wait()
    run = window.session.runs[-1]
    row = run.under(158.0, 147.0)  # well off the galaxy's centre, inside it
    assert row is not None and run.under(5.0, 5.0) is None
    cx, cy = run.centre_of(row)
    assert abs(cx - 150) < 1 and abs(cy - 140) < 1
    # pair: Pick A, click inside the galaxy
    window._pick_target = "pair:A:centre"
    window._centre_picked(158.0, 147.0)
    assert window.pair_panel.cards["A"].centre() == pytest.approx((cx, cy), abs=0.01)
    # ELLIPROF the same
    window._pick_target = "centre"
    window._centre_picked(158.0, 147.0)
    wait()
    assert (window.elliprof.x0.value(), window.elliprof.y0.value()) == pytest.approx(
        (cx, cy), abs=0.01
    )
    # the centres as small crosses, on a tick
    marks = window.viewer.source_centres
    assert len(marks.data) == len(run)
    window.finder.show_centres.setChecked(False)
    wait()
    assert len(marks.data) == 0 and not window.layers.checked("srccentres")
    window.close()
