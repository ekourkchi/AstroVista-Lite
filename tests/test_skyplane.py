"""The sky as a plane: the fitter, the skyplane command and the Sky plane tool."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from astro_vista import sky, synthetic
from astro_vista.commands import Session
from astro_vista.images import Image


def test_a_flat_sky_has_no_significant_gradient():
    scene = synthetic.blank(seed=3)
    fit = sky.fit_plane_sky(scene.data, "blank")
    assert fit.converged and len(fit.passes) <= 2
    assert not fit.significant
    assert abs(fit.plane["s0"] - scene.sky) < 0.3 * fit.sky.sigma
    assert 0.7 < fit.scatter < 1.5


def test_a_steep_gradient_is_recovered_where_one_level_fails():
    scene = synthetic.blank(seed=3, gradient=(0.02, -0.01))  # 2 and 1 sigma per 1000 px
    one = sky.measure(scene.data, "gradient")
    fit = sky.fit_plane_sky(scene.data, "gradient")
    p, e = fit.plane, fit.errors
    assert fit.significant and fit.converged
    for got, err, true in ((p["gx"], e["gx"], 0.02), (p["gy"], e["gy"], -0.01)):
        assert abs(got - true) < 4 * err + 0.001
    assert abs(p["s0"] - scene.sky) < 0.3 * fit.sky.sigma
    # against one level, the bright side is taken for a source and the level is biased
    assert one.support_fraction < 0.5 < fit.sky.support_fraction
    assert abs(one.level - scene.sky) > abs(p["s0"] - scene.sky)
    # the flat image has its level kept and the tilt gone
    again = sky.fit_plane_sky(fit.flat, "flat")
    assert not again.significant


def test_the_skyplane_command_makes_the_flat_image_with_its_sky():
    session = Session()
    scene = synthetic.pair(gradient=(0.01, 0.006))
    session.buffers[1] = Image("pair", scene.data)
    reply = session.execute("skyplane 1 TOFLAT=2 TOPLANE=3 TOSUB=4")
    assert "sky plane" in reply.text and reply.show == 2
    plane, flat, sub = (session.buffers[n].data for n in (3, 2, 4))
    s0 = reply.values["sky"]
    assert np.allclose(flat, scene.data - (plane - s0), atol=1e-2)
    assert np.allclose(sub, scene.data - plane, atol=1e-2)
    # the flat image is an image of its own, whose sky is measured and current
    flat_image = session.buffers[2]
    sid = session.sky_of[flat_image.key]
    assert session.sky_stale(sid) is None
    assert abs(session.skies[sid].level - s0) < session.skies[sid].sigma
    assert flat_image.key != session.buffers[1].key
    assert session.masks["skymask"].role == "sky"
    # the gradient (with the galaxies' halo in the frame) is close to the truth
    assert abs(reply.values["skygx"] - 0.01) < 0.002
    assert abs(reply.values["skygy"] - 0.006) < 0.002


def test_the_sky_plane_tool_in_the_window():
    from PySide6 import QtCore, QtWidgets

    from astro_vista.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()

    def settle(seconds=60):
        deadline = QtCore.QDeadlineTimer(int(seconds * 1000))
        while window.busy() and not deadline.hasExpired():
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        app.processEvents()

    scene = synthetic.blank(seed=5, gradient=(0.02, 0.0))
    window.session.buffers[1] = Image("tilted", scene.data)
    window._refresh_buffers()
    window.show_buffer(1)
    panel = window.skyplane
    panel.add_plane.setChecked(True)
    panel.run_requested.emit()
    settle()
    names = {b.name for b in window.session.buffers.values()}
    assert {"tilted flat", "tilted sky plane"} <= names
    # the flat image is shown; back on the image, the panel shows its plane
    assert window.viewer.image.name == "tilted flat"
    window.show_buffer(1)
    app.processEvents()
    assert "significant gradient" in panel.card.text()
    # run again: the same buffers are reused
    count = len(window.session.buffers)
    panel.run_requested.emit()
    settle()
    assert len(window.session.buffers) == count
    window.close()
