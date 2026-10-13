"""A whole session in the real window, as a user would have it: the example opened, the
isophotes fitted, the sources found and masked, the fit repeated, and the results written.
Run it under any Qt platform (CI runs it under X11 through xvfb):

    python tests/smoke.py OUTPUT_FOLDER

It exits with 0 when every step worked, and 1 (saying which step) when one did not.
"""

import pathlib
import sys
import tempfile

from PySide6 import QtCore, QtWidgets

from astro_vista import __version__
from astro_vista.app import MainWindow


def main() -> int:
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "smoke-output")
    out.mkdir(parents=True, exist_ok=True)
    # a scratch settings folder: the run never reads or changes the user's settings
    QtCore.QSettings.setPath(
        QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, tempfile.mkdtemp()
    )
    app = QtWidgets.QApplication(sys.argv)
    print(f"AstroVista Lite {__version__} on the Qt platform {app.platformName()}")
    window = MainWindow()
    window.resize(1400, 900)
    window.show()

    def wait():
        deadline = QtCore.QDeadlineTimer(120000)
        while window.busy() and not deadline.hasExpired():
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        for _ in range(5):
            app.processEvents()

    def check(ok, step):
        if not ok:
            print(f"FAILED: {step}")
            raise SystemExit(1)
        print(f"ok: {step}")

    window.open_example()
    wait()
    check(window.viewer.image is not None, "the example image opened")
    window.run_fit()
    wait()
    check(window.profile.table.rowCount() > 10, "the isophotes were fitted")
    window.run_detection()
    wait()
    check(window.sources.table.rowCount() > 20, "the sources were found")
    window.mask_sources()
    wait()
    window.run_fit()
    wait()
    check(window.profile.table.rowCount() > 10, "the fit with the sources masked")
    window.profile.write_png(out / "profile.png", "UGC 12517")
    window.profile.write_csv(out / "profile.csv")
    window.plots.write_png(out / "plot.png", "UGC 12517")
    for name in ("profile.png", "plot.png"):
        check((out / name).read_bytes()[:4] == b"\x89PNG", f"{name} written")
    window.run_command("[9] = [1] - median([1])")
    wait()
    check(9 in window.session.buffers, "arithmetic on the buffers")
    window.grab().save(str(out / "window.png"))
    window.close()
    print("every step worked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
