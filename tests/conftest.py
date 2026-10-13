"""The tests run with a scratch settings folder: they never read or change the user's."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore


@pytest.fixture(autouse=True, scope="session")
def scratch_settings(tmp_path_factory):
    folder = tmp_path_factory.mktemp("settings")
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(folder))
    yield folder


@pytest.fixture(autouse=True, scope="session")
def tidy_qt_at_the_end():
    """Close and delete every window before Python exits, so that Qt objects are not
    destroyed in the wrong order at shutdown."""
    yield
    import gc

    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    for widget in QtWidgets.QApplication.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    for _ in range(3):
        app.processEvents()
        app.sendPostedEvents(None, 0)
    gc.collect()
