"""The AstroVista Lite window: buffers, a viewer with layers and masks, ELLIPROF and the
Source finder.

Every action that changes data is a command line (commands.py), echoed in the Terminal
panel, so that what was done can be read, repeated and saved as a command file.  Display
settings are not commands: they never change pixel values.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
import threading
import traceback
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import __version__, detection, guide, icons
from . import pair as pairs
from . import sky as skies
from .docks import FoldableDock
from .commands import SLOW, CommandError, Reply, Session, completions, verb_of
from .images import image_hdus, is_bitmap_mask, is_fits, read_fits
from .panels import (
    QuickInspector,
    CommandPanel,
    ContourPanel,
    DisplayPanel,
    ElliprofPanel,
    HeaderPanel,
    LayersPanel,
    PlotsPanel,
    ProfilePanel,
    SourceFinderPanel,
    SourcesPanel,
    scrolled,
)
from .pairpanel import PairPanel, PairResultsPanel
from .skyplanepanel import SkyPlanePanel
from .viewer import Viewer, app_icon

TITLE = "AstroVista Lite"
LAYOUT_VERSION = 6  # saved window layouts of another version are not restored

# The example, shipped with the package: an HST WFC3/IR image of UGC 12517 and its mask,
# from the elliprof package's published example, with the parameters published there.
EXAMPLE_FILES = ("u12517j.fits.gz", "u12517j.dmask")
EXAMPLE_FIT = {"x0": 567, "y0": 562, "r0": 9, "r1": 347, "nr": 23, "niter": 10}
EXAMPLE_FIT.update({"sky": 3246.0, "rmstar": True})

TOOLS = (
    # mode, label, icon, shortcut, explanation
    ("pan", "Pan and select", "✥", "P", "Drag to pan, wheel to zoom, click a source to select it"),
    ("pick", "Pick the centre", "⌖", "C", "Click the galaxy's centre (inside a source it snaps)"),
    (
        "crosshair",
        "Crosshair",
        "✛",
        "X",
        "The crosshair follows the mouse; drag to pan; Alt+arrows nudge it",
    ),
    ("circle", "Mask circle", "◯", "M", "Click for a circle of the radius set, or drag one"),
    (
        "ellipse",
        "Mask ellipse",
        "⬭",
        "E",
        "Drag from the centre; then move, resize or turn it; Enter applies",
    ),
    ("box", "Mask box", "▭", "B", "Drag a box; then move, resize or turn it; Enter applies"),
    (
        "polygon",
        "Mask polygon",
        "⬠",
        "G",
        "Click the corners, Enter closes it; then move, resize or turn it; Enter applies",
    ),
    ("brush", "Mask brush", "🖌", "R", "Drag to paint"),
    ("eraser", "Mask eraser", "⌫", "U", "Drag to unmask (a brush that erases)"),
)


def name_the_mac_app(name: str) -> None:
    """On macOS, the name in the menu bar (and the Dock's menu) is the program's: python3.
    Give the running application its own name, through the Objective-C runtime (no extra
    package); this must happen before the QApplication is made.  Elsewhere: nothing."""
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        call = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(send)
        call_str = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p
        )(send)
        call_2 = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        )(send)

        def nsstring(text):
            return call_str(
                objc.objc_getClass(b"NSString"),
                objc.sel_registerName(b"stringWithUTF8String:"),
                text.encode(),
            )

        bundle = call(objc.objc_getClass(b"NSBundle"), objc.sel_registerName(b"mainBundle"))
        for selector in (b"localizedInfoDictionary", b"infoDictionary"):
            info = call(bundle, objc.sel_registerName(selector))
            if info:
                call_2(
                    info,
                    objc.sel_registerName(b"setObject:forKey:"),
                    nsstring(name),
                    nsstring("CFBundleName"),
                )
    except Exception:  # the name stays python3: nothing else depends on it
        pass


def padded(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """``widget`` with a little room above it, below a panel's title bar."""
    box = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(0, 6, 0, 0)
    lay.addWidget(widget)
    return box


def remove_mac_services() -> None:
    """Take the Services item out of the macOS application menu (AstroVista Lite ▸
    Services), through the Objective-C runtime; elsewhere, nothing."""
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        call = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(send)
        call_long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(send)
        call_at = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long
        )(send)
        call_obj = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(send)
        utf8 = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)(send)
        call_bool = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)(send)
        sel = objc.sel_registerName
        app = call(objc.objc_getClass(b"NSApplication"), sel(b"sharedApplication"))
        main = call(app, sel(b"mainMenu"))
        if not main:
            return
        first = call_at(main, sel(b"itemAtIndex:"), 0)
        menu = call(first, sel(b"submenu")) if first else None
        if not menu:
            return
        services = call(app, sel(b"servicesMenu"))
        for k in reversed(range(call_long(menu, sel(b"numberOfItems")))):
            item = call_at(menu, sel(b"itemAtIndex:"), k)
            sub = call(item, sel(b"submenu"))
            title = utf8(call(item, sel(b"title")), sel(b"UTF8String")) or b""
            if (services and sub == services) or title == b"Services":
                call_obj(menu, sel(b"removeItem:"), item)
                # the separator left after it goes too
                if k < call_long(menu, sel(b"numberOfItems")):
                    nxt = call_at(menu, sel(b"itemAtIndex:"), k)
                    if call_bool(nxt, sel(b"isSeparatorItem")):
                        call_obj(menu, sel(b"removeItem:"), nxt)
    except Exception:  # the menu stays as macOS made it
        pass


def settings() -> QtCore.QSettings:
    """What AstroVista Lite remembers (the window's layout), in an ini
    file in the user's settings folder."""
    return QtCore.QSettings(
        QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, "AstroVista", "astro-vista"
    )


def file_stem(name: str) -> str:
    """A file name stem for a buffer or a result: 'u12517j.fits.gz residual' gives
    'u12517j_residual', 'x.fits[2]' gives 'x_2'."""
    stem = re.sub(r"\.(fits|fit|fts|dmask)(\.gz|\.fz)?", "", name, flags=re.I)
    stem = re.sub(r"[\[\]]", " ", stem)
    stem = re.sub(r"[^\w.+-]+", "_", stem.strip())
    return stem.strip("_") or "image"


def example_folder() -> Path:
    return Path(__file__).resolve().parent / "data"


def glyph_icon(text: str, size: int = 32) -> QtGui.QIcon:
    """A tool-bar icon drawn from a character, in the palette's text colour."""
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setPen(QtWidgets.QApplication.palette().color(QtGui.QPalette.WindowText))
    font = painter.font()
    font.setPixelSize(int(size * (0.5 if len(text) > 1 else 0.72)))
    font.setBold(len(text) > 1)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, text)
    painter.end()
    return QtGui.QIcon(pixmap)


def eraser_icon(size: int = 32) -> QtGui.QIcon:
    """An eraser: a tilted block, its rubber end pink, over a short line it has rubbed."""
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    p = QtGui.QPainter(pixmap)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    ink = QtWidgets.QApplication.palette().color(QtGui.QPalette.WindowText)
    p.translate(size * 0.52, size * 0.45)
    p.rotate(-35)
    w, h = size * 0.62, size * 0.32
    body = QtCore.QRectF(-w / 2, -h / 2, w, h)
    rubber = QtCore.QRectF(-w / 2, -h / 2, w * 0.38, h)
    p.setPen(QtGui.QPen(ink, size / 16))
    p.setBrush(QtGui.QColor("#7aa7ff"))
    p.drawRoundedRect(body, size / 16, size / 16)
    p.setBrush(QtGui.QColor("#ff8fb0"))
    p.drawRoundedRect(rubber, size / 16, size / 16)
    p.resetTransform()
    p.setPen(QtGui.QPen(ink, size / 18, Qt.SolidLine, Qt.RoundCap))
    p.drawLine(QtCore.QPointF(size * 0.12, size * 0.88), QtCore.QPointF(size * 0.6, size * 0.88))
    p.end()
    return QtGui.QIcon(pixmap)


class WheelGuard(QtCore.QObject):
    """The mouse wheel over a number box, a menu or a slider scrolls the panel instead of
    changing the setting: settings change only when clicked or typed."""

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Wheel and isinstance(
            obj, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QSlider)
        ):
            parent = obj.parentWidget()
            while parent is not None and not isinstance(parent, QtWidgets.QAbstractScrollArea):
                parent = parent.parentWidget()
            if parent is not None:  # scroll what holds the setting
                QtWidgets.QApplication.sendEvent(parent.verticalScrollBar(), event)
            return True
        return False


class ElidedLabel(QtWidgets.QLabel):
    """A label that shortens its text with … instead of widening the window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full = ""
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

    def setText(self, text: str) -> None:
        self._full = text
        self._elide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        metrics = self.fontMetrics()
        super().setText(metrics.elidedText(self._full, Qt.ElideLeft, max(self.width() - 4, 0)))
        self.setToolTip(self._full)

    def minimumSizeHint(self):
        return QtCore.QSize(40, super().minimumSizeHint().height())


class Task(QtCore.QObject):
    """Runs a command's function and reports back in the window's thread.

    Slow commands (``threaded``) run in a Python thread, as the pair fit does, so that the
    window stays alive while they run: buffers can be shown, layers ticked and the view
    changed, and shell commands can be stopped with Ctrl+C.  The thread makes no Qt object;
    its signals arrive queued in the window's thread, where this object lives.  PySide6
    6.12.0, which let the two overlap without Python's interpreter lock, is excluded by the
    requirements; ASTROVISTA_MAIN_THREAD=1 runs them in the window's thread instead."""

    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, function: Callable, parent=None, threaded: bool = False):
        super().__init__(parent)
        self._function = function
        self._threaded = threaded

    def start(self) -> None:
        if self._threaded:
            threading.Thread(target=self._run, name="astrovista-task", daemon=True).start()
        else:  # at the next turn of the event loop, once "Running: ..." is shown
            QtCore.QTimer.singleShot(0, self, self._run)

    def _run(self) -> None:
        try:
            value = self._function()
        except CommandError as e:
            self.failed.emit(str(e))
        except Exception as e:  # shown to the user, not raised in the worker
            self.failed.emit(f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=3))
        else:
            self.done.emit(value)


class PairWorker(QtCore.QObject):
    """Runs a pair fit (pair.run) in a Python thread: NumPy and elliprof subprocesses only,
    no Qt object made there.  Its signals arrive queued in the window's thread."""

    progress = QtCore.Signal(object)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, job, cancelled: Callable[[], bool], parent=None):
        super().__init__(parent)
        self._job = job
        self._cancelled = cancelled

    def start(self) -> None:
        threading.Thread(target=self._run, name="astrovista-pair", daemon=True).start()

    def _run(self) -> None:
        try:
            measured = self._job.measurement
            result = pairs.run_search(
                self._job.snap,
                measured.support if measured is not None else None,
                progress=self.progress.emit,
                cancelled=self._cancelled,
            )
        except Exception as e:  # shown to the user, not raised in the worker
            self.failed.emit(f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=3))
        else:
            self.done.emit(result)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)
        self.session = Session()
        self.shown: Optional[int] = None  # the buffer in the viewer
        self.active_mask: Optional[str] = None  # the mask the mask tools draw on
        self._queue: List[str] = []
        self._busy = False
        self._continued = ""  # the start of a command line continued with \\
        self._when_done: List[Callable[[], None]] = []  # run once the queue is empty
        self._residual_display = None  # (stretch, limits) to restore after a residual
        self._residual_shown = None
        self._pair_worker = None
        self._task = None
        self._running: Optional[str] = None  # the command being run, if any
        self._patch_cache = None  # (key, labels) for the Ctrl-hover outline
        self._pick_target = "centre"  # what the pick tool picks: centre, r0 or r1

        self.viewer = Viewer()
        self.setCentralWidget(self.viewer)

        self.buffer_list = QtWidgets.QListWidget()
        self.buffer_list.currentItemChanged.connect(self._buffer_chosen)
        self.buffer_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.buffer_list.customContextMenuRequested.connect(self._buffer_menu)
        self.buffer_list.setIconSize(QtCore.QSize(40, 40))
        self.layers = LayersPanel()
        left = QtWidgets.QSplitter(Qt.Vertical)
        left.addWidget(self.buffer_list)
        left.addWidget(self.layers)
        left.setSizes([120, 560])
        self.docks = {}
        self._dock("Buffers and layers", left, Qt.LeftDockWidgetArea)

        self.elliprof = ElliprofPanel()
        self.finder = SourceFinderPanel()
        self.display = DisplayPanel()
        self.contour_panel = ContourPanel()
        self.quick = QuickInspector()
        self.inspector = scrolled(self.quick)  # a tab of its own, left of the tools
        self.tools = QtWidgets.QTabWidget()
        self.tools.addTab(scrolled(self.elliprof), "ELLIPROF")
        self.tools.addTab(scrolled(self.finder), "Source finder")
        self.pair_panel = PairPanel()
        self.tools.addTab(scrolled(self.pair_panel), "Pair fit")
        self.skyplane = SkyPlanePanel()
        self.tools.addTab(scrolled(self.skyplane), "Sky plane")
        self.tools.addTab(scrolled(self.display), "Display")
        self.tools.addTab(scrolled(self.contour_panel), "Contours")
        self.tools.setDocumentMode(True)  # flat tabs that the panel's title never covers
        # native tabs, all four shown: their names look like every other tab's
        self.tools.tabBar().setElideMode(Qt.ElideNone)
        self.tools.tabBar().setExpanding(False)
        self.console = CommandPanel()
        self.profile = ProfilePanel()
        self.sources = SourcesPanel()
        self.plots = PlotsPanel()
        self.header = HeaderPanel()
        self.results = QtWidgets.QTabWidget()  # what the tools found
        self.results.setDocumentMode(True)
        for widget, name in (
            (self.profile, "Profile"),
            (self.sources, "Sources"),
            (self.plots, "Plots"),
        ):
            self.results.addTab(widget, name)
        self.pair_results = PairResultsPanel()
        self.results.addTab(self.pair_results, "Pair")
        # the right-hand panel: the inspector, the tools, their results and the header
        self.side = QtWidgets.QTabWidget()
        self.side.setDocumentMode(True)
        for tabs in (self.side, self.results):  # rows of tabs that may shrink
            tabs.tabBar().setUsesScrollButtons(True)
            tabs.tabBar().setExpanding(False)
            tabs.setMinimumWidth(0)
        self.side.addTab(self.inspector, "Inspector")
        self.side.addTab(self.tools, "Tools")
        self.side.addTab(self.results, "Results")
        self.side.addTab(self.header, "Header")
        tools_dock = self._dock("Tools", self.side, Qt.RightDockWidgetArea)
        # no title bar: the pop-out and fold buttons sit on the row of tabs
        self._side_title = QtWidgets.QWidget()
        tools_dock.setTitleBarWidget(self._side_title)
        corner = QtWidgets.QWidget()
        corner_row = QtWidgets.QHBoxLayout(corner)
        corner_row.setContentsMargins(0, 0, 4, 0)
        corner_row.setSpacing(2)
        self.side_pop = QtWidgets.QToolButton()
        self.side_fold = QtWidgets.QToolButton()
        for b in (self.side_pop, self.side_fold):
            b.setAutoRaise(True)
            b.setFixedSize(22, 22)
            corner_row.addWidget(b)
        self.side_pop.clicked.connect(tools_dock.toggle_float)
        self.side_fold.clicked.connect(tools_dock.toggle_fold)
        self.side.setCornerWidget(corner, Qt.TopRightCorner)
        tools_dock.fold_hook = self._fold_side
        tools_dock.topLevelChanged.connect(self._side_floating)
        # the bottom panel: the terminal alone
        terminal_dock = self._dock("Terminal", self.console, Qt.BottomDockWidgetArea)
        terminal_dock.fold_hook = self._fold_terminal
        terminal_dock.topLevelChanged.connect(lambda _f: self._place_command_line())
        terminal_dock.folded_changed.connect(lambda _f: self._place_command_line())
        self.results.currentChanged.connect(lambda _i: self._inspect())
        self.tools.currentChanged.connect(lambda _i: self._inspect())
        self.side.currentChanged.connect(lambda _i: self._inspect())

        # the narrow bar at the bottom: the command line (while the terminal is folded),
        # the latest message, the position under the cursor, the zoom, the panel buttons
        self.console.input_row.setMaximumHeight(28)
        self.message = ElidedLabel()
        self.message.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.message.setStyleSheet("color: #8b9099")
        self.statusBar().addPermanentWidget(self.message, 1)
        self.statusBar().messageChanged.connect(lambda text: text and self.message.setText(text))
        self.position = ElidedLabel()
        self.zoom_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.position, 1)
        self.statusBar().addPermanentWidget(self.zoom_label)
        # a button per panel, to fold and unfold it
        self.panel_buttons = {}
        for title, label in (
            ("Buffers and layers", "Layers"),
            ("Tools", "Tools"),
            ("Terminal", "Terminal"),
        ):
            b = QtWidgets.QToolButton()
            b.setAutoRaise(True)
            b.setToolTip(f"Fold or unfold the {title} panel")
            b.clicked.connect(
                self.toggle_terminal if title == "Terminal" else self.docks[title].toggle_fold
            )
            self.statusBar().addPermanentWidget(b)
            self.panel_buttons[title] = (b, label)
            self.docks[title].folded_changed.connect(lambda _f: self._panel_labels())
        self._panel_labels()

        self._inspect_at = None
        # the inspector redraws at most every 30 ms while the mouse moves (not after it stops)
        self.inspect_timer = QtCore.QTimer(self, singleShot=True, interval=30)
        self.inspect_timer.timeout.connect(self._inspect)

        self._actions()
        self._menus()
        self._toolbars()
        self._connect()

        self.resize(1480, 940)
        self._restore_layout()
        self.side.setCurrentWidget(self.inspector)  # the Inspector is the tab at the start
        self._place_command_line()
        self._side_buttons()
        self.console.say(f"AstroVista Lite {__version__}. Type help for the commands.")
        self.statusBar().showMessage("Open a FITS image, or File ▸ Open example")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Files dropped on the window: FITS images open; .dmask bitmaps become masks."""
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.open_path(Path(url.toLocalFile()))
        event.acceptProposedAction()

    def _dock(self, title: str, widget, area, keep=None) -> FoldableDock:
        dock = FoldableDock(title, widget, self, keep=keep)
        dock.setAllowedAreas(area)  # it docks only in its own place (or pops out)
        self.addDockWidget(area, dock)
        self.docks[title] = dock
        return dock

    # ---- actions, menus and tool bars
    def _action(self, text, slot, icon=None, shortcut=None, tip=None, checkable=False):
        action = QtGui.QAction(text, self)
        if icon:
            action.setIcon(glyph_icon(icon))
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        action.setToolTip(tip or text.replace("…", "").replace("&", ""))
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        return action

    def _actions(self) -> None:
        a = self._action
        self.a_open = a("&Open FITS image…", self.open_dialog, "📂", QtGui.QKeySequence.Open)
        self.a_example = a("Open &example (UGC 12517)", self.open_example, "★")
        self.a_pair_example = a("Open the &pair example (synthetic)", self.open_pair_example)
        self.a_pair_folder = a("Open a pair &folder…", lambda: self.open_pair_folder())
        self.a_save = a("&Save buffer as FITS…", self.save_buffer, "💾", QtGui.QKeySequence.Save)
        self.a_run_file = a("&Run a command file…", self.run_file)
        self.a_save_commands = a("Save the commands as a file…", self.save_commands)
        self.a_profile = a("Export the &profile (CSV)…", self.export_profile)
        self.a_regions = a("Export the isophotes (DS9 regions)…", self.export_regions)
        self.a_sources = a("Export the &sources (CSV)…", self.export_sources)
        self.a_profile_png = a("Export the profile plot (PNG)…", self.export_profile_png)
        self.a_plot_png = a("Export the sources plot (PNG)…", self.export_plot_png)
        self.a_quit = a("&Quit", self.close, shortcut=QtGui.QKeySequence.Quit)
        self.a_undo = a("&Undo mask edit", lambda: self.run_command("undo"), "↶", "Ctrl+Z")
        self.a_redo = a("&Redo mask edit", lambda: self.run_command("redo"), "↷", "Ctrl+Shift+Z")

        self.modes = QtGui.QActionGroup(self)
        self.mode_actions = {}
        for mode, label, icon, key, tip in TOOLS:
            act = a(
                label, lambda _c=False, m=mode: self.set_mode(m), icon, key, f"{tip} ({key})", True
            )
            if mode == "eraser":
                act.setIcon(eraser_icon())
            self.modes.addAction(act)
            self.mode_actions[mode] = act
        self.mode_actions["pan"].setChecked(True)
        # a toggle beside the mask tools: they unmask instead of masking (as Alt or Shift)
        self.a_unmask = a(
            "Unmask",
            self._unmask_toggled,
            tip="Unmask: the mask tools take pixels out of the mask instead of adding them "
            "(as holding Alt or Shift does); click again to mask",
            checkable=True,
        )
        self.a_unmask.setIcon(icons.icon("unmask"))

        self.a_zoom_in = a("Zoom in", lambda: self.viewer.zoom(1.5), "+", "=")
        self.a_zoom_out = a("Zoom out", lambda: self.viewer.zoom(1 / 1.5), "−", "-")
        self.a_actual = a("Actual pixels (1:1)", self.viewer.actual_size, "1:1", "1")
        self.a_fit_view = a("Whole image", self.viewer.fit_view, "⤢", "0")
        self.a_north = a(
            "North up, East left",
            self.toggle_north_up,
            "N",
            "Ctrl+N",
            "North up and East left (click again for the image as it is)",
            True,
        )
        self.a_rot_left = a("Rotate ⟲ 90°", lambda: self._turn_by_hand(90))
        self.a_rot_right = a("Rotate ⟳ 90°", lambda: self._turn_by_hand(-90))
        self.a_mirror = a("Mirror", lambda: self._turn_by_hand(None))
        self.a_orient_reset = a("Reset the orientation", lambda: self._turn_by_hand(0))
        self.a_invert = a("Invert colours", self.viewer.invert.toggle, checkable=True)
        self.viewer.invert.toggled.connect(self.a_invert.setChecked)
        self.a_hist = a(
            "Histogram and colour bar", self.viewer.histogram_box.toggle, checkable=True
        )
        self.a_hist.setChecked(self.viewer.histogram_box.isChecked())
        self.viewer.histogram_box.toggled.connect(self.a_hist.setChecked)
        self.a_display_bar = a(
            "Display settings bar", lambda on: self.viewer.set_bar_folded(not on), checkable=True
        )
        self.a_display_bar.setChecked(True)
        self.viewer.bar_folded_changed.connect(self._display_bar_folded)
        self.a_goto = a("Go to…", self.go_to, shortcut="Ctrl+G")
        self.a_terminal = a("Command line", self.toggle_terminal, shortcut="Ctrl+`")
        self.a_font_larger = a("Larger", lambda: self.terminal_font(+1), shortcut="Ctrl+Alt+=")
        self.a_font_smaller = a("Smaller", lambda: self.terminal_font(-1), shortcut="Ctrl+Alt+-")
        self.a_font_default = a("Default size", lambda: self.terminal_font(0))

        self.a_fit = a("Elliprof", self.run_fit, None, "Ctrl+E", "Fit the isophotes (ELLIPROF)")
        self.a_fit.setIcon(icons.icon("elliprof"))
        self.a_detect = a(
            "Find sources", self.run_detection, None, "Ctrl+D", "Find the stars and galaxies"
        )
        self.a_detect.setIcon(icons.icon("sources"))
        # open a tool's panel (Tools menu)
        self.a_show_elliprof = a("ELLIPROF settings", lambda: self._show_tool(self.elliprof))
        self.a_show_elliprof.setIcon(icons.icon("elliprof"))
        self.a_show_finder = a("Source finder settings", lambda: self._show_tool(self.finder))
        self.a_show_finder.setIcon(icons.icon("sources"))
        self.a_show_display = a("Display settings", lambda: self._show_tool(self.display))
        self.a_show_display.setIcon(icons.icon("display"))
        self.a_show_contours = a("Contour settings", lambda: self._show_tool(self.contour_panel))
        self.a_show_contours.setIcon(icons.icon("contours"))
        self.a_pick_r0 = a("Pick R0 on the image", lambda: self.pick_radius("r0"))
        self.a_pick_r1 = a("Pick R1 on the image", lambda: self.pick_radius("r1"))
        self.a_contours_apply = a("Draw the contours", self._apply_contours)
        self.a_contours_apply.setIcon(icons.icon("contours"))
        self.a_contours_clear = a("Clear the contours", self._clear_contours)
        self.a_tips = a("Tips", lambda: self.show_tip(+1))
        # the top bar's buttons open the tools; the tools' own buttons run them
        self.tb_elliprof = a(
            "Elliprof",
            lambda: self._show_tool(self.elliprof),
            None,
            None,
            "Open ELLIPROF in the Tools panel",
        )
        self.tb_elliprof.setIcon(icons.icon("elliprof"))
        self.tb_finder = a(
            "Find sources",
            lambda: self._show_tool(self.finder),
            None,
            None,
            "Open the Source finder in the Tools panel",
        )
        self.tb_finder.setIcon(icons.icon("sources"))
        self.tb_display = a(
            "Display", lambda: self._show_tool(self.display), tip="Open Display in the Tools panel"
        )
        self.tb_display.setIcon(icons.icon("display"))
        self.tb_contours = a(
            "Contours",
            lambda: self._show_tool(self.contour_panel),
            tip="Open Contours in the Tools panel",
        )
        self.tb_contours.setIcon(icons.icon("contours"))
        self.a_sky = a("Estimate the sky", self.estimate_sky)
        self.a_skyplane = a("Fit the sky plane", self.run_skyplane)
        self.a_skyplane.setIcon(icons.icon("skyplane"))
        self.a_show_skyplane = a("Sky plane settings", lambda: self._show_tool(self.skyplane))
        self.a_stats = a("Statistics of the image", self.statistics)

        self.a_new_mask = a("New mask layer…", self.layers._new)
        self.a_load_mask = a("Load a mask…", self.layers._load)
        self.a_from_buffer = a("Mask from a buffer…", self.mask_from_buffer)
        self.a_union = a("Union of masks…", lambda: self.combine_masks("union"))
        self.a_intersect = a("Intersection of masks…", lambda: self.combine_masks("intersect"))
        self.a_mask_sources = a("Mask the found sources", self.mask_sources)
        self.a_mask_where = a("Mask the sources where…", self.mask_sources_where)
        self.a_mask_selected = a("Mask the selected source", self.mask_selected_source)
        self.a_mask_segments = a(
            "Mask the found sources' segments (layer segments)…", self.mask_segments
        )
        self.a_mask_nan = a("Mask the pixels that are not finite", self.mask_nan)
        self.a_mask_range = a("Mask a range of values…", self.mask_range)
        self.a_grow = a("Grow the mask…", lambda: self._active_radius("grow"))
        self.a_shrink = a("Shrink the mask…", lambda: self._active_radius("shrink"))
        self.a_invert_mask = a("Invert the mask", lambda: self._active_mask_command("invert"))
        self.a_clear_mask = a("Clear the mask", lambda: self._active_mask_command("clear"))
        self.a_delete_mask = a("Delete the mask", lambda: self._active_mask_command("delete"))

    def _menus(self) -> None:
        def menu(title, items):
            m = self.menuBar().addMenu(title)
            for x in items:
                m.addSeparator() if x is None else m.addAction(x)
            return m

        self.recent_menu = QtWidgets.QMenu("Open &recent", self)
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        menu(
            "&File",
            (
                self.a_open,
                self.recent_menu.menuAction(),
                self.a_example,
                self.a_pair_example,
                self.a_pair_folder,
                None,
                self.a_save,
                None,
                self.a_run_file,
                self.a_save_commands,
                None,
                self.a_profile,
                self.a_regions,
                self.a_sources,
                self.a_profile_png,
                self.a_plot_png,
                None,
                self.a_quit,
            ),
        )
        # "Edit" with a zero-width space: macOS then does not add its own AutoFill, Writing
        # Tools, Dictation and Emoji items to the menu
        menu("&Edit\u200b", (self.a_undo, self.a_redo))
        view = menu(
            "&View",
            (
                self.a_zoom_in,
                self.a_zoom_out,
                self.a_actual,
                self.a_fit_view,
                self.a_goto,
                None,
                self.a_north,
                self.a_rot_left,
                self.a_rot_right,
                self.a_mirror,
                self.a_orient_reset,
                None,
                self.a_invert,
                self.a_hist,
                self.a_display_bar,
                None,
                self.a_terminal,
            ),
        )
        font = view.addMenu("Terminal font size")
        for x in (self.a_font_larger, self.a_font_smaller, self.a_font_default):
            font.addAction(x)
        view.addSeparator()
        for title, dock in self.docks.items():
            view.addAction(f"Fold or unfold: {title}", dock.toggle_fold)
        for title, dock in self.docks.items():
            view.addAction(f"Pop out or back: {title}", dock.toggle_float)
        view.addAction("Reset the layout", self.reset_layout)

        # Tools: ELLIPROF, the Source finder, the display and the contours
        tools = self.menuBar().addMenu("&Tools")
        ellip = tools.addMenu(icons.icon("elliprof"), "ELLIPROF")
        for x in (
            self.a_fit,
            None,
            self.mode_actions["pick"],
            self.a_pick_r0,
            self.a_pick_r1,
            self.a_sky,
            None,
            self.a_show_elliprof,
        ):
            ellip.addSeparator() if x is None else ellip.addAction(x)
        finder = tools.addMenu(icons.icon("sources"), "Source finder")
        for x in (self.a_detect, None, self.a_show_finder):
            finder.addSeparator() if x is None else finder.addAction(x)
        show = tools.addMenu(icons.icon("display"), "Display")
        for x in (
            self.mode_actions["pan"],
            self.mode_actions["crosshair"],
            None,
            self.a_north,
            self.a_invert,
            self.a_hist,
            None,
            self.a_stats,
            None,
            self.a_show_display,
        ):
            show.addSeparator() if x is None else show.addAction(x)
        plane = tools.addMenu(icons.icon("skyplane"), "Sky plane")
        for x in (self.a_skyplane, None, self.a_show_skyplane):
            plane.addSeparator() if x is None else plane.addAction(x)
        contours = tools.addMenu(icons.icon("contours"), "Contours")
        for x in (self.a_contours_apply, self.a_contours_clear, None, self.a_show_contours):
            contours.addSeparator() if x is None else contours.addAction(x)

        # Masks: the drawing tools, the common operations, and the rest under More
        masks = self.menuBar().addMenu("&Masks")
        masks.addAction(self.a_unmask)
        masks.addSeparator()
        for mode in ("circle", "ellipse", "box", "polygon", "brush", "eraser"):
            masks.addAction(self.mode_actions[mode])
        masks.addSeparator()
        for x in (
            self.a_new_mask,
            self.a_load_mask,
            self.a_mask_sources,
            None,
            self.a_undo,
            self.a_redo,
            None,
        ):
            masks.addSeparator() if x is None else masks.addAction(x)
        more = masks.addMenu("More")
        for x in (
            self.a_mask_where,
            self.a_mask_selected,
            self.a_mask_segments,
            self.a_mask_nan,
            self.a_mask_range,
            self.a_from_buffer,
            None,
            self.a_union,
            self.a_intersect,
            None,
            self.a_grow,
            self.a_shrink,
            self.a_invert_mask,
            self.a_clear_mask,
            self.a_delete_mask,
        ):
            more.addSeparator() if x is None else more.addAction(x)

        # "Help" with a zero-width space: macOS then adds no search field to it
        help_menu = self.menuBar().addMenu("&Help\u200b")
        docs = help_menu.addAction("Documentation", lambda: guide.show("overview", self))
        docs.setShortcut(QtGui.QKeySequence.HelpContents)
        topics = help_menu.addMenu("Documentation topics")
        for key, (title, _text) in guide.TOPICS.items():
            topics.addAction(title, lambda k=key: guide.show(k, self))
        help_menu.addAction(self.a_tips)
        help_menu.addSeparator()
        help_menu.addAction("FAQ", lambda: guide.show("faq", self))
        help_menu.addAction("Troubleshooting", lambda: guide.show("problems", self))
        help_menu.addAction("List the commands", lambda: guide.show("allcommands", self))
        help_menu.addSeparator()
        help_menu.addAction("&About AstroVista Lite", self.about)

    def _toolbars(self) -> None:

        def bar(name, actions):
            tb = self.addToolBar(name)
            tb.setObjectName(name)
            tb.setIconSize(QtCore.QSize(22, 22))
            for x in actions:
                tb.addSeparator() if x is None else tb.addAction(x)
            return tb

        bar("File", (self.a_open, self.a_example, self.a_save, None, self.a_undo, self.a_redo))
        # the tools in groups, divided: looking (pan, pick, crosshair), masking (Unmask and
        # the mask tools), then the mask tools' radius and angle
        m = self.mode_actions
        tools = bar(
            "Tools",
            (m["pan"], m["pick"], m["crosshair"], None, self.a_unmask)
            + tuple(m[k] for k in ("circle", "ellipse", "box", "polygon", "brush", "eraser"))
            + (None,),
        )
        tools.addWidget(QtWidgets.QLabel(" r "))
        self.brush = QtWidgets.QDoubleSpinBox()
        self.brush.setRange(0.5, 2000)
        self.brush.setValue(10)
        self.brush.setSuffix(" px")
        self.brush.setToolTip("the radius of a clicked circle, of the brush and of the eraser")
        self.brush.valueChanged.connect(lambda v: setattr(self.viewer, "brush", v))
        tools.addWidget(self.brush)
        tools.addWidget(QtWidgets.QLabel(" angle "))
        self.angle = QtWidgets.QDoubleSpinBox()
        self.angle.setRange(-180, 180)
        self.angle.setSuffix("°")
        self.angle.setToolTip(
            "the angle of the ellipse tool, and the rotation of a shape waiting to be applied "
            "(degrees, counter-clockwise from +x)"
        )
        self.angle.valueChanged.connect(lambda v: setattr(self.viewer, "ellipse_angle", v))
        self.angle.valueChanged.connect(self.viewer.set_pending_angle)
        tools.addWidget(self.angle)
        bar(
            "View",
            (self.a_zoom_in, self.a_zoom_out, self.a_actual, self.a_fit_view, self.a_north),
        )
        run = bar("Run", (self.tb_elliprof, self.tb_finder, self.tb_display, self.tb_contours))
        run.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)  # "Elliprof", "Find sources"

    def _connect(self) -> None:
        self.console.submitted.connect(self.run_command)
        self.console.interrupt.connect(self._interrupt)
        self.console.link.connect(self._terminal_link)
        self.console.complete = lambda line: completions(self.session, line)
        e = self.elliprof
        e.radius_pick_requested.connect(self.pick_radius)
        e.r0.valueChanged.connect(lambda _v: self._overlays())
        e.r1.valueChanged.connect(lambda _v: self._overlays())
        e.run_requested.connect(self.run_fit)
        e.pick_requested.connect(lambda: self.mode_actions["pick"].trigger())
        e.source_centre_requested.connect(self.centre_from_selected)
        e.sky_requested.connect(self.estimate_sky)
        e.mask_sources_requested.connect(self.mask_sources)
        e.x0.valueChanged.connect(lambda _v: self._overlays())
        e.y0.valueChanged.connect(lambda _v: self._overlays())
        self.finder.run_requested.connect(self.run_detection)
        pp = self.pair_panel
        pp.sky_requested.connect(self.pair_sky)
        pp.sky_mask_toggled.connect(self.pair_sky_mask)
        pp.pick_requested.connect(self.pair_pick)
        pp.run_requested.connect(self.run_pair)
        pp.stop_requested.connect(self.stop_pair)
        pp.export_requested.connect(self.export_pair)
        pp.blink_requested.connect(self.blink_pair)
        pp.report_requested.connect(self.copy_pair_report)
        pp.iteration_requested.connect(self.show_pair_iteration)
        pp.objects_requested.connect(self.pair_mask_objects)
        self.skyplane.run_requested.connect(self.run_skyplane)
        self.skyplane.mask_toggled.connect(self.pair_sky_mask)
        pp.changed.connect(self._overlays)
        self.finder.show_sources.toggled.connect(lambda on: self.layers.set_checked("sources", on))

        v = self.viewer
        v.moved.connect(self._show_position)
        v.clicked.connect(self._clicked)
        v.picked.connect(self._centre_picked)
        v.crosshair_moved.connect(self._crosshair_moved)
        v.circle_drawn.connect(
            lambda x, y, r, erase: self._mask_edit(f"circle {{}} {x:.2f} {y:.2f} {r:.2f}", erase)
        )
        v.ellipse_drawn.connect(
            lambda x, y, a, b, t, erase: self._mask_edit(
                f"ellipse {{}} {x:.2f} {y:.2f} {a:.2f} {b:.2f} {t:g}", erase
            )
        )
        v.box_drawn.connect(
            lambda x0, y0, x1, y1, erase: self._mask_edit(
                f"box {{}} {x0:.2f} {y0:.2f} {x1:.2f} {y1:.2f}", erase
            )
        )
        v.polygon_drawn.connect(
            lambda points, erase: self._mask_edit(
                "polygon {} " + " ".join(f"{x:.2f} {y:.2f}" for x, y in points), erase
            )
        )
        v.brush_drawn.connect(self._brushed)
        v.unpatch.connect(self._unpatch)
        v.hover_patch.connect(self._hover_patch)
        v.context.connect(self._context_menu)
        v.view_changed.connect(self._zoom_changed)
        v.welcome_action.connect(self._welcome)
        v.blink_step.connect(self._blink_step)
        v.contrast_changed.connect(self._contrast_dragged)
        v.pending_angle.connect(self._pending_angle)
        v.hint.connect(self.statusBar().showMessage)

        self.profile.row_selected.connect(self._isophote_selected)
        self.sources.source_selected.connect(self._source_selected)
        self.sources.source_activated.connect(lambda _r, _i: self.centre_from_selected())
        self.sources.run_changed.connect(self._run_changed)
        self.plots.source_selected.connect(self._plot_selected)
        self.profile.save_png.connect(self.export_profile_png)
        self.profile.save_csv.connect(self.export_profile)
        self.plots.save_png.connect(self.export_plot_png)
        self.sources.save_csv.connect(self.export_sources)
        self.layers.tip_folded.connect(lambda f: settings().setValue("mask_tip_folded", f))

        self.layers.toggled.connect(self._layer_toggled)
        self.layers.mask_command.connect(self._mask_command)
        self.layers.colour_changed.connect(self._layer_colour)
        self.layers.opacity_changed.connect(self._mask_opacity)
        self.layers.active_changed.connect(self._set_active_mask)
        self.layers.show_only.connect(self._show_only_mask)
        self.layers.from_buffer.connect(self.mask_from_buffer)

        d = self.display
        d.orientation.connect(self._orientation)
        d.north_up.connect(lambda: self.a_north.trigger())
        d.contrast_bias.connect(self._contrast_bias)
        c = self.contour_panel
        c.apply.connect(self._apply_contours)
        c.clear.connect(self._clear_contours)
        c.style_changed.connect(self.viewer.contour_style)
        c.from_display.connect(self._contours_from_display)

    # ---- running commands
    def run_command(self, line: str) -> None:
        """Run a command line (queued while another command runs)."""
        self.viewer.commit_pending()  # a shape waiting to be applied goes first
        stripped = line.strip()
        if stripped.endswith("\\") and not stripped.startswith("!"):  # continued
            self._continued += stripped[:-1] + " "
            self.console.echo(line)
            return
        if self._continued:
            line, self._continued = self._continued + stripped, ""
        self._queue.append(line)
        self._next()

    def terminal_font(self, step: int) -> None:
        """The terminal's text one point larger (+1) or smaller (-1), or back to the
        system's size (0); remembered."""
        if step == 0:
            self.console.set_font_size(0)
            settings().setValue("terminal_font_size", 0)
        else:
            size = self.console.font_size() + step
            self.console.set_font_size(size)
            settings().setValue("terminal_font_size", self.console.font_size())
        self.statusBar().showMessage(f"Terminal text: {self.console.font_size():g} pt")

    def _unmask_toggled(self, on: bool) -> None:
        """The Unmask toggle: the mask tools unmask (on) or mask (off)."""
        self.viewer.set_unmask(on)
        self.statusBar().showMessage(
            "Unmask on: the mask tools take pixels out of the mask (Alt or Shift masks)"
            if on
            else "Unmask off: the mask tools mask (Alt or Shift unmasks)"
        )

    def _pending_angle(self, angle: float) -> None:
        """The pending shape turned (its handle, the keys): the angle box shows it."""
        self.angle.blockSignals(True)
        self.angle.setValue(angle)
        self.angle.blockSignals(False)
        self.viewer.ellipse_angle = angle

    def busy(self) -> bool:
        return self._busy or bool(self._queue)

    def _next(self) -> None:
        if self._busy or not self._queue:
            return
        line = self._queue.pop(0)
        self.console.echo(line)
        if verb_of(line) == "pair":
            self._start_pair(line)
            return
        if verb_of(line) in SLOW or self.session.is_shell(line):
            self._busy = True
            QtWidgets.QApplication.setOverrideCursor(Qt.BusyCursor)
            self.statusBar().showMessage(f"Running: {line}")
            self._enable_runs(False)
            self._running = line
            shell = self.session.is_shell(line)
            threaded = shell or os.environ.get("ASTROVISTA_MAIN_THREAD", "") != "1"
            self._task = Task(lambda: self.session.execute(line), self, threaded=threaded)
            self._task.done.connect(lambda reply: self._finished(line, reply, None))
            self._task.failed.connect(lambda message: self._finished(line, None, message))
            self._task.start()
            return
        try:
            reply = self.session.execute(line)
        except CommandError as e:
            self.console.say(str(e), error=True)
            self.statusBar().showMessage(str(e).splitlines()[0])
        else:
            self.apply(reply)
            if reply.text:
                self.statusBar().showMessage(reply.text.splitlines()[0])
        self._next()

    def _enable_runs(self, on: bool) -> None:
        for action in (self.a_fit, self.a_detect):
            action.setEnabled(on)
        self.elliprof.run.setEnabled(on)
        self.pair_panel.run.setEnabled(on)
        if on:
            self.finder.refresh()
        else:
            self.finder.run.setEnabled(False)

    def _finished(self, line: str, reply: Optional[Reply], error: Optional[str]) -> None:
        self._busy = False
        self._task = None
        self._running = None
        QtWidgets.QApplication.restoreOverrideCursor()
        self._enable_runs(True)
        if error is not None:
            self.console.say(error, error=True)
            self.statusBar().showMessage(error.splitlines()[0])
            self._queue.clear()  # what followed may depend on the failed command
            self.show_terminal()
        else:
            self.apply(reply)
            self.statusBar().showMessage(reply.text.splitlines()[0] if reply.text else "Done")
            verb = verb_of(line)
            if verb == "elliprof":
                self.elliprof.status.setText(reply.text.splitlines()[0])
                self.show_result(self.profile)
            elif verb == "sources":
                self.finder.status.setText(reply.text.splitlines()[0])
                self.show_result(self.sources)
            elif verb == "pair":
                self.show_result(self.pair_results)
        self._next()
        if not self._busy and not self._queue:
            todo, self._when_done = self._when_done, []
            for fn in todo:
                fn()

    def after_commands(self, fn: Callable[[], None]) -> None:
        """Call ``fn`` once the commands queued now have run (now, if none is)."""
        if self._busy or self._queue:
            self._when_done.append(fn)
        else:
            fn()

    def show_pair_tool(self) -> None:
        self.side.setCurrentWidget(self.tools)
        for k in range(self.tools.count()):
            if self.tools.widget(k).widget() is self.pair_panel:
                self.tools.setCurrentIndex(k)

    def apply(self, reply: Reply) -> None:
        """Bring the window up to date with the session after a command."""
        self.console.say(reply.text, rich=reply.html)
        s = self.session
        if "x0" in reply.values:
            self.elliprof.set_centre(reply.values["x0"], reply.values["y0"])
        if "sky" in reply.values:
            self.elliprof.use_sky.setChecked(True)
            self.elliprof.sky.setValue(reply.values["sky"])
        if "buffers" in reply.changed:
            self._refresh_buffers()
        if reply.show is not None:
            self.show_buffer(reply.show)
        elif "data" in reply.changed and self.shown is not None:
            if self.shown not in s.buffers or s.buffers[self.shown] is not self.viewer.image:
                self.show_buffer(self.shown if self.shown in s.buffers else None)
        if "fits" in reply.changed:
            image = self.viewer.image
            self.profile.set_result(s.fits.get(image.key) if image else None)
        if "sources" in reply.changed:
            self.sources.set_runs(s.runs)
        if "sky" in reply.changed or "pairs" in reply.changed or "buffers" in reply.changed:
            self._refresh_pair()
            self._refresh_skyplane()
        if "masks" in reply.changed:
            if self.active_mask not in s.masks:
                self.active_mask = next(reversed(s.masks), None) if s.masks else None
            self._patch_cache = None
            self._refresh_masks()
        self._overlays()

    # ---- buffers
    def _thumbnail(self, image) -> QtGui.QIcon:
        """A small picture of the image (a 99% stretch)."""
        data = image.data[:: max(image.shape[0] // 40, 1), :: max(image.shape[1] // 40, 1)]
        finite = data[np.isfinite(data)]
        if not finite.size:
            return QtGui.QIcon()
        lo, hi = np.percentile(finite, [1, 99])
        t = np.clip((np.nan_to_num(data, nan=lo) - lo) / ((hi - lo) or 1.0), 0, 1)
        grey = np.ascontiguousarray((np.sqrt(t[::-1]) * 255).astype(np.uint8))
        qimage = QtGui.QImage(
            grey.data, grey.shape[1], grey.shape[0], grey.shape[1], QtGui.QImage.Format_Grayscale8
        )
        return QtGui.QIcon(QtGui.QPixmap.fromImage(qimage.copy()))

    def _refresh_buffers(self) -> None:
        self.buffer_list.blockSignals(True)
        self.buffer_list.clear()
        for n, image in sorted(self.session.buffers.items()):
            wcs = "  WCS" if image.wcs() is not None else ""
            made = image.note if len(image.note) <= 40 else image.note[:39] + "…"
            text = f"{n}  {image.label}\n    {image.shape[1]} × {image.shape[0]}{wcs}"
            if made and not image.path:  # how it was made, as a reminder
                text += f"\n    {made}"
            item = QtWidgets.QListWidgetItem(self._thumbnail(image), text)
            item.setData(Qt.UserRole, n)
            tip = [image.name]
            if image.tag:
                tip.insert(0, f"tag: {image.tag}")
            tip.append(f"made by: {image.note}" if image.note else str(image.path or ""))
            item.setToolTip("\n".join(t for t in tip if t))
            self.buffer_list.addItem(item)
            if n == self.shown:
                self.buffer_list.setCurrentItem(item)
        self.buffer_list.blockSignals(False)
        # the blink menu: tags where given, with how each buffer was made as its tooltip
        # the result buffers offered in the tools: the free ones (the occupied on request)
        occupied = {n: b.label for n, b in self.session.buffers.items()}
        for choice in (
            self.elliprof.model_buffer,
            self.elliprof.residual_buffer,
            self.finder.segmentation_buffer,
            *self.skyplane.choices(),
        ):
            choice.set_buffers(occupied)
        self.viewer.set_blink_buffers(
            [
                (n, b.label, f"{b.name}: {b.note}" if b.note else b.name)
                for n, b in sorted(self.session.buffers.items())
            ]
        )
        self._refresh_masks()
        if self.shown not in self.session.buffers:
            self.show_buffer(min(self.session.buffers) if self.session.buffers else None)

    def show_buffer(self, n: Optional[int], keep_limits: bool = False) -> None:
        if self._residual_display is not None and n != self._residual_shown:
            stretch, mode = self._residual_display  # the display before the residual
            self._residual_display = None
            self.viewer.stretch_box.setCurrentText(stretch)
            self.viewer.limit_mode.setCurrentText(mode)
        image = self.session.buffers.get(n) if n is not None else None
        previous = self.viewer.image
        same = previous is not None and image is not None and previous.key == image.key
        self.shown = n if image is not None else None
        self.viewer.set_image(image, keep_view=same, keep_limits=keep_limits)
        if image is not None and image.display == "smooth":
            self._show_smooth(n)
        self.buffer_list.blockSignals(True)
        for k in range(self.buffer_list.count()):
            item = self.buffer_list.item(k)
            if item.data(Qt.UserRole) == self.shown:
                self.buffer_list.setCurrentItem(item)
        self.buffer_list.blockSignals(False)
        fit = self.session.fits.get(image.key) if image else None
        if fit is not None and self.profile.result is not fit:
            self.profile.set_result(fit)
        self.header.show_header(image)
        self.contour_panel.image = image
        self._refresh_masks()
        self._overlays()
        self._zoom_changed()

    def _buffer_chosen(self, item, _previous) -> None:
        if item is not None:
            n = item.data(Qt.UserRole)
            if n != self.shown:
                if self.busy():  # a command is running: show it now, without queueing tv
                    self.show_buffer(n)
                else:
                    self.run_command(f"tv {n}")

    def _buffer_menu(self, pos) -> None:
        item = self.buffer_list.itemAt(pos)
        if item is None:
            return
        n = item.data(Qt.UserRole)
        menu = QtWidgets.QMenu(self)
        menu.addAction("Save as FITS…", lambda: self.save_buffer(n))
        menu.addAction("Statistics", lambda: self.run_command(f"stats {n}"))
        menu.addAction("Tag (give it a name)…", lambda: self._tag_buffer(n))
        menu.addAction(
            "Use as a mask layer", lambda: self._mask_command(f"mask frombuf buffer{n} {n}")
        )
        menu.addAction(
            "Copy to a new buffer",
            lambda: self.run_command(f"copy {n} {self.session.free_buffer()}"),
        )
        menu.addSeparator()
        menu.addAction("Close", lambda: self.run_command(f"close {n}"))
        menu.exec(self.buffer_list.mapToGlobal(pos))

    def _tag_buffer(self, n: int) -> None:
        image = self.session.buffers.get(n)
        if image is None:
            return
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Tag", f"A name for buffer {n}:", text=image.tag or image.name
        )
        if ok and text.strip():
            self.run_command(f'tag {n} "{text.strip()}"')

    def _current_number(self, say: bool = True) -> Optional[int]:
        """The buffer shown; for a model or residual, the image it was made from."""
        image = self.viewer.image
        if image is None:
            if say:
                QtWidgets.QMessageBox.information(self, TITLE, "Open an image first.")
            return None
        if image.family is not None:
            for n, b in self.session.buffers.items():
                if b.name == image.family:
                    return n
        return self.shown

    # ---- files
    def open_dialog(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open FITS images",
            str(self.session.folder),
            "All files (*);;FITS (*.fits *.fit *.fts *.fits.gz *.dmask)",
        )
        for path in paths:
            self.open_path(Path(path))

    def open_path(self, path: Path) -> None:
        """Open a FITS file whatever its name, or one of its images: name.fits[sci],
        name.fits[1], name.fits[sci,2].  A legacy bitmap mask (.dmask) becomes a mask
        layer of the image of its size, or an image when there is none."""
        m = re.fullmatch(r"(.+)\[([^\]]+)\]", str(path))
        if m and not path.exists() and Path(m.group(1)).expanduser().exists():
            n = self.session.free_buffer()
            self.run_command(f"rd {n} {shlex.quote(str(path))}")
            self.remember(str(path))
            return
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, TITLE, f"There is no file {path}.")
            return
        if not is_fits(path):
            QtWidgets.QMessageBox.warning(self, TITLE, f"{path.name} is not a FITS file.")
            return
        if is_bitmap_mask(path):
            shape = read_fits(path).shape
            fitting = [n for n, b in sorted(self.session.buffers.items()) if b.shape == shape]
            if fitting:
                n = self.shown if self.shown in fitting else fitting[0]
                name = re.sub(r"\W", "_", path.name.split(".")[0]) or "mask"
                self._mask_command(f"mask load {name} {shlex.quote(str(path))} {n}")
                self.elliprof.tick(name)
                return
        try:
            hdus = image_hdus(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, TITLE, f"Could not open {path.name}:\n{e}")
            return
        name = str(path)
        if len(hdus) > 1:
            ext = self._choose_hdu(path, hdus)
            if ext is None:
                return
            name += f"[{ext}]"
        waiting = self._queue + ([self._running] if self._running else [])
        queued = {int(m.group(1)) for q in waiting if (m := re.match(r"rd (\d+) ", q))}
        n = self.session.free_buffer()
        while n in queued:
            n += 1
        self.run_command(f"rd {n} {shlex.quote(name)}")
        self.remember(name)

    # ---- recently opened files
    def _recent(self) -> List[str]:
        value = settings().value("recent", []) or []
        return [value] if isinstance(value, str) else list(value)

    def remember(self, path: str) -> None:
        """Put ``path`` at the top of File ▸ Open recent (ten files at most)."""
        recent = [path] + [p for p in self._recent() if p != path]
        settings().setValue("recent", recent[:10])

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        recent = [p for p in self._recent() if Path(p.split("[")[0]).exists()]
        if not recent:
            self.recent_menu.addAction("(nothing yet)").setEnabled(False)
            return
        for p in recent:
            action = self.recent_menu.addAction(Path(p).name, lambda p=p: self.open_path(Path(p)))
            action.setToolTip(p)
        self.recent_menu.addSeparator()
        self.recent_menu.addAction("Clear the list", lambda: settings().setValue("recent", []))

    def _choose_hdu(self, path: Path, hdus: List[int]) -> Optional[int]:
        from astropy.io import fits

        with fits.open(path) as hdul:
            labels = []
            for k in hdus:
                h = hdul[k].header
                shape = " × ".join(str(h.get(f"NAXIS{a}")) for a in range(1, h["NAXIS"] + 1))
                labels.append(f"[{k}] {h.get('EXTNAME', 'PRIMARY' if k == 0 else '')}  {shape}")
        label, ok = QtWidgets.QInputDialog.getItem(
            self, TITLE, f"{path.name} holds several images. Open:", labels, 0, False
        )
        return hdus[labels.index(label)] if ok else None

    def save_buffer(self, n=None) -> None:
        n = n if isinstance(n, int) and not isinstance(n, bool) else self.shown
        if n is None or n not in self.session.buffers:
            return
        image = self.session.buffers[n]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save the buffer",
            str(self.session.folder / (file_stem(image.name) + ".fits")),
            "FITS (*.fits)",
        )
        if path:
            self.run_command(f"wd {n} {shlex.quote(path)}")

    def run_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Run a command file",
            str(self.session.folder),
            "Commands (*.txt *.av);;All files (*)",
        )
        if path:
            self.run_command(f"run {shlex.quote(path)}")

    def save_commands(self) -> None:
        lines = [
            l[2:] for l in self.console.output.toPlainText().splitlines() if l.startswith("› ")
        ]
        if not lines:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the commands", str(self.session.folder / "session.txt"), "Commands (*.txt)"
        )
        if path:
            Path(path).write_text("\n".join(lines) + "\n")
            self.console.say(f"saved {len(lines)} commands in {path}")

    # ---- the pair fit
    def open_pair_example(self) -> None:
        """Load the synthetic pair, its star mask, and the settings that fit it."""
        folder = example_folder()
        n = self.session.free_buffer()
        self.run_command(f"rd {n} {shlex.quote(str(folder / 'pair.fits.gz'))}")
        self.run_command(f"mask load stars {shlex.quote(str(folder / 'pair.dmask'))} {n}")
        self.run_command("mask role stars fit")
        self.run_command(f"sources {n}")
        p = self.pair_panel
        p.cards["A"].set_values(
            {"name": "A", "x0": 294.3, "y0": 263.8, "r0": 3, "r1": 120, "nr": 25}
        )
        p.cards["B"].set_values(
            {"name": "B", "x0": 421.18, "y0": 322.97, "r0": 3, "r1": 50, "nr": 20}
        )
        p.tick("stars")
        p.set_sky_value(1000.0, "typed: the synthetic image's true sky (FITS header SKY)")
        self.after_commands(self.show_pair_tool)
        self.statusBar().showMessage(
            "The synthetic pair is loaded with its true sky: press Measure sky to compare, "
            "then Fit pair"
        )

    def open_pair_folder(self, folder: Optional[Path] = None) -> None:
        """Open a pair's folder: its FITS image, its .dmask masks (as fitting layers), the
        sources, and the two centres of its DS9 region file, with starting radii."""
        if folder is None:
            chosen = QtWidgets.QFileDialog.getExistingDirectory(
                self, "A folder with a FITS image, .dmask masks and a region file of A and B"
            )
            if not chosen:
                return
            folder = Path(chosen)
        try:
            info = pairs.read_folder(folder)
        except pairs.PairError as e:
            QtWidgets.QMessageBox.information(self, TITLE, str(e))
            return
        n = self.session.free_buffer()
        self.run_command(f"rd {n} {shlex.quote(info.rd_name)}")
        names = []
        for path in info.masks:
            name = re.sub(r"\W", "_", path.stem)
            self.run_command(f"mask load {name} {shlex.quote(str(path))} {n}")
            self.run_command(f"mask role {name} fit")
            names.append(name)
        self.run_command(f"sources {n}" + (f" MASK={','.join(names)}" if names else ""))
        p = self.pair_panel
        p.tick(*names)
        if len(info.centres) == 2:
            from astropy.io import fits

            with fits.open(info.image, memmap=True) as f:
                shape = (f[info.hdu] if info.hdu else f[0]).data.shape
            r1 = pairs.suggested_r1(info.centres, shape)
            for cid, (x, y, name) in info.centres.items():
                # start at a third of the final R1 and grow by 1.5 each cycle (starting
                # values to check, not results); fixed centres
                p.cards[cid].set_values(
                    {
                        "name": name or cid,
                        "x0": x,
                        "y0": y,
                        "r0": 3,
                        "r1": r1[cid],
                        "nr": 25,
                        "start": round(r1[cid] / 3, 1),
                        "growth": 0,
                        "growth_value": 1.5,
                        "fixctr": 1,
                        "adaptive": False,
                    }
                )
        p.sky_source.setText("no sky yet")
        self.after_commands(self.show_pair_tool)
        if len(info.centres) == 2:
            self.after_commands(self._snap_pair_centres)

    def _snap_pair_centres(self) -> None:
        """Move each galaxy's centre to the light peak within 3 px (region-file centres are
        approximate hints; with fixed centres an offset leaves a dipole in the core)."""
        image = self.viewer.image
        if image is None:
            return
        for cid, card in self.pair_panel.cards.items():
            x, y = card.centre()
            found = pairs.peak_centre(image.data, x, y)
            if found is not None:
                card.set_centre(*found)
                card.match.setText(
                    f"{cid}: centre at the light peak {found[0]:.2f}, {found[1]:.2f} (the "
                    f"region file said {x:.1f}, {y:.1f})"
                )

    def show_pair_iteration(self, number: int) -> None:
        """Put one call's model, and the image it was fitted on, into buffers and show the
        model (the iteration table's rows)."""
        image = self.viewer.image
        n = self._current_number(say=False)
        record = self.session.pairs.get(image.key) if image is not None else None
        if record is None or n is None or not 1 <= number <= len(record.result.history):
            return
        self.run_command(f"pairstep {n} {number}")

    def copy_pair_report(self) -> None:
        """Put the report of the last pair fit of the image shown on the clipboard."""
        image = self.viewer.image
        record = self.session.pairs.get(image.key) if image is not None else None
        if record is None:
            QtWidgets.QMessageBox.information(self, TITLE, "Fit a pair first.")
            return
        text = pairs.report(record.result, record.job.snap, record.job.line, record.job.measurement)
        QtWidgets.QApplication.clipboard().setText(text)
        self.statusBar().showMessage(
            f"The pair report ({len(text.splitlines())} lines) is on the clipboard: paste it "
            "anywhere (pairreport N file writes it to a file)"
        )

    def pair_sky(self) -> None:
        n = self._current_number()
        if n is not None:
            self.run_command(self.pair_panel.sky_command(n))
            if self.pair_panel.show_mask.isChecked():
                self.run_command("mask show skymask")

    def pair_sky_mask(self, on: bool) -> None:
        """Show or hide the sky-exclusion layer of the last sky measurement."""
        if "skymask" in self.session.masks:
            self.run_command(f"mask {'show' if on else 'hide'} skymask")
        elif on:
            self.statusBar().showMessage("No sky mask yet: press Measure sky")

    def pair_mask_objects(self, grow: float) -> None:
        """The mask layer 'objects' (role fit, ticked in the Pair panel): every source the
        Source finder detects but those at A's and B's centres; finds the sources first if
        the image has none."""
        n = self._current_number()
        if n is None:
            return
        centres = [card.centre() for card in self.pair_panel.cards.values()]
        if not all(x or y for x, y in centres):
            QtWidgets.QMessageBox.information(self, TITLE, "Set A's and B's centres first.")
            return
        image = self.session.buffers[n]
        if not any(r.image_key == image.key for r in self.session.runs):
            self.run_command(f"sources {n}")
        name = "objects"
        if name not in self.session.masks:
            self.run_command(f"mask new {name} {n}")
        xs = ",".join(f"{x:.2f}" for x, _y in centres)
        ys = ",".join(f"{y:.2f}" for _x, y in centres)
        self.run_command(f"mask segments {name} X0={xs} Y0={ys} GROW={grow:g}")
        self.run_command(f"mask role {name} fit")
        self.after_commands(lambda: (self.pair_panel.tick(name), self._overlays()))

    def pair_pick(self, cid: str, what: str) -> None:
        if self.viewer.image is None:
            return
        card = self.pair_panel.cards[cid]
        if what in ("r0", "r1", "start"):
            x0, y0 = card.centre()
            if not (x0 or y0):
                QtWidgets.QMessageBox.information(self, TITLE, f"Set {cid}'s centre first.")
                return
            self.viewer.radius_from = (x0, y0)
        self.set_mode("pick")
        self._pick_target = f"pair:{cid}:{what}"
        self.statusBar().showMessage(
            f"Click {cid}'s centre (Alt-click: exactly where you click)"
            if what == "centre"
            else f"Click at the distance {what.upper()} from {cid}'s centre"
        )

    def _pair_picked(self, x: float, y: float) -> None:
        _p, cid, what = self._pick_target.split(":")
        card = self.pair_panel.cards[cid]
        if what in ("r0", "r1", "start"):
            cx, cy = card.centre()
            r = round(float(np.hypot(x - cx, y - cy)), 2)
            getattr(card, what).setValue(r)
            self.statusBar().showMessage(f"{cid}: {what.upper()} = {r:g}")
            self._pick_target = "centre"
            self.viewer.radius_from = None
            self.set_mode("pan")
            return
        run = self._run_on_shown()
        exact = bool(QtWidgets.QApplication.keyboardModifiers() & Qt.AltModifier)
        said = f"{cid}: centre {x:.2f}, {y:.2f}"
        if run is not None and len(run) and not exact:
            row = run.nearest(x, y)
            sx = float(run.columns["X_IMAGE"][row]) - 0.5
            sy = float(run.columns["Y_IMAGE"][row]) - 0.5
            if np.hypot(sx - x, sy - y) <= 3.0:
                x, y = sx, sy
                said = f"{cid}: snapped to source {int(run.columns['NUMBER'][row])} at {x:.2f}, {y:.2f}"
        card.set_centre(x, y)
        self.statusBar().showMessage(said)
        card.match.setText(said)
        if cid == "A":  # then B's centre
            self._pick_target = "pair:B:centre"
            self.statusBar().showMessage(said + "   — now click B's centre")
        else:  # both placed: picking ends
            self._pick_target = "centre"
            self.set_mode("pan")

    def run_pair(self) -> None:
        n = self._current_number()
        if n is None:
            return
        p = self.pair_panel
        if not p.sky_source.text() or p.sky_source.text() == "no sky yet":
            QtWidgets.QMessageBox.information(
                self,
                TITLE,
                "Choose the sky first: Measure sky, then Use this sky, or Enter a value.",
            )
            return
        if p.sky_id:
            stale = self.session.sky_stale(p.sky_id)
            if stale:
                QtWidgets.QMessageBox.information(
                    self, TITLE, f"The sky measurement is stale ({stale}). Measure it again."
                )
                return
        line = p.command(n)
        note_text = getattr(p, "sky_note", "")
        if not p.sky_id and note_text:
            line += f"   # sky typed: {note_text}"
        self.run_command(line)

    def _start_pair(self, line: str) -> None:
        """Read a pair command here, run it in a worker thread, and take its result back
        here (the window stays alive; Stop ends it after the current elliprof call)."""
        text = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ""
        try:
            job = self.session.pair_prepare(text)
        except CommandError as e:
            self.console.say(str(e), error=True)
            self.statusBar().showMessage(str(e).splitlines()[0])
            self._queue.clear()
            self._next()
            return
        self._busy = True
        self._running = line
        QtWidgets.QApplication.setOverrideCursor(Qt.BusyCursor)
        self._enable_runs(False)
        self.pair_panel.set_running(True)
        self._pair_stop = False
        worker = PairWorker(job, lambda: self._pair_stop, self)
        worker.progress.connect(self._pair_progress)
        worker.done.connect(lambda result: self._pair_done(line, job, result))
        worker.failed.connect(lambda message: self._pair_failed(line, message))
        self._pair_worker = worker
        worker.start()
        self.statusBar().showMessage(f"Running: {line}")

    def _pair_progress(self, info: dict) -> None:
        if info.get("phase") == "call":
            self.pair_panel.add_iteration(info)
            return
        if info.get("phase") == "fit":
            cycle = info["cycle"]
            text = (
                f"Pair • call {info['call']} of at most {info['calls']} • "
                + ("start" if str(cycle).startswith("init") else f"cycle {cycle}")
                + f" • fitting {info['component']} on {info.get('input', 'D')} "
                f"(R1 {info['r1']:.1f})"
            )
            if info.get("retry"):
                text += f" • adaptive retry {info['retry']}"
        else:
            m = info.get("metrics", {})
            text = f"Pair • cycle {info['cycle']} done: " + (
                f"A {'ok' if m.get('A_ok') else 'failed'}, B {'ok' if m.get('B_ok') else 'failed'}"
            )
        if info.get("variant_number"):
            text = (
                f"Variant {info['variant_number']} of {info['variants']} ({info['variant']}) • "
                + text
            )
        elif info.get("variant"):  # the first pass of R1=AUTO
            text = f"{info['variant'][:1].upper()}{info['variant'][1:]} • " + text
        if info.get("sky_iteration"):
            text = (
                f"Sky fit {info['sky_iteration']} of at most {info['sky_iterations']} "
                f"(factor {info['sky_factor']:.5f}) • " + text
            )
        if self._pair_stop:
            text = "stopping after the current elliprof call… " + text
        self.pair_panel.progress.setText(text)
        self.statusBar().showMessage(text)

    def stop_pair(self) -> None:
        if getattr(self, "_pair_worker", None) is None:
            return
        self._pair_stop = True
        self.pair_panel.progress.setText(
            "stopping after the current elliprof call (at most its timeout)…"
        )

    def _pair_done(self, line: str, job, result) -> None:
        self._pair_worker = None
        self.pair_panel.set_running(False)
        self.pair_panel.progress.setText("")
        if job.snap.run_id != result.run_id:
            return  # a result of another run: never shown
        reply = self.session.pair_commit(job, result)
        self.pair_panel.show_result(result)
        if result.sky_iterations:  # the factor of the fit kept
            self.pair_panel.factor.setValue(job.snap.settings.sky_factor)
        self._finished(line, reply, None)
        n = job.outputs.get("TORESID")
        if result.complete and n in self.session.buffers:
            self.show_residual(n)

    def _show_smooth(self, n: int) -> None:
        """A smooth image (a sky plane): linear, from its minimum to its maximum, where a
        percentile or logarithmic stretch would show it as one flat grey; the display is
        restored for the next buffer."""
        data = self.session.buffers[n].data
        finite = data[np.isfinite(data)]
        if not finite.size:
            return
        v = self.viewer
        if self._residual_display is None:
            self._residual_display = (v.stretch_box.currentText(), v.limit_mode.currentText())
        self._residual_shown = n
        v.stretch_box.setCurrentText("Linear")
        lo, hi = float(finite.min()), float(finite.max())
        v.set_limits(lo, hi if hi > lo else lo + 1.0)

    def show_residual(self, n: int) -> None:
        """Show a residual: grey, linear, symmetric about 0 (5 robust sigma each way)."""
        self.show_buffer(n)
        data = self.session.buffers[n].data
        finite = data[np.isfinite(data)]
        if not finite.size:
            return
        sigma = 1.4826 * float(np.median(np.abs(finite - np.median(finite)))) or 1.0
        v = self.viewer
        if self._residual_display is None:  # what to go back to for the next buffer
            self._residual_display = (v.stretch_box.currentText(), v.limit_mode.currentText())
        self._residual_shown = n
        v.stretch_box.setCurrentText("Linear")
        v.set_limits(-5 * sigma, 5 * sigma)

    def _pair_failed(self, line: str, message: str) -> None:
        self._pair_worker = None
        self.pair_panel.set_running(False)
        self.pair_panel.progress.setText("")
        self._finished(line, None, message)

    def _refresh_pair(self) -> None:
        """The pair panel's sky card and the Pair results, for the image shown."""
        image = self.viewer.image
        p = self.pair_panel
        if image is None:
            p.show_sky(None, None)
            self.pair_results.set_record(None)
            return
        sid = self.session.sky_of.get(image.key)
        result = self.session.skies.get(sid) if sid else None
        stale = self.session.sky_stale(sid) if result else None
        values = None
        if result is not None and sid in self.session.sky_buffer:
            measured = self.session.sky_buffer[sid][1]
            if measured.data.shape == result.support.shape:
                values = measured.data[result.support]
        p.show_sky(result, stale, values)
        record = self.session.pairs.get(image.key)
        if record is not self.pair_results.record:
            text = pairs.summary(record.result, record.job.snap) if record else ""
            residual = None
            if record is not None and record.result.residual is not None:
                measured = record.job.measurement
                if measured is not None:
                    use = measured.support & ~record.job.snap.fit_mask
                    residual = record.result.residual[use]
            self.pair_results.set_record(record, text, residual)
            p.show_result(record.result if record else None)
            p.show_history(record.result if record else None)
            if record is not None:  # an automatic R1: show the radius it became
                for cid, card in p.cards.items():
                    comp = record.job.snap.components.get(cid)
                    if comp is not None and card.r1_auto.isChecked():
                        card.r1.blockSignals(True)
                        card.r1.setValue(comp.r1)
                        card.r1.blockSignals(False)
            p.show_residual_sky(
                record.result.sky_check if record else None,
                residual,
                record.result.sky_iterations if record else (),
            )

    def _pair_drawing(self) -> dict:
        """What the pair tool draws on the image shown: the cards' guides (centre, R0 and
        R1, the cores), and the fitted ellipses and centre tracks of the last pair fit."""
        image = self.viewer.image
        p = self.pair_panel
        out = {}
        record = self.session.pairs.get(image.key)
        shown_tool = (
            self.side.currentWidget() is self.tools and self.tools.currentWidget().widget() is p
        )
        shown = self.layers.checked
        for cid, card in p.cards.items():
            d = {}
            if not card.visible.isChecked() or not shown("pair"):
                out[cid] = d
                continue
            x, y = card.centre()
            if (x or y) and (shown_tool or record is None):
                d["centre"] = (x, y)
                radii = (card.r0.value(), card.r1.value(), card.start.value())
                d["guide"] = [(x, y, r, r, 0.0) for r in radii if r > 0]
            if record is not None:
                snap, result = record.job.snap, record.result
                core = snap.cores.get(cid)
                if core is not None and core.shape != "none" and shown_tool:
                    d["core"] = [(core.x, core.y, core.a, core.b, core.angle)]
                commit = result.commits.get(cid)
                if commit is not None and shown("isophotes"):
                    d["fit"] = commit.ellipses
                    d["track"] = [(e[0], e[1]) for e in commit.ellipses]
                    if not d.get("centre"):
                        d["centre"] = (snap.components[cid].x0, snap.components[cid].y0)
            out[cid] = d
        return out

    def export_pair(self) -> None:
        n = self._current_number()
        if n is None:
            return
        if self.session.pairs.get(self.session.buffers[n].key) is None:
            QtWidgets.QMessageBox.information(self, TITLE, "Fit a pair to this image first.")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "A folder for the pair fit", str(self.session.folder)
        )
        if folder:
            target = Path(folder) / f"{self._stem()}_pair"
            self.run_command(f"pairexport {n} {shlex.quote(str(target))}")

    def blink_pair(self) -> None:
        image = self.viewer.image
        if image is None or image.key not in self.session.pairs:
            return
        record = self.session.pairs[image.key]
        order = ("VIEWI", "TOMODEL", "VIEWA", "VIEWB")
        frames = [record.job.outputs[k] for k in order if k in record.job.outputs]
        frames = [n for n in frames if n in self.session.buffers]
        if not frames:
            return
        self.viewer.set_blink_choice(frames)
        self.show_buffer(frames[0])
        self.statusBar().showMessage(
            "Blinking image − sky, A + B, image − sky − B, image − sky − A on one scale: "
            "the arrow keys step through them"
        )

    # ---- the example
    def open_example(self, show_tool: bool = True) -> None:
        """Load the example; ``show_tool`` opens the ELLIPROF tab for it (not at the start,
        when the Inspector is the tab shown)."""
        self._example_ready(show_tool)

    def _example_ready(self, show_tool: bool = True) -> None:
        folder = example_folder()
        n = self.session.free_buffer()
        image = shlex.quote(str(folder / EXAMPLE_FILES[0]))
        mask = shlex.quote(str(folder / EXAMPLE_FILES[1]))
        self.run_command(f"rd {n} {image}")
        self.run_command(f"mask load u12517j {mask} {n}")
        self.elliprof.set_values({**EXAMPLE_FIT, "mask_file": ""})
        self.elliprof.tick("u12517j")
        if show_tool:
            self.side.setCurrentWidget(self.tools)
            self.tools.setCurrentWidget(self.tools.widget(0))  # ELLIPROF
        self.statusBar().showMessage(
            "The example is loaded with elliprof's published parameters and mask: "
            "press Fit isophotes"
        )

    # ---- ELLIPROF
    def run_fit(self) -> None:
        n = self._current_number()
        if n is None:
            return
        s = self.session
        panel = self.elliprof
        # the buffers chosen in the panel, or Auto: a free one (or the one used before)
        model = resid = None
        if panel.make_model.isChecked():
            model = panel.model_buffer.value() or s.output_buffer(n, "model")
        if panel.make_residual.isChecked():
            resid = panel.residual_buffer.value() or s.output_buffer(n, "residual")
            if resid == model and panel.residual_buffer.value() is None:
                resid = s.free_buffer(model)
        if n in (model, resid) or (model is not None and model == resid):
            self.statusBar().showMessage(
                "The model and the residual need two buffers, other than the image's own"
            )
            return
        self.run_command(self.elliprof.command(n, model, resid))

    def _centre_picked(self, x: float, y: float) -> None:
        if str(self._pick_target).startswith("pair:"):
            self._pair_picked(x, y)
            return
        if self._pick_target in ("r0", "r1"):
            cx, cy = self.viewer.radius_from
            r = round(float(np.hypot(x - cx, y - cy)), 2)
            getattr(self.elliprof, self._pick_target).setValue(r)
            self.statusBar().showMessage(f"{self._pick_target.upper()} = {r:g}")
            self._pick_target = "centre"
            self.viewer.radius_from = None
            self.set_mode("pan")
            return
        n = self._current_number(say=False)
        run = self._run_on_shown()
        if self.elliprof.snap.isChecked() and run is not None and n is not None and len(run):
            row = run.nearest(x, y)
            sx, sy, a, _b, _t = run.ellipses()[row]
            if np.hypot(sx - x, sy - y) <= max(a, 3.0):
                number = int(run.columns["NUMBER"][row])
                self.run_command(f"centre {n} RUN={run.number} SOURCE={number}")
                return
        self.elliprof.set_centre(x, y)
        self.statusBar().showMessage(f"Centre X0 = {x:.2f}, Y0 = {y:.2f}")

    def centre_from_selected(self) -> None:
        n = self._current_number()
        if n is None:
            return
        run = self.sources.current()
        index = self.sources.selected_index()
        if run is None or index is None:
            QtWidgets.QMessageBox.information(
                self, TITLE, "Select a source in the Sources table first."
            )
            return
        number = int(run.columns["NUMBER"][index])
        self.run_command(f"centre {n} RUN={run.number} SOURCE={number}")

    def run_skyplane(self) -> None:
        """Tools ▸ Sky plane: fit the plane of the image shown, with the results asked for
        in buffers of their own (Auto: the buffer the same result went into before, or a
        free one)."""
        n = self._current_number()
        if n is None:
            return
        image = self.session.buffers[n]
        names = {
            "TOFLAT": f"{image.name} flat",
            "TOPLANE": f"{image.name} sky plane",
            "TOSUB": f"{image.name} minus sky plane",
        }
        used, outputs = {n}, {}
        for key, chosen in self.skyplane.wanted().items():
            if chosen is None:
                chosen = next(
                    (m for m, b in self.session.buffers.items() if b.name == names[key]), None
                )
            if chosen is None or chosen in used:
                chosen = self.session.free_buffer(n)
                while chosen in used:
                    chosen = self.session.free_buffer(chosen)
            used.add(chosen)
            outputs[key] = chosen
        self._show_tool(self.skyplane)
        self.run_command(self.skyplane.command(n, outputs))

    def _refresh_skyplane(self) -> None:
        """The Sky plane panel: the latest plane of the image shown."""
        image = self.viewer.image
        fit = self.session.planes.get(image.key) if image is not None else None
        stale = None
        if (
            fit is not None
            and image.name == fit.source
            and (image.shape != fit.flat.shape or skies.checksum(image.data) != fit.image_id)
        ):
            stale = "the image has changed since the plane was fitted"
        self.skyplane.show_fit(fit, stale)

    def estimate_sky(self) -> None:
        n = self._current_number()
        if n is None:
            return
        layers = self.elliprof.ticked_layers()
        self.run_command(f"sky {n}" + (f" MASK={','.join(layers)}" if layers else ""))

    def statistics(self) -> None:
        if self.shown is None:
            return
        layers = self.elliprof.ticked_layers()
        self.run_command(f"stats {self.shown}" + (f" MASK={','.join(layers)}" if layers else ""))
        self.show_terminal()

    def _isophote_selected(self, index: int) -> None:
        image = self.viewer.image
        result = self.session.fits.get(image.key) if image else None
        if result is not None and index < len(result.ellipses):
            self.viewer.show_selected(result.ellipses[index])

    # ---- the Source finder
    def run_detection(self) -> None:
        n = self._current_number()
        if n is None:
            return
        toseg = None
        if self.finder.add_segmentation.isChecked():
            toseg = self.finder.segmentation_buffer.value() or self.session.free_buffer(n)
            if toseg == n:
                self.statusBar().showMessage(
                    "The segmentation image needs a buffer other than the image's own"
                )
                return
        self.run_command(self.finder.command(n, toseg))

    def _run_on_shown(self) -> Optional[detection.SourceRun]:
        """The source-finder run shown for the image in the viewer (the one chosen in the
        Sources table, or the latest on this image)."""
        image = self.viewer.image
        if image is None:
            return None
        current = self.sources.current()
        if current is not None and current.image_key == image.key:
            return current
        runs = [r for r in self.session.runs if r.image_key == image.key]
        return runs[-1] if runs else None

    def _run_changed(self, _number: int) -> None:
        self.plots.set_run(self.sources.current())
        self._overlays()

    def _source_selected(self, number: int, index: int) -> None:
        run = self.sources.current()
        if run is None:
            return
        ellipse = run.ellipses()[index]
        self.viewer.show_selected(ellipse)
        self.viewer.centre_on(ellipse[0], ellipse[1])
        self.plots.mark(index)

    def _plot_selected(self, index: int) -> None:
        self.sources.select(index)
        run = self.sources.current()
        if run is not None:
            ellipse = run.ellipses()[index]
            self.viewer.show_selected(ellipse)
            self.viewer.centre_on(ellipse[0], ellipse[1])

    def _clicked(self, x: float, y: float) -> None:
        """A click near a source selects it in the Sources table."""
        run = self._run_on_shown()
        if run is None or run is not self.sources.current() or not len(run):
            return
        index = run.nearest(x, y)
        sx, sy, a, _b, _t = run.ellipses()[index]
        if np.hypot(sx - x, sy - y) <= max(a, 4.0):
            self.sources.select(index)
            self.viewer.show_selected(run.ellipses()[index])
            self.plots.mark(index)
            self.show_result(self.sources)

    # ---- masks
    def set_mode(self, mode: str) -> None:
        if mode != "pick":
            self._pick_target = "centre"
            self.viewer.radius_from = None
        if mode == "crosshair" and self.viewer.crosshair_at is None and self.viewer.image:
            ny, nx = self.viewer.image.shape
            self._place_crosshair(nx / 2, ny / 2)
        if mode in ("circle", "ellipse", "box", "polygon", "brush", "eraser") and self.viewer.image:
            self._ensure_mask()
        self.mode_actions[mode].setChecked(True)
        self.viewer.set_mode(mode)
        tip = next(t[4] for t in TOOLS if t[0] == mode)
        if mode in ("circle", "ellipse", "box", "polygon", "brush", "eraser"):
            tip += f" on '{self.active_mask}'; " + (
                "unmasking (Alt or Shift masks)"
                if self.a_unmask.isChecked()
                else "Alt or Shift unmasks"
            )
        self.statusBar().showMessage(tip)

    def _ensure_mask(self) -> None:
        """The mask tools draw on the selected mask layer; make one when no layer fits
        the image."""
        image = self.viewer.image
        m = self.session.masks.get(self.active_mask)
        if m is not None and m.shape == image.shape:
            return
        fitting = [k for k, v in self.session.masks.items() if v.shape == image.shape]
        if fitting:
            self.active_mask = fitting[-1]
            self._refresh_masks()
            return
        name = "mask"
        while name in self.session.masks:
            name += "_"
        self.run_command(f"mask new {name} {self.shown}")
        self.active_mask = name
        self.elliprof.tick(name)
        self._refresh_masks()

    def _mask_edit(self, template: str, erase: bool) -> None:
        """A drawn shape: the command ``mask <template>`` on the selected mask layer."""
        self._ensure_mask()
        self.run_command("mask " + template.format(self.active_mask) + (" ERASE" if erase else ""))

    def _brushed(self, points, r: float, erase: bool) -> None:
        # keep the path short: points closer than a quarter of the radius add nothing
        kept = [points[0]]
        for x, y in points[1:]:
            if np.hypot(x - kept[-1][0], y - kept[-1][1]) >= r / 4:
                kept.append((x, y))
        if kept[-1] != points[-1]:
            kept.append(points[-1])
        path = " ".join(f"{x:.1f} {y:.1f}" for x, y in kept)
        self._mask_edit(f"brush {{}} {r:g} {path}", erase)

    def _mask_at(self, x: float, y: float) -> Optional[str]:
        """The shown mask layer masking the pixel under (x, y): the selected one first."""
        image = self.viewer.image
        if image is None:
            return None
        i, j = int(np.floor(x)), int(np.floor(y))
        if not (0 <= j < image.shape[0] and 0 <= i < image.shape[1]):
            return None
        names = [self.active_mask] + list(reversed(list(self.session.masks)))
        for name in names:
            m = self.session.masks.get(name)
            if m is not None and m.visible and m.shape == image.shape and m.masked[j, i]:
                return name
        return None

    def _unpatch(self, x: float, y: float) -> None:
        name = self._mask_at(x, y)
        if name is None:
            self.statusBar().showMessage("No mask layer masks that pixel")
            return
        self.viewer.show_patch(None)
        self.run_command(f"mask unpatch {name} {x:.2f} {y:.2f}")

    def _hover_patch(self, x: float, y: float) -> None:
        """While Ctrl is held: outline the masked area a click would remove."""
        if not np.isfinite(x):
            self.viewer.show_patch(None)
            return
        name = self._mask_at(x, y)
        if name is None:
            self.viewer.show_patch(None)
            return
        from scipy import ndimage

        m = self.session.masks[name]
        key = (name, id(m.masked), m.count())
        if self._patch_cache is None or self._patch_cache[0] != key:
            labels, _n = ndimage.label(m.masked, structure=np.ones((3, 3), bool))
            self._patch_cache = (key, labels)
        labels = self._patch_cache[1]
        self.viewer.show_patch(labels == labels[int(np.floor(y)), int(np.floor(x))])

    def mask_sources(self) -> None:
        n = self._current_number()
        if n is None:
            return
        run = self._run_on_shown()
        if run is None:
            QtWidgets.QMessageBox.information(
                self, TITLE, "Find the sources first (the Source finder tab)."
            )
            return
        scale = self.elliprof.mask_scale.value()
        x0, y0 = self.elliprof.x0.value(), self.elliprof.y0.value()  # the galaxy kept
        self.run_command(f"mask new sources {n}")
        self.run_command(
            f"mask sources sources RUN={run.number} SCALE={scale:g} X0={x0:g} Y0={y0:g}"
        )
        self.active_mask = "sources"
        self.elliprof.tick("sources")

    def mask_sources_where(self) -> None:
        n = self._current_number()
        run = self._run_on_shown()
        if n is None or run is None:
            QtWidgets.QMessageBox.information(self, TITLE, "Detect sources first.")
            return
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Mask the sources where",
            "A condition on the catalogue's columns, e.g.\n" "FWHM_IMAGE < 3 and MAG_AUTO < 20",
            text="FWHM_IMAGE < 3",
        )
        if ok and text.strip():
            self._ensure_mask()
            scale = self.elliprof.mask_scale.value()
            x0, y0 = self.elliprof.x0.value(), self.elliprof.y0.value()
            self.run_command(
                f"mask sources {self.active_mask} RUN={run.number} SCALE={scale:g} "
                f"X0={x0:g} Y0={y0:g} WHERE={shlex.quote(text.strip())}"
            )

    def mask_selected_source(self) -> None:
        run, index = self.sources.current(), self.sources.selected_index()
        if run is None or index is None or self.viewer.image is None:
            QtWidgets.QMessageBox.information(self, TITLE, "Select a source first.")
            return
        self._ensure_mask()
        number = int(run.columns["NUMBER"][index])
        scale = self.elliprof.mask_scale.value()
        self.run_command(
            f"mask sources {self.active_mask} RUN={run.number} SCALE={scale:g} KEEP=none "
            f"IDS={number}"
        )

    def mask_segments(self) -> None:
        run = self._run_on_shown()
        if run is None or self.viewer.image is None:
            QtWidgets.QMessageBox.information(self, TITLE, "Detect sources first.")
            return
        grow, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "Mask the sources' segments",
            "Grow each source by (pixels):",
            2.0,
            0.0,
            200.0,
            1,
        )
        if ok:
            # its own layer: the segments replace what it masked, so not the active layer
            name = "segments"
            if name not in self.session.masks:
                self.run_command(f"mask new {name} {self.shown}")
            x0, y0 = self.elliprof.x0.value(), self.elliprof.y0.value()
            self.run_command(
                f"mask segments {name} RUN={run.number} GROW={grow:g} X0={x0:g} Y0={y0:g}"
            )

    def mask_nan(self) -> None:
        if self._current_number() is None:
            return
        self._ensure_mask()
        self.run_command(f"mask nan {self.active_mask} {self.shown}")

    def mask_range(self) -> None:
        if self.viewer.image is None:
            return
        lo, hi = self.viewer._limits
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Mask a range of values",
            f"Mask the pixels of buffer {self.shown} from LOW to HIGH:",
            text=f"{hi:.6g} 1e30",
        )
        if ok:
            words = text.replace(",", " ").split()
            if len(words) == 2:
                self._ensure_mask()
                self.run_command(
                    f"mask range {self.active_mask} {self.shown} {words[0]} {words[1]}"
                )

    def mask_from_buffer(self) -> None:
        if not self.session.buffers:
            return
        items = [f"{n}  {b.name}" for n, b in sorted(self.session.buffers.items())]
        label, ok = QtWidgets.QInputDialog.getItem(
            self, "Mask from a buffer", "Pixels that are 0 or NaN are masked:", items, 0, False
        )
        if ok:
            n = label.split()[0]
            self._mask_command(f"mask frombuf buffer{n} {n}")

    def combine_masks(self, how: str) -> None:
        names = list(self.session.masks)
        if len(names) < 2:
            QtWidgets.QMessageBox.information(self, TITLE, "Make two mask layers or more first.")
            return
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            f"{how.capitalize()} of masks",
            f"The layers to combine (from {', '.join(names)}), then the new layer's name:",
            text=f"{' '.join(names[:2])} -> {how}",
        )
        if ok and "->" in text:
            sources, target = text.split("->", 1)
            self._mask_command(f"mask {how} {target.strip()} {sources.strip()}")

    def _active_radius(self, action: str) -> None:
        if self.active_mask:
            self.layers._radius(self.active_mask, action)

    def _active_mask_command(self, action: str) -> None:
        if self.active_mask:
            self.run_command(f"mask {action} {self.active_mask}")

    def load_mask_file(self, path: Path) -> None:
        """A mask file (FITS or .dmask, 0 = masked) as a mask layer of its own size, also
        shown as a buffer, ticked for the fit (``astrovista -m``)."""
        path = Path(path).expanduser()
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, TITLE, f"There is no file {path}.")
            return
        name = re.sub(r"\W", "_", path.name.split(".")[0]) or "mask"
        while name in self.session.masks:  # two files of the same name: keep both
            name += "_2"
        self.run_command(f"mask load {name} {shlex.quote(str(path))}")
        self.active_mask = name
        self.elliprof.tick(name)
        self._refresh_masks()

    def _mask_command(self, line: str) -> None:
        words = shlex.split(line)
        if line.startswith(("mask new ", "mask load ")) and self.shown is not None:
            if len(words) == (3 if words[1] == "new" else 4):
                line += f" {self.shown}"
        self.run_command(line)
        if len(words) > 2 and words[1] in ("new", "load", "copy", "frombuf", "union", "intersect"):
            self.active_mask = words[2]
            if words[1] in ("load", "frombuf"):
                self.elliprof.tick(words[2])  # a mask brought in is meant for the fit
            self._refresh_masks()

    def _mask_opacity(self, name: str, opacity: float) -> None:
        for m in self.session.masks.values():
            if m.name == name or not name:
                m.opacity = opacity
        self._refresh_masks()

    def _show_only_mask(self, name: str) -> None:
        for m in self.session.masks.values():
            m.visible = m.name == name
        self._refresh_masks()

    def _set_active_mask(self, name: str) -> None:
        self.active_mask = name
        if self.viewer.mode in ("circle", "ellipse", "box", "polygon", "brush", "eraser"):
            self.set_mode(self.viewer.mode)
        self._refresh_masks()

    def _refresh_masks(self) -> None:
        image = self.viewer.image
        masks = list(self.session.masks.values())
        self.layers.set_masks(masks, self.active_mask)
        fitting = [m for m in masks if image is not None and m.shape == image.shape]
        # the buffers of the image's size, except the image and what was made from it
        buffers = [
            (n, b.label)
            for n, b in sorted(self.session.buffers.items())
            if image is not None and b.shape == image.shape and b.key != image.key
        ]
        self.elliprof.set_mask_layers([m.name for m in fitting], buffers)
        self.pair_panel.set_mask_layers(fitting)
        self._refresh_pair()
        self._refresh_skyplane()
        self.viewer.show_masks([m for m in fitting if m.visible])

    # ---- the command line
    def toggle_terminal(self) -> None:
        """Open the terminal (or bring a popped-out one forward), or minimise it."""
        dock = self.docks["Terminal"]
        if dock.isFloating():
            dock.raise_()
        else:
            dock.set_folded(not dock.folded)
        self.console.line.setFocus()

    def _fold_terminal(self, folded: bool) -> None:
        """Minimise the terminal out of sight (its command line goes to the bar at the
        bottom, whose Terminal button opens it again), or open it."""
        dock = self.docks["Terminal"]
        if folded:
            dock.hide()
        else:
            dock.show()
            dock.setMinimumHeight(0)
            dock.setMaximumHeight(16777215)
        dock.bar.update_button()

    def _fold_side(self, folded: bool) -> None:
        """Fold the right panel out of sight (the Tools button in the bottom bar opens it
        again), or open it."""
        dock = self.docks["Tools"]
        dock.setVisible(not folded)
        self._side_buttons()

    def _side_floating(self, floating: bool) -> None:
        """A popped-out panel gets a title bar to move it by (and to put it back)."""
        self.docks["Tools"].setTitleBarWidget(None if floating else self._side_title)
        self._side_buttons()

    def _side_buttons(self) -> None:
        dock = self.docks["Tools"]
        self.side_pop.setText("⤓" if dock.isFloating() else "⧉")
        self.side_pop.setToolTip(
            "Put the panel back in its place"
            if dock.isFloating()
            else "Pop the panel out as a window of its own (for another monitor)"
        )
        self.side_fold.setVisible(not dock.isFloating())
        self.side_fold.setText("▸")
        self.side_fold.setToolTip("Fold the panel (the Tools button in the bottom bar opens it)")

    def _show_tool(self, panel) -> None:
        """Bring a tool's settings into view in the right panel."""
        self.docks["Tools"].set_folded(False)
        self.side.setCurrentWidget(self.tools)
        for k in range(self.tools.count()):
            if self.tools.widget(k).widget() is panel:
                self.tools.setCurrentIndex(k)

    def show_tip(self, step: int) -> None:
        """Help ▸ Tips: one short tip at a time."""
        from .guide import TIP_LIST

        self._tip = (getattr(self, "_tip", -1) + step) % len(TIP_LIST)
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Tips")
        box.setIconPixmap(app_icon().pixmap(48, 48))
        box.setText(f"<b>Tip {self._tip + 1} of {len(TIP_LIST)}</b>")
        box.setInformativeText(TIP_LIST[self._tip])
        next_tip = box.addButton("Next tip", QtWidgets.QMessageBox.ActionRole)
        box.addButton(QtWidgets.QMessageBox.Close)
        box.exec()
        if box.clickedButton() is next_tip:
            self.show_tip(+1)

    def show_result(self, widget) -> None:
        """Bring a result (the profile, the sources...) into view in the right panel."""
        self.side.setCurrentWidget(self.results)
        self.results.setCurrentWidget(widget)

    def show_terminal(self) -> None:
        dock = self.docks["Terminal"]
        if dock.folded:
            dock.set_folded(False)

    def _place_command_line(self) -> None:
        """The command line belongs to the terminal while it is open (or popped out), as
        in a shell; while it is minimised, it waits in the bar at the bottom."""
        dock = self.docks["Terminal"]
        inside = dock.isFloating() or not dock.folded
        row = self.console.input_row
        if inside and row.parent() is not self.console.surface:
            self.statusBar().removeWidget(row)
            self.console.take_line(True)
        elif not inside and row.parent() is self.console.surface:
            self.console.take_line(False)
            self.statusBar().insertPermanentWidget(0, row, 10)
            row.show()

    def _panel_labels(self) -> None:
        for title, (b, label) in self.panel_buttons.items():
            b.setText(f"{label} {'▴' if self.docks[title].folded else '▾'}")

    def _display_bar_folded(self, folded: bool) -> None:
        """The display bar above the image folded or unfolded: the menu follows, and the
        next start remembers it."""
        self.a_display_bar.setChecked(not folded)
        settings().setValue("display_bar_folded", folded)

    def reset_layout(self) -> None:
        """The panels where they are at the first start, unfolded, at their first sizes."""
        self.viewer.set_bar_folded(False)
        for dock in self.docks.values():
            if dock.isFloating():
                dock.setFloating(False)
            dock.set_folded(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.docks["Buffers and layers"])
        self.addDockWidget(Qt.RightDockWidgetArea, self.docks["Tools"])
        self.addDockWidget(Qt.BottomDockWidgetArea, self.docks["Terminal"])
        QtCore.QTimer.singleShot(0, self._default_sizes)

    def _default_sizes(self) -> None:
        """Most of the window for the image; the panels narrow (they can be widened)."""
        self.resizeDocks([self.docks["Terminal"]], [150], Qt.Vertical)
        self.resizeDocks(
            [self.docks["Buffers and layers"], self.docks["Tools"]], [230, 430], Qt.Horizontal
        )

    def _save_layout(self) -> None:
        for dock in self.docks.values():  # sizes are saved unfolded: panels open next time
            dock.set_folded(False)
        store = settings()
        store.setValue("layout/geometry", self.saveGeometry())
        store.setValue("layout/state", self.saveState(LAYOUT_VERSION))

    def _restore_layout(self) -> None:
        store = settings()
        # the display bar as it was left (folded or not)
        self.viewer.set_bar_folded(str(store.value("display_bar_folded", False)).lower() == "true")
        # the unmasking tip starts folded to its bulb, until it is unfolded
        self.layers.set_tip_folded(str(store.value("mask_tip_folded", True)).lower() == "true")
        size = float(store.value("terminal_font_size", 0) or 0)
        if size > 0:
            self.console.set_font_size(size)
        geometry, state = store.value("layout/geometry"), store.value("layout/state")
        if geometry is not None and state is not None:
            self.restoreGeometry(geometry)
            if self.restoreState(state, LAYOUT_VERSION):
                for dock in self.docks.values():  # every panel open at the start
                    dock.show()
                return
        self._default_sizes()

    def closeEvent(self, event):
        """Closing the main window ends the program: the layout is saved, then the
        documentation, the popped-out panels and any other window close with it."""
        for dock in self.docks.values():
            if dock.isFloating():
                dock.setFloating(False)  # back in place (and so saved there next time)
        self._save_layout()
        guide.close_window()
        for widget in QtWidgets.QApplication.topLevelWidgets():
            if widget is not self and widget.isWindow() and widget.isVisible():
                if not isinstance(widget, MainWindow):  # another main window stays open
                    widget.close()
        super().closeEvent(event)

    def _terminal_link(self, link: str) -> None:
        """A click in a listing: into a folder (and list it), or open a FITS file."""
        kind, _, path = link.partition(":")
        if kind == "cd":
            self.run_command(f"cd {shlex.quote(path)}")
            self.run_command("ls")
        elif kind == "open":
            self.open_path(Path(path))
        self.console.line.setFocus()

    def _interrupt(self) -> None:
        if self.session.interrupt():
            self.console.say("^C")
        elif self._queue:
            self._queue.clear()
            self.console.say("^C (the commands waiting were dropped)")

    def _welcome(self, key: str) -> None:
        if key == "example":
            self.open_example()
        elif key == "open":
            self.open_dialog()
        else:
            guide.show("lesson", self)

    # ---- picking the centre and the radii
    def pick_radius(self, key: str) -> None:
        x0, y0 = self.elliprof.x0.value(), self.elliprof.y0.value()
        if self.viewer.image is None:
            return
        if not (x0 or y0):
            QtWidgets.QMessageBox.information(self, TITLE, "Set the centre (X0, Y0) first.")
            return
        self._pick_target = key
        self.viewer.radius_from = (x0, y0)
        self.set_mode("pick")
        self.statusBar().showMessage(f"Click at the distance {key.upper()} from the centre")

    def _contrast_dragged(self, contrast: float, bias: float) -> None:
        self.display.show_contrast_bias(contrast, bias)
        self.statusBar().showMessage(f"contrast {contrast:.2f}   bias {bias:.2f}")

    # ---- layers and display
    def _layer_toggled(self, key: str, on: bool) -> None:
        if key.startswith("mask:"):
            m = self.session.masks.get(key[5:])
            if m is not None:
                m.visible = on
                self._refresh_masks()
        elif key == "contours":
            self.viewer.show_contours(on)
        elif key == "crosshair":
            if on and self.viewer.mode != "crosshair" and self.viewer.image is not None:
                self.set_mode("crosshair")  # it follows the mouse
            elif not on:
                self.viewer.set_crosshair(None, None)
                if self.viewer.mode == "crosshair":
                    self.set_mode("pan")
        else:
            if key == "sources":
                self.finder.show_sources.blockSignals(True)
                self.finder.show_sources.setChecked(on)
                self.finder.show_sources.blockSignals(False)
            self._overlays()

    def _say_no_wcs(self, on: bool) -> None:
        if on and self.viewer.image is not None and self.viewer.image.wcs() is None:
            self.statusBar().showMessage("This image has no celestial WCS: no compass")

    def _layer_colour(self, key: str, colour: str) -> None:
        if key.startswith("mask:"):
            m = self.session.masks.get(key[5:])
            if m is not None:
                m.colour = colour
                self._refresh_masks()
        elif key == "centre":
            self.viewer.centre.setPen(pg_pen(colour))
        else:
            self.viewer.set_layer_colour(key, colour)

    def _apply_contours(self) -> None:
        panel = self.contour_panel
        if self.viewer.image is None:
            return
        try:
            levels = panel.values()
        except ValueError:
            panel.status.setText("The values must be numbers, one per line.")
            return
        if not levels:
            panel.generate_requested()
            levels = panel.values()
        self.viewer.set_contours(levels, panel.method.currentText(), panel.smoothness.value())
        self.viewer.contour_style(panel.colour, panel.width.value(), panel.dashed.isChecked())
        if not self.layers.checked("contours"):
            self.layers.set_checked("contours", True)  # shows them
        else:
            self.viewer.show_contours(True)
        how = f"{panel.method.currentText().lower()} {panel.smoothness.value()} px"
        panel.status.setText(f"{len(levels)} levels ({how}) on {self.viewer.image.name}")

    def _clear_contours(self) -> None:
        self.viewer.set_contours([])
        self.contour_panel.status.setText("")

    def _contours_from_display(self) -> None:
        panel, v = self.contour_panel, self.viewer
        panel.scale.setCurrentText(v.stretch_box.currentText())
        panel.exponent.setValue(v.exponent.value())
        mode = v.limit_mode.currentText()
        panel.limit_mode.setCurrentText("User" if mode == "Manual" else mode)
        panel.set_limits(*v._limits)
        panel.generate_requested()

    def _contrast_bias(self, contrast: float, bias: float) -> None:
        self.viewer.contrast, self.viewer.bias = contrast, bias
        self.viewer.apply_colours()

    def _orientation(self, rotation: float, mirror: bool) -> None:
        """A rotation or mirror chosen by hand: no longer 'North up'."""
        self.viewer.set_orientation(rotation, mirror)
        self.a_north.setChecked(False)  # (North up is set by the viewer, not from here)

    def _turn_by_hand(self, how) -> None:
        self.a_north.setChecked(False)
        if how is None:
            self.display._flip()
        elif how == 0:
            self.display._set(0.0, False)
        else:
            self.display._turn(how)

    def toggle_north_up(self, on: bool) -> None:
        """On: North up, East left; off: the image as it is stored."""
        if on:
            self.north_up()
        else:
            self.display._set(0.0, False)
            self.statusBar().showMessage("The image as it is stored")

    def north_up(self) -> None:
        if self.viewer.image is None or not self.viewer.north_up():
            self.a_north.setChecked(False)
            self.statusBar().showMessage(
                "This image has no celestial WCS" if self.viewer.image else "Open an image first"
            )
            return
        self.a_north.setChecked(True)
        self.display.show_orientation(self.viewer.rotation, self.viewer.mirror)
        self.statusBar().showMessage("North up, East left (click N again to undo)")

    def _blink_step(self, step: int) -> None:
        """The arrow keys: the previous (-1) or next (+1) buffer of the blink list."""
        frames = [n for n in self.viewer.blink_frames() if n in self.session.buffers]
        if len(frames) < 2:
            self.statusBar().showMessage("Blinking needs two buffers or more")
            return
        k = frames.index(self.shown) + step if self.shown in frames else 0
        n = frames[k % len(frames)]
        # zoomed in: the next buffer is shown at the same place (on the sky when both have
        # a WCS that matches, else at the same pixel) and zoom
        lock = self.viewer.view_lock()
        # with "same scale", the next buffer keeps the limits of the one shown
        self.show_buffer(n, keep_limits=self.viewer.same_scale.isChecked())
        how = ""
        if lock is not None:
            mode = self.viewer.apply_lock(lock)
            how = "  locked on the WCS" if mode == "wcs" else "  locked on the image"
        self.statusBar().showMessage(f"Blink: buffer {n}  ({self.session.buffers[n].name}){how}")

    def go_to(self) -> None:
        image = self.viewer.image
        if image is None:
            return
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Go to",
            "A pixel position 'x y', or RA Dec ('12:30:49.4 +12:23:28' or degrees):",
        )
        if not ok or not text.strip():
            return
        words = text.replace(",", " ").split()
        try:
            if len(words) == 2 and all(re.fullmatch(r"[-+]?[\d.]+(e[-+]?\d+)?", w) for w in words):
                x, y = float(words[0]), float(words[1])
                w = image.wcs()
                if w is not None and not (0 <= x <= image.shape[1] and 0 <= y <= image.shape[0]):
                    raise ValueError("outside the image")
            else:
                from astropy import units as u
                from astropy.coordinates import SkyCoord

                w = image.wcs()
                if w is None:
                    raise ValueError("this image has no WCS")
                unit = (u.hourangle, u.deg) if ":" in text or "h" in text else (u.deg, u.deg)
                sky = SkyCoord(" ".join(words), unit=unit)
                px, py = w.world_to_pixel(sky)
                x, y = float(px) + 0.5, float(py) + 0.5
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, TITLE, f"Cannot go to {text!r}: {e}")
            return
        self.viewer.centre_on(x, y)
        self._place_crosshair(x, y)

    def _context_menu(self, x: float, y: float, where: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        menu.addAction(
            f"Copy  {x:.2f} {y:.2f}",
            lambda: QtWidgets.QApplication.clipboard().setText(f"{x:.2f} {y:.2f}"),
        )
        w = self.viewer.image.wcs() if self.viewer.image else None
        if w is not None:
            sky = w.pixel_to_world(x - 0.5, y - 0.5).to_string("hmsdms", precision=2)
            menu.addAction(f"Copy  {sky}", lambda: QtWidgets.QApplication.clipboard().setText(sky))
        menu.addSeparator()
        menu.addAction("Set the ELLIPROF centre here", lambda: self.elliprof.set_centre(x, y))
        menu.addAction("Centre the view here", lambda: self.viewer.centre_on(x, y))
        menu.addAction(
            "Crosshair here",
            lambda: (self.viewer.set_crosshair(x, y), self.layers.set_checked("crosshair", True)),
        )
        menu.addSeparator()
        menu.addAction(
            f"Mask a circle here (r {self.viewer.brush:g})",
            lambda: self._mask_edit(f"circle {{}} {x:.2f} {y:.2f} {self.viewer.brush:g}", False),
        )
        if self._mask_at(x, y):
            menu.addAction("Remove the masked area here", lambda: self._unpatch(x, y))
        menu.exec(where)

    # ---- overlays and the status bar
    def _overlays(self) -> None:
        image = self.viewer.image
        result = self.session.fits.get(image.key) if image else None
        show = self.layers.checked
        self.viewer.show_isophotes(result.ellipses if result and show("isophotes") else [])
        run = self._run_on_shown()
        if run is not None and show("sources"):
            self.viewer.show_sources(run.ellipses())
        else:
            self.viewer.show_sources([])
        x, y = self.elliprof.x0.value(), self.elliprof.y0.value()
        if image is not None and (x or y) and show("centre"):
            self.viewer.show_centre(x, y)
            self.viewer.show_radii(x, y, (self.elliprof.r0.value(), self.elliprof.r1.value()))
        else:
            self.viewer.show_centre(None, None)
            self.viewer.show_radii(None, None, ())
        self.viewer.show_pair(self._pair_drawing() if image is not None else {})

    def _show_position(self, x: float, y: float) -> None:
        image = self.viewer.image
        if image is None:
            return
        i, j = int(np.floor(x)), int(np.floor(y))
        text = f"buffer {self.shown}   x {x:.1f}  y {y:.1f}"
        if 0 <= j < image.shape[0] and 0 <= i < image.shape[1]:
            text += f"   pixel {i + 1}, {j + 1}   value {image.data[j, i]:.6g}"
            w = image.wcs()
            if w is not None:
                try:
                    sky = w.pixel_to_world(x - 0.5, y - 0.5)  # 0-based pixel centres
                    text += "   " + sky.to_string("hmsdms", precision=2)
                except Exception:
                    pass
        self.position.setText(text)
        if self.viewer.mode != "crosshair":  # with the crosshair, it follows that
            self._inspect_at = (x, y)
            if not self.inspect_timer.isActive():
                self.inspect_timer.start()

    def _place_crosshair(self, x: float, y: float) -> None:
        self.viewer.set_crosshair(x, y)
        self._crosshair_moved(x, y)

    def _crosshair_moved(self, x: float, y: float) -> None:
        if not self.layers.checked("crosshair"):
            self.layers.set_checked("crosshair", True)
        self._inspect_at = (x, y)
        self._inspect()
        self._show_position(x, y)

    def _inspect(self) -> None:
        if self._inspect_at is None:
            return
        if self.side.currentWidget() is self.inspector and not self.docks["Tools"].folded:
            self.quick.show_at(self.viewer, self.shown, *self._inspect_at)

    def _zoom_changed(self) -> None:
        if self.viewer.image is not None:
            self.zoom_label.setText(f"zoom {self.viewer.zoom_factor():.3g}×")

    # ---- exports
    def _stem(self) -> str:
        """The file name stem of the image the shown buffer was made from."""
        image = self.viewer.image
        return file_stem(image.family or image.name) if image is not None else "image"

    def _fit_or_say(self):
        image = self.viewer.image
        result = self.session.fits.get(image.key) if image else None
        if result is None:
            QtWidgets.QMessageBox.information(self, TITLE, "Fit isophotes to this image first.")
        return result

    def export_profile(self) -> None:
        result = self._fit_or_say()
        if result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "The profile",
            str(self.session.folder / f"{self._stem()}_profile.csv"),
            "CSV (*.csv)",
        )
        if path:
            self.profile.write_csv(path, result.profile)  # comma separated, columns aligned
            self.console.say(f"profile written to {path}")

    def export_regions(self) -> None:
        result = self._fit_or_say()
        if result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "The isophotes",
            str(self.session.folder / f"{self._stem()}_isophotes.reg"),
            "DS9 (*.reg)",
        )
        if path:
            from elliprof import write_ds9_regions

            write_ds9_regions(result.profile, path)
            self.console.say(f"isophotes written to {path}")

    def export_sources(self) -> None:
        run = self.sources.current()
        if run is None:
            QtWidgets.QMessageBox.information(self, TITLE, "Detect sources first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "The sources",
            str(self.session.folder / f"{file_stem(run.image_name)}_sources_{run.number}.csv"),
            "CSV (*.csv)",
        )
        if path:
            from .exports import aligned_csv

            aligned_csv(path, run.columns)  # comma separated, every column aligned
            self.console.say(f"sources of run {run.number} written to {path}")

    def export_profile_png(self) -> None:
        """Results ▸ Profile's plot as a PNG figure (matplotlib)."""
        result = self.profile.result
        if result is None:
            QtWidgets.QMessageBox.information(self, TITLE, "Fit isophotes first.")
            return
        image = self.viewer.image
        stem = file_stem(image.name) if image else "profile"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "The profile plot",
            str(self.session.folder / f"{stem}_profile.png"),
            "PNG (*.png)",
        )
        if path:
            self.profile.write_png(path, title=image.label if image else "")
            self.console.say(f"the profile plot written to {path}")

    def export_plot_png(self) -> None:
        """Results ▸ Plots' scatter plot of the sources as a PNG figure (matplotlib)."""
        run = self.plots.run
        if run is None:
            QtWidgets.QMessageBox.information(self, TITLE, "Detect sources first.")
            return
        x, y = self.plots.x.currentText(), self.plots.y.currentText()
        name = f"{file_stem(run.image_name)}_{x}_{y}_{run.number}.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "The plot", str(self.session.folder / name), "PNG (*.png)"
        )
        if path:
            self.plots.write_png(path, title=f"{run.image_name}, run {run.number}")
            self.console.say(f"the plot written to {path}")

    def about(self) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("About AstroVista Lite")
        box.setIconPixmap(app_icon().pixmap(96, 96))
        box.setText(f"<h3>AstroVista Lite {__version__}</h3>")
        box.setInformativeText(
            "<p>The lite edition of AstroVista, an astronomical image analysis studio in "
            "development: isophote fitting, a source finder, masks, image arithmetic and "
            "image commands in one window. Made for everyday tasks and for teaching.</p>"
            "<p>Built on ELLIPROF and the command language of MONSTA, by John Tonry.</p>"
            "<p>Ehsan Kourkchi (Edwin Kay)</p>"
        )
        box.exec()


def pg_pen(colour: str):
    import pyqtgraph as pg

    return pg.mkPen(colour, width=2)


def make_parser() -> argparse.ArgumentParser:
    """The command line: astrovista [images] [-i images] [-m masks] [--example] [--run file]."""
    parser = argparse.ArgumentParser(
        prog="astrovista",
        description=TITLE,
        epilog="example: astrovista -i galaxy.fits other.fits[sci] -m galaxy.dmask",
    )
    parser.add_argument("images", nargs="*", help="FITS images to open")
    parser.add_argument(
        "-i",
        "--image",
        dest="more_images",
        nargs="+",
        action="extend",
        default=[],
        metavar="FILE",
        help="images to load into the buffers (1, 2, ... in this order); name.fits[sci] "
        "or name.fits[1] for one image of a file",
    )
    parser.add_argument(
        "-m",
        "--mask",
        dest="masks",
        nargs="+",
        action="extend",
        default=[],
        metavar="FILE",
        help="masks (FITS or .dmask, 0 = masked) to load as mask layers, ticked for the fit",
    )
    parser.add_argument("--example", action="store_true", help="open the UGC 12517 example")
    parser.add_argument("--run", metavar="FILE", help="run a command file at the start")
    parser.add_argument(
        "--pair",
        metavar="FOLDER",
        help="open a pair's folder (FITS image, .dmask masks, a region file of A and B)",
    )
    parser.add_argument("--version", action="version", version=f"astrovista {__version__}")
    return parser


def open_arguments(window: "MainWindow", args) -> None:
    """Load what the command line asks for: the images into buffers 1, 2, ... in order,
    then the masks as mask layers, then the example and the command file."""
    for name in args.images + args.more_images:
        window.open_path(Path(name))
    for name in args.masks:
        window.load_mask_file(Path(name))
    if args.example:
        window.open_example(show_tool=False)
    if args.pair:
        window.open_pair_folder(Path(args.pair))
    if args.run:
        window.run_command(f"run {shlex.quote(args.run)}")


def main(argv=None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    name_the_mac_app("AstroVista Lite")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("astro-vista")
    app.setApplicationDisplayName(TITLE)
    app.setOrganizationName("AstroVista")
    app.setWindowIcon(app_icon())  # also the Dock and task-bar icon
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    window = MainWindow()
    window.show()
    window.viewer.canvas.setFocus()  # the image has the keyboard at the start
    QtCore.QTimer.singleShot(300, remove_mac_services)  # once macOS has built its menu
    open_arguments(window, args)
    code = app.exec()
    # delete the window (and its Qt objects) before Python exits, in the right order
    window.deleteLater()
    app.processEvents()
    app.sendPostedEvents(None, 0)
    return code
