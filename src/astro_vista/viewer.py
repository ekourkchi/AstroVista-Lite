"""The image viewer: one image, its display, and its layers.

Display: limits (a mode, typed values, or dragged on the histogram), stretch, colour map,
inversion, contrast and bias (drag with the right mouse button: across for the bias, up
and down for the contrast), rotation and mirroring (display only), and a colour bar.

Layers: mask layers, isophotes, sources, the ELLIPROF centre with R0 and R1, contours and
a crosshair.  The compass and scale bar are always shown (when the image has a WCS).

Tools (the mode): ``pan`` (drag to pan, wheel to zoom, click a source to select it),
``pick`` (click a point), ``crosshair`` (the crosshair follows the mouse while the image
pans and zooms as in the pan mode; Alt with the arrow keys nudges it; it stays where it is
when another tool is chosen), and the
mask tools ``circle``, ``ellipse``, ``box``, ``polygon`` and ``brush``.  With the mask
tools, Alt or Shift unmasks.  In every mode, Ctrl-click (⌘-click on macOS) removes the
masked area under the cursor, and dragging with the right button, or with Ctrl, ⌘ or
Control held, changes contrast and bias.  The left and right arrow keys blink: they show
the previous or the next buffer.  The ``eraser`` tool is a brush that unmasks.

All points are reported in ELLIPROF's coordinates (images.py).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import scales, wcsaids
from .contours import trace_smoothed
from .flow import FlowLayout
from .images import Image


def app_icon():
    """The AstroVista logo."""
    from pathlib import Path

    return QtGui.QIcon(str(Path(__file__).resolve().parent / "data" / "astrovista.svg"))


pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

DRAW_MODES = ("circle", "ellipse", "box", "polygon", "brush", "eraser")
MODIFIERS_ERASE = Qt.AltModifier | Qt.ShiftModifier


def cosmetic_pen(colour, width=1.0, style=Qt.SolidLine) -> QtGui.QPen:
    pen = pg.mkPen(colour, width=width, style=style)
    pen.setCosmetic(True)
    return pen


def remove(item) -> None:
    """Take a graphics item off the scene."""
    item.setParentItem(None)
    if item.scene() is not None:
        item.scene().removeItem(item)


class Ellipses(pg.GraphicsObject):
    """Many ellipses (or any path) drawn as one path, with a pen of constant screen width."""

    def __init__(self, colour: str, width: float = 1.3, style=Qt.SolidLine):
        super().__init__()
        self._pen = cosmetic_pen(colour, width, style)
        self._path = QtGui.QPainterPath()

    def set_colour(self, colour: str) -> None:
        self._pen.setColor(QtGui.QColor(colour))
        self.update()

    def set_ellipses(self, ellipses: Sequence) -> None:
        path = QtGui.QPainterPath()
        for x, y, a, b, angle in ellipses:
            shape = QtGui.QPainterPath()
            shape.addEllipse(QtCore.QPointF(0.0, 0.0), float(a), float(b))
            t = QtGui.QTransform()
            t.translate(float(x), float(y))
            t.rotate(float(angle))  # counter-clockwise from +x, since y points up
            path.addPath(t.map(shape))
        self.set_path(path)

    def set_path(self, path: QtGui.QPainterPath) -> None:
        self.prepareGeometryChange()
        self._path = path
        self.update()

    def boundingRect(self):
        return self._path.boundingRect().adjusted(-2, -2, 2, 2)

    def paint(self, painter, *_args):
        painter.setPen(self._pen)
        painter.drawPath(self._path)


class ColourBar(QtWidgets.QWidget):
    """The colour table, with the pixel values at five places along it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self.table = scales.colour_table("Grey")
        self.ticks = []

    def set_bar(self, table, ticks) -> None:
        self.table, self.ticks = np.ascontiguousarray(table, dtype=np.uint8), ticks
        self.update()

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        w, h = self.width() - 16, 14
        image = QtGui.QImage(self.table.tobytes(), 256, 1, 768, QtGui.QImage.Format_RGB888)
        p.drawImage(QtCore.QRect(8, 2, w, h), image)
        p.setPen(self.palette().color(QtGui.QPalette.WindowText))
        font = p.font()
        font.setPointSizeF(max(font.pointSizeF() - 2, 7))
        p.setFont(font)
        metrics = QtGui.QFontMetrics(font)
        for k, (pos, value) in enumerate(self.ticks):
            x = 8 + pos * w
            p.drawLine(QtCore.QPointF(x, 2 + h), QtCore.QPointF(x, 5 + h))
            text = f"{value:.4g}"
            tw = metrics.horizontalAdvance(text)
            left = x - tw / 2 if 0 < k < len(self.ticks) - 1 else (x if k == 0 else x - tw)
            p.drawText(QtCore.QPointF(left, 6 + h + metrics.ascent()), text)


class Compass(QtWidgets.QWidget):
    """North and East arrows and a scale bar, drawn in a corner of the view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFixedSize(150, 110)
        self.north = self.east = None  # screen directions (unit vectors, y down)
        self.bar = None  # (length in screen pixels, label)
        self.colour = QtGui.QColor("#ffffff")

    def paintEvent(self, _event):
        if self.north is None:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtGui.QPen(self.colour, 1.6))
        cx, cy, length = self.width() - 45, 50, 32
        for vec, label in ((self.north, "N"), (self.east, "E")):
            ex, ey = cx + vec[0] * length, cy + vec[1] * length
            p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(ex, ey))
            back = math.atan2(-vec[1], -vec[0])
            for side in (-0.45, 0.45):
                tip = QtCore.QPointF(ex + 7 * math.cos(back + side), ey + 7 * math.sin(back + side))
                p.drawLine(QtCore.QPointF(ex, ey), tip)
            p.drawText(QtCore.QPointF(ex + vec[0] * 9 - 4, ey + vec[1] * 9 + 4), label)
        if self.bar is not None:
            pixels, label = self.bar
            x1, y = self.width() - 10, self.height() - 12
            p.drawLine(QtCore.QPointF(x1 - pixels, y), QtCore.QPointF(x1, y))
            for x in (x1 - pixels, x1):
                p.drawLine(QtCore.QPointF(x, y - 4), QtCore.QPointF(x, y + 4))
            p.drawText(QtCore.QPointF(x1 - pixels / 2 - 8, y - 6), label)


class Viewer(QtWidgets.QWidget):
    """Shows one image with its display settings and layers."""

    moved = QtCore.Signal(float, float)
    bar_folded_changed = QtCore.Signal(bool)  # the display bar folded away or back
    clicked = QtCore.Signal(float, float)  # a click in the pan mode
    picked = QtCore.Signal(float, float)  # a click in the pick mode
    crosshair_moved = QtCore.Signal(float, float)
    circle_drawn = QtCore.Signal(float, float, float, bool)  # x, y, r, erase
    ellipse_drawn = QtCore.Signal(float, float, float, float, float, bool)  # x y a b angle erase
    box_drawn = QtCore.Signal(float, float, float, float, bool)
    polygon_drawn = QtCore.Signal(object, bool)  # [(x, y), ...], erase
    brush_drawn = QtCore.Signal(object, float, bool)  # [(x, y), ...], radius, erase
    unpatch = QtCore.Signal(float, float)  # Ctrl-click
    hover_patch = QtCore.Signal(float, float)  # Ctrl held over the image (NaN: released)
    context = QtCore.Signal(float, float, QtCore.QPoint)  # right click without a drag
    view_changed = QtCore.Signal()
    welcome_action = QtCore.Signal(str)  # "example", "open" or "guide"
    blink_step = QtCore.Signal(int)  # the arrow keys: -1 the previous buffer, +1 the next
    contrast_changed = QtCore.Signal(float, float)  # contrast, bias (a drag)
    pending_angle = QtCore.Signal(float)  # the rotation of the shape waiting to be applied
    hint = QtCore.Signal(str)  # what can be done now (for the status bar)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image: Optional[Image] = None
        self.mode = "pan"
        self.brush = 10.0  # radius of the circle and brush tools
        self.ellipse_angle = 0.0  # angle of the ellipse tool
        self.contrast, self.bias = 1.0, 0.5
        self.rotation, self.mirror = 0.0, False
        self._limits = (0.0, 1.0)
        self._drag = None  # what a mouse drag is doing
        self._drag_click = "menu"  # what the press does when it does not become a drag
        self._polygon: List[tuple] = []
        self._compass_on, self._contours_on = True, False
        self.colours = {"contours": "#ffd84a", "crosshair": "#ff9f40"}

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addLayout(self._display_bar())

        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.setBackground("#15171c")
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.view = self.canvas.addViewBox(lockAspect=True, invertY=False)
        self.view.setMenuEnabled(False)
        self.group = pg.ItemGroup()  # everything drawn on the image, turned with it
        self.view.addItem(self.group)
        self.item = pg.ImageItem()
        self.item.setParentItem(self.group)
        self.mask_items: List[pg.ImageItem] = []
        self.patch_item = pg.ImageItem()
        self.patch_item.setZValue(15)
        self.patch_item.setParentItem(self.group)
        self.contours = pg.PlotCurveItem(
            connect="pairs", pen=cosmetic_pen(self.colours["contours"])
        )
        self.isophotes = Ellipses("#7CFC00", 1.4)
        self.sources = Ellipses("#00d5ff", 1.0)
        self.selected = Ellipses("#ffd200", 2.4)
        # the crosshair: two lines upright on the screen (not turned with the image)
        self.cross_v = pg.InfiniteLine(
            angle=90, movable=False, pen=cosmetic_pen(self.colours["crosshair"])
        )
        self.cross_h = pg.InfiniteLine(
            angle=0, movable=False, pen=cosmetic_pen(self.colours["crosshair"])
        )
        for line in (self.cross_v, self.cross_h):
            line.setZValue(45)
            line.hide()
            self.view.addItem(line, ignoreBounds=True)
        self.crosshair_at = None
        overlays = (
            self.contours,
            self.isophotes,
            self.sources,
            self.selected,
        )
        for z, overlay in enumerate(overlays):
            overlay.setZValue(20 + z)
            overlay.setParentItem(self.group)
        self.preview = Ellipses("#ffffff", 1.2, Qt.DashLine)
        self.preview.setZValue(40)
        self.preview.setParentItem(self.group)
        # a shape drawn but not applied yet: its outline and its rotation handle
        self.pending_item = Ellipses("#ff3fc8", 2.0)
        self.pending_item.setZValue(41)
        self.pending_item.setParentItem(self.group)
        self._pending: Optional[dict] = None
        self.unmask = False  # the Unmask toggle: the mask tools unmask (Alt/Shift masks)
        self.centre = pg.ScatterPlotItem(
            size=9, symbol="+", pen=pg.mkPen("#ff4d4d", width=1.4), brush=None
        )
        self.centre.setZValue(41)
        self.centre.setParentItem(self.group)
        self.radii = Ellipses("#ff4d4d", 1.2, Qt.DashLine)  # ELLIPROF's R0 and R1
        self.radii.setZValue(39)
        self.radii.setParentItem(self.group)
        self.radius_from = None  # while picking a radius: the centre it is measured from
        lay.addWidget(self.canvas, 1)
        self.compass = Compass(self.canvas)
        self.compass.hide()
        self.welcome = self._welcome_card()

        self.histogram = pg.PlotWidget()
        self.histogram.setFixedHeight(70)
        self.histogram.setMenuEnabled(False)
        self.histogram.setMouseEnabled(x=False, y=False)
        self.histogram.hideAxis("left")
        self.hist_curve = self.histogram.plot(
            stepMode="center", fillLevel=0, brush=(80, 140, 220, 90), pen=pg.mkPen("#6aa8ff")
        )
        self.limit_region = pg.LinearRegionItem(brush=(255, 210, 0, 40), pen=pg.mkPen("#ffd200"))
        self.limit_region.sigRegionChangeFinished.connect(self._region_dragged)
        self.histogram.addItem(self.limit_region)
        self.histogram.setToolTip("Drag the yellow edges (or the band) to set the display limits")
        self.colour_bar = ColourBar()
        lay.addWidget(self.histogram)
        lay.addWidget(self.colour_bar)
        self.show_histogram(self.histogram_box.isChecked())

        self.canvas.scene().sigMouseMoved.connect(self._mouse_moved)
        self.canvas.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.canvas.viewport().installEventFilter(self)
        self.canvas.installEventFilter(self)
        self.view.sigRangeChanged.connect(lambda *_: self._view_moved())
        self._later = QtCore.QTimer(self, singleShot=True, interval=120)
        self._later.timeout.connect(self._refresh_view_layers)

    # ---- the display bar
    def _display_bar(self) -> QtWidgets.QHBoxLayout:
        """The display settings in small groups that wrap onto more rows when the viewer is
        narrow (so that the panels beside it can be wide), with a button that folds them
        away (set_bar_folded) for more room for the image."""
        rows = QtWidgets.QHBoxLayout()
        rows.setContentsMargins(6, 4, 6, 0)
        rows.setSpacing(2)
        holder = self.bar_holder = QtWidgets.QWidget()
        flow = FlowLayout(holder, spacing=14)
        rows.addWidget(holder, 1)
        rows.addStretch()  # folded: the button goes to the right
        self.fold_button = QtWidgets.QToolButton()
        self.fold_button.setAutoRaise(True)
        self.fold_button.clicked.connect(lambda: self.set_bar_folded(not self.bar_folded))
        rows.addWidget(self.fold_button, 0, Qt.AlignTop)
        self.bar_folded = False
        self._fold_text()

        def group(*widgets):
            box = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(4)
            for w in widgets:
                lay.addWidget(QtWidgets.QLabel(w) if isinstance(w, str) else w)
            flow.addWidget(box)

        def combo(names, tip):
            box = QtWidgets.QComboBox()
            box.addItems(list(names))
            box.setToolTip(tip)
            return box

        self.limit_mode = combo(scales.LIMITS, "how the display limits are chosen")
        self.limit_mode.currentTextChanged.connect(lambda _t: self.render())
        self.low = self._limit_box("low")
        self.high = self._limit_box("high")
        group("Limits", self.limit_mode, self.low, self.high)
        self.stretch_box = combo(scales.STRETCHES, "the stretch between the limits")
        self.stretch_box.setCurrentText("Log")  # galaxies: the faint outskirts show
        self.stretch_box.currentTextChanged.connect(lambda _t: self.render())
        self.exponent = QtWidgets.QSpinBox()
        self.exponent.setRange(2, 100000)
        self.exponent.setValue(1000)
        self.exponent.setToolTip("the constant of the Log and Power stretches")
        self.exponent.valueChanged.connect(lambda _v: self.render())
        group("Scale", self.stretch_box, self.exponent)
        self.colour_box = combo(scales.COLOURMAPS, "the colour map")
        self.colour_box.currentTextChanged.connect(lambda _t: self.apply_colours())
        self.invert = QtWidgets.QCheckBox("Invert")
        self.invert.toggled.connect(lambda _on: self.apply_colours())
        reset = QtWidgets.QToolButton(text="Reset")
        reset.setToolTip("Contrast 1 and bias 0.5 (drag with the right button to change them)")
        reset.clicked.connect(self.reset_contrast)
        group("Colours", self.colour_box, self.invert, reset)
        self.histogram_box = QtWidgets.QCheckBox("Histogram")
        self.histogram_box.setChecked(False)  # off at the start: more room for the image
        self.histogram_box.setToolTip(
            "Show the histogram (drag its edges to set the limits) and the colour bar"
        )
        self.histogram_box.toggled.connect(self.show_histogram)
        group(self.histogram_box)
        self.blink_box = QtWidgets.QCheckBox("Blink")
        self.blink_box.setToolTip(
            "When ticked, ← and → (after a click on the image) show the previous or the next "
            "of the buffers chosen in the menu beside"
        )
        self.blink_menu = QtWidgets.QMenu(self)
        self.blink_menu.setToolTipsVisible(True)
        self.blink_menu.triggered.connect(lambda _a: self._blink_label())
        # at the top of the menu: keep the same scale from one buffer to the next
        self.same_scale = QtGui.QAction("Same scale for every buffer", self)
        self.same_scale.setCheckable(True)
        self.same_scale.setChecked(True)
        self.same_scale.setToolTip(
            "Keep the display limits from one buffer to the next while blinking, so that "
            "the same brightness looks the same in each"
        )
        self.blink_button = QtWidgets.QToolButton()
        self.blink_button.setToolTip(
            "Same scale, and the buffers to blink between (all of them when none is chosen)"
        )
        self.blink_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.blink_button.setMenu(self.blink_menu)
        self.blink_button.setMinimumWidth(150)
        divider = QtWidgets.QFrame()  # a thin line before Blink
        divider.setFrameShape(QtWidgets.QFrame.VLine)
        divider.setFrameShadow(QtWidgets.QFrame.Sunken)
        divider.setFixedHeight(20)
        gap = QtWidgets.QWidget()
        gap.setFixedWidth(8)
        group(divider, self.blink_box, gap, self.blink_button)
        self.set_blink_buffers([])
        return rows

    def set_bar_folded(self, folded: bool) -> None:
        """Fold the display bar away (more room for the image), or unfold it."""
        if folded == self.bar_folded:
            return
        self.bar_folded = folded
        self.bar_holder.setVisible(not folded)
        self._fold_text()
        self.bar_folded_changed.emit(folded)

    def _fold_text(self) -> None:
        self.fold_button.setText("▾ Display" if self.bar_folded else "▴")
        self.fold_button.setToolTip(
            "Show the display settings" if self.bar_folded else "Hide the display settings"
        )

    def set_blink_buffers(self, buffers) -> None:
        """The buffers the blink menu offers ((number, name[, how it was made])), keeping
        the choice;
        "Same scale" stays at the top of the menu."""
        chosen = set(self.blink_choice())
        self.blink_menu.clear()
        self.blink_menu.addAction(self.same_scale)
        self.blink_menu.addSeparator()
        for item in buffers:
            n, name = item[0], item[1]
            action = self.blink_menu.addAction(f"{n}  {name}")
            action.setCheckable(True)
            action.setChecked(n in chosen)
            action.setData(n)
            if len(item) > 2 and item[2]:  # how it was made
                action.setToolTip(item[2])
        self._blink_label()

    def _buffer_actions(self):
        return [
            a for a in self.blink_menu.actions() if a.data() is not None and not a.isSeparator()
        ]

    def blink_choice(self):
        return [a.data() for a in self._buffer_actions() if a.isChecked()]

    def blink_frames(self):
        """The buffers to step through: those chosen, or all of them."""
        return self.blink_choice() or [a.data() for a in self._buffer_actions()]

    def _blink_label(self) -> None:
        chosen = self.blink_choice()
        self.blink_button.setText(
            ("buffers " + ", ".join(str(n) for n in chosen) if chosen else "all buffers") + " ▾"
        )

    def _welcome_card(self) -> QtWidgets.QWidget:
        """What the viewer shows before an image is opened."""
        card = QtWidgets.QFrame(self.canvas)
        card.setObjectName("welcome")
        card.setStyleSheet(
            "#welcome { background: rgba(30, 34, 44, 235); border-radius: 12px; }"
            "#welcome QLabel { color: #e8ecf3; }"
        )
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(28, 22, 28, 22)
        logo = QtWidgets.QLabel()
        logo.setPixmap(app_icon().pixmap(72, 72))
        logo.setAlignment(Qt.AlignCenter)
        lay.addWidget(logo)
        title = QtWidgets.QLabel("AstroVista Lite")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: 700")
        lay.addWidget(title)
        text = QtWidgets.QLabel(
            "Fit the isophotes of a galaxy, find the sources around it and mask them.\n"
            "Start with the example, or open your own FITS image."
        )
        text.setAlignment(Qt.AlignCenter)
        lay.addWidget(text)
        row = QtWidgets.QHBoxLayout()
        for key, label in (
            ("example", "Open the example"),
            ("open", "Open an image…"),
            ("guide", "First lesson"),
        ):
            b = QtWidgets.QPushButton(label)
            b.setMinimumHeight(30)
            b.clicked.connect(lambda _c=False, k=key: self.welcome_action.emit(k))
            row.addWidget(b)
        lay.addLayout(row)
        card.adjustSize()
        return card

    def _place_welcome(self) -> None:
        self.welcome.setVisible(self.image is None)
        self.welcome.adjustSize()
        w, h = self.welcome.width(), self.welcome.height()
        self.welcome.move((self.canvas.width() - w) // 2, (self.canvas.height() - h) // 2)
        self.welcome.raise_()

    def _limit_box(self, which) -> QtWidgets.QLineEdit:
        box = QtWidgets.QLineEdit()
        box.setMinimumWidth(60)
        box.setMaximumWidth(96)
        box.setToolTip(f"the {which} display limit; typing one switches the limits to Manual")
        box.editingFinished.connect(self._typed_limits)
        box.setFocusPolicy(Qt.ClickFocus)  # keys go to the image unless the field is clicked
        return box

    # ---- the image
    def set_image(
        self, image: Optional[Image], keep_view: bool = False, keep_limits: bool = False
    ) -> None:
        """Show ``image``; ``keep_limits`` keeps the display limits of the image before
        (for blinking with the same scale)."""
        if image is not self.image:
            self.commit_pending()  # a shape waiting on the image before is applied to it
        same = self.image is not None and image is not None and image.shape == self.image.shape
        self.image = image
        self._keep_limits = keep_limits
        self.render()
        self._histogram()
        self._orient()
        if not (keep_view and same):
            self.fit_view()
        self.refresh_aids()
        self._place_welcome()

    def render(self) -> None:
        """Limits and stretch, then the colours: after a change of either."""
        if self.image is None:
            self.item.clear()
            return
        if self.limit_mode.currentText() != "Manual" and not getattr(self, "_keep_limits", False):
            self._limits = scales.limits(self.image.data, self.limit_mode.currentText())
        self._keep_limits = False  # only for the image just shown
        lo, hi = self._limits
        self.low.setText(f"{lo:.6g}")
        self.high.setText(f"{hi:.6g}")
        self._limit_text = (self.low.text(), self.high.text())  # what typing must change
        if self.limit_mode.currentText() != "Manual":
            self._histogram()
        self.limit_region.blockSignals(True)
        self.limit_region.setRegion((lo, hi))
        self.limit_region.blockSignals(False)
        name = self.stretch_box.currentText()
        self.exponent.setEnabled(name in ("Log", "Power"))
        data = scales.scaled(self.image.data, lo, hi, name, self.exponent.value())
        self.item.setImage(data, levels=(0.0, 1.0), autoLevels=False)
        self.item.setRect(QtCore.QRectF(0, 0, self.image.shape[1], self.image.shape[0]))
        self.apply_colours()
        if self._contours_on:
            self.draw_contours()

    def apply_colours(self) -> None:
        """The colour table only (colour map, inversion, contrast, bias): instant."""
        table = scales.colour_table(
            self.colour_box.currentText(), self.invert.isChecked(), self.contrast, self.bias
        )
        self.item.setLookupTable(table)
        if self.image is None:
            return
        lo, hi = self._limits
        name = self.stretch_box.currentText()
        if name == "Histogram":  # the colour bar marks quantiles of the pixels shown
            values = np.sort(scales.sample(np.clip(self.image.data, lo, hi)))
            ticks = []
            if values.size:
                for m in np.linspace(0, 1, 5):
                    ticks.append((float(m), float(values[int(m * (values.size - 1))])))
        else:
            ticks = scales.colour_bar_ticks(lo, hi, name, self.exponent.value())
        self.colour_bar.set_bar(table, ticks)

    def reset_contrast(self) -> None:
        self.contrast, self.bias = 1.0, 0.5
        self.apply_colours()

    def set_limits(self, lo: float, hi: float) -> None:
        if not hi > lo:
            return
        self._limits = (float(lo), float(hi))
        self.limit_mode.blockSignals(True)
        self.limit_mode.setCurrentText("Manual")
        self.limit_mode.blockSignals(False)
        self.render()

    def _typed_limits(self) -> None:
        if (self.low.text(), self.high.text()) == getattr(self, "_limit_text", None):
            return  # only the focus left the field
        try:
            lo, hi = float(self.low.text()), float(self.high.text())
        except ValueError:
            return
        if (lo, hi) != self._limits:
            self.set_limits(lo, hi)

    def _region_dragged(self) -> None:
        lo, hi = self.limit_region.getRegion()
        self.set_limits(lo, hi)

    def _histogram(self) -> None:
        if self.image is None:
            return
        values = scales.sample(self.image.data)
        if not values.size:
            return
        # around the limits (half their span on each side), within the data
        span = self._limits[1] - self._limits[0]
        lo = max(float(values.min()), self._limits[0] - 0.5 * span)
        hi = min(float(values.max()), self._limits[1] + 0.5 * span)
        if not hi > lo:
            hi = lo + 1
        counts, edges = np.histogram(values, bins=256, range=(lo, hi))
        self.hist_curve.setData(edges, np.log10(counts + 1.0))
        self.histogram.setXRange(lo, hi, padding=0.02)

    def show_histogram(self, on: bool) -> None:
        self.histogram.setVisible(on)
        self.colour_bar.setVisible(on)

    # ---- the view
    def fit_view(self) -> None:
        if self.image is not None:
            self.view.autoRange(items=[self.item], padding=0.02)

    def zoom(self, factor: float) -> None:
        self.view.scaleBy((1 / factor, 1 / factor))

    def actual_size(self) -> None:
        """One image pixel to one screen pixel, about the middle of the view."""
        r, size = self.view.viewRect(), self.view.size()
        c = r.center()
        w, h = size.width(), size.height()
        self.view.setRange(QtCore.QRectF(c.x() - w / 2, c.y() - h / 2, w, h), padding=0)

    def zoom_factor(self) -> float:
        """Screen pixels per image pixel."""
        r = self.view.viewRect()
        return self.view.size().width() / r.width() if r.width() else 1.0

    def view_lock(self) -> Optional[dict]:
        """What blinking keeps from the view: the image point at the middle of the view, the
        zoom and the image's WCS; None when the whole image is in view (not zoomed in), so
        that the next image is shown whole."""
        if self.image is None:
            return None
        ny, nx = self.image.shape
        whole = self.group.mapRectToParent(QtCore.QRectF(0, 0, nx, ny))
        if self.view.viewRect().adjusted(-1, -1, 1, 1).contains(whole):
            return None
        middle = self.group.mapFromParent(self.view.viewRect().center())
        return dict(x=middle.x(), y=middle.y(), zoom=self.zoom_factor(), wcs=self.image.wcs())

    def apply_lock(self, lock: dict) -> str:
        """Show the image now in the viewer at the place and zoom kept by ``view_lock``:
        the same sky position and angular size when both images have a WCS that covers it
        ("wcs"), otherwise the same pixel position and zoom ("image")."""
        ny, nx = self.image.shape
        x, y, zoom, how = lock["x"], lock["y"], lock["zoom"], "image"
        old, new = lock["wcs"], self.image.wcs()
        if old is not None and new is not None:
            try:
                from astropy.wcs.utils import proj_plane_pixel_scales

                # viewer coordinates are FITS pixels - 0.5, astropy's are FITS pixels - 1
                sky = old.pixel_to_world(x - 0.5, y - 0.5)
                px, py = (float(v) + 0.5 for v in new.world_to_pixel(sky))
                ratio = float(np.mean(proj_plane_pixel_scales(new))) / float(
                    np.mean(proj_plane_pixel_scales(old))
                )
                if 0 <= px <= nx and 0 <= py <= ny and np.isfinite(ratio) and ratio > 0:
                    x, y, zoom, how = px, py, zoom * ratio, "wcs"
            except Exception:  # skies that do not match: lock on the image instead
                pass
        if how == "image":  # the same pixel, kept inside the image
            x, y = min(max(x, 0.0), nx), min(max(y, 0.0), ny)
        p = self.group.mapToParent(QtCore.QPointF(x, y))
        size = self.view.size()
        w, h = size.width() / zoom, size.height() / zoom
        self.view.setRange(QtCore.QRectF(p.x() - w / 2, p.y() - h / 2, w, h), padding=0)
        return how

    def centre_on(self, x: float, y: float) -> None:
        p = self.group.mapToParent(QtCore.QPointF(x, y))
        r = self.view.viewRect()
        w, h = r.width(), r.height()
        self.view.setRange(QtCore.QRectF(p.x() - w / 2, p.y() - h / 2, w, h), padding=0)

    def set_orientation(self, rotation: float, mirror: bool) -> None:
        """Rotate (degrees counter-clockwise) and mirror (left-right) the display only."""
        self.rotation, self.mirror = rotation % 360.0, mirror
        self._orient()
        self.fit_view()
        self.refresh_aids()

    def _orient(self) -> None:
        if self.image is None:
            return
        ny, nx = self.image.shape
        t = QtGui.QTransform()
        t.translate(nx / 2, ny / 2)
        t.rotate(self.rotation)
        if self.mirror:
            t.scale(-1, 1)
        t.translate(-nx / 2, -ny / 2)
        self.group.setTransform(t)
        if self.crosshair_at is not None:
            self.set_crosshair(*self.crosshair_at)

    def north_up(self) -> bool:
        """Turn the display so that North is up and East left; False without a WCS."""
        w = self.image.wcs() if self.image else None
        if w is None:
            return False
        ny, nx = self.image.shape
        north, east, _scale = wcsaids.sky_axes(w, nx / 2, ny / 2, 0, 0)
        self.set_orientation(*wcsaids.north_up(north, east))
        return True

    def _view_moved(self) -> None:
        self._later.start()
        self.view_changed.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.compass.move(self.canvas.width() - self.compass.width() - 8, 8)
        self._place_welcome()

    def showEvent(self, event):
        super().showEvent(event)
        self._place_welcome()

    def _refresh_view_layers(self) -> None:
        self._compass()
        self._draw_pending()  # the rotation handle keeps its size on the screen

    # ---- layers
    def show_isophotes(self, ellipses) -> None:
        self.isophotes.set_ellipses(ellipses or [])

    def show_sources(self, ellipses) -> None:
        self.sources.set_ellipses(list(ellipses or []))

    def show_selected(self, ellipse) -> None:
        self.selected.set_ellipses([ellipse] if ellipse is not None else [])

    def show_centre(self, x: Optional[float], y: Optional[float]) -> None:
        self.centre.setData([x] if x is not None else [], [y] if y is not None else [])

    def show_radii(self, x, y, radii) -> None:
        """Dashed circles of the radii (R0, R1) about the centre (x, y); None: none."""
        if x is None:
            self.radii.set_ellipses([])
        else:
            self.radii.set_ellipses([(x, y, r, r, 0.0) for r in radii if r > 0])

    def show_masks(self, layers) -> None:
        """Draw the mask layers (Mask objects) in their colours and opacities."""
        for item in self.mask_items:
            remove(item)
        self.mask_items = []
        for z, m in enumerate(layers):
            rgba = np.zeros(m.shape + (4,), np.uint8)
            c = QtGui.QColor(m.colour)
            rgba[m.masked] = (c.red(), c.green(), c.blue(), int(255 * m.opacity))
            item = pg.ImageItem(rgba)
            item.setRect(QtCore.QRectF(0, 0, m.shape[1], m.shape[0]))
            item.setZValue(10 + z * 0.01)
            item.setParentItem(self.group)
            self.mask_items.append(item)

    def show_patch(self, patch: Optional[np.ndarray]) -> None:
        """Outline in red the masked area a Ctrl-click would remove (None: nothing)."""
        if patch is None or not patch.any():
            self.patch_item.hide()
            return
        rows = np.flatnonzero(patch.any(axis=1))
        cols = np.flatnonzero(patch.any(axis=0))
        j0, j1, i0, i1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        rgba = np.zeros((j1 - j0, i1 - i0, 4), np.uint8)
        rgba[patch[j0:j1, i0:i1]] = (255, 40, 40, 170)
        self.patch_item.setImage(rgba)
        self.patch_item.setRect(QtCore.QRectF(i0, j0, i1 - i0, j1 - j0))
        self.patch_item.show()

    def set_layer_colour(self, key: str, colour: str) -> None:
        self.colours[key] = colour
        if key == "contours":
            self.contours.setPen(cosmetic_pen(colour))
        elif key == "crosshair":
            for line in (self.cross_v, self.cross_h):
                line.setPen(cosmetic_pen(colour))
        elif key == "compass":
            self.compass.colour = QtGui.QColor(colour)
            self.compass.update()
        elif key == "isophotes":
            self.isophotes.set_colour(colour)
        elif key == "sources":
            self.sources.set_colour(colour)

    def set_contours(self, levels, method: str = "Smooth", size: int = 1) -> None:
        """Contours at ``levels`` (values), after smoothing (contours.smoothed)."""
        self._contour_spec = (list(levels), method, int(size))
        self.draw_contours()

    def show_contours(self, on: bool) -> None:
        self._contours_on = on
        self.draw_contours()

    def contour_style(self, colour: str, width: float, dashed: bool) -> None:
        self.colours["contours"] = colour
        style = Qt.DashLine if dashed else Qt.SolidLine
        self.contours.setPen(cosmetic_pen(colour, width, style))

    def draw_contours(self) -> None:
        levels, method, size = getattr(self, "_contour_spec", ([], "Smooth", 1))
        if not self._contours_on or self.image is None or not levels:
            self.contours.setData([], [])
            return
        geometry = trace_smoothed(self.image.data, levels, method, size)
        self.contours.setData(geometry.x, geometry.y)

    def show_compass(self, on: bool) -> None:
        self._compass_on = on
        self.refresh_aids()

    def refresh_aids(self) -> None:
        """The compass and scale bar (only when the image has a WCS)."""
        self._compass()

    def _compass(self) -> None:
        w = self.image.wcs() if self.image else None
        if not self._compass_on or w is None:
            self.compass.hide()
            return
        t = self.group.transform()
        r = t.inverted()[0].mapRect(self.view.viewRect())
        north, east, scale = wcsaids.sky_axes(w, r.center().x(), r.center().y(), 0, 0)

        def on_screen(v):  # through the display's turn; screen y points down
            p0, p1 = t.map(QtCore.QPointF(0, 0)), t.map(QtCore.QPointF(v[0], v[1]))
            d = np.array([p1.x() - p0.x(), -(p1.y() - p0.y())])
            return d / (np.hypot(*d) or 1.0)

        # East is drawn at right angles to North, as on the sky, on the side the WCS
        # gives (a skewed WCS, such as HST WFC3/IR's, puts it a few degrees off on the
        # pixel grid)
        n, e = on_screen(north), on_screen(east)
        side = 1.0 if n[0] * e[1] - n[1] * e[0] > 0 else -1.0
        self.compass.north = n
        self.compass.east = np.array([-n[1], n[0]]) * side
        zoom = self.zoom_factor()
        length, label = wcsaids.scale_bar_length(scale, 120 / max(zoom, 1e-6))
        self.compass.bar = (length * zoom, label)
        self.compass.show()
        self.compass.raise_()
        self.compass.update()

    def set_crosshair(self, x: Optional[float], y: Optional[float]) -> None:
        """Put the crosshair at (x, y) of the image; its lines stay horizontal and vertical
        on the screen however the display is turned."""
        self.crosshair_at = None if x is None else (x, y)
        if x is None or self.image is None:
            self.cross_v.hide()
            self.cross_h.hide()
            return
        p = self.group.mapToParent(QtCore.QPointF(x, y))
        self.cross_v.setPos(p.x())
        self.cross_h.setPos(p.y())
        self.cross_v.show()
        self.cross_h.show()

    # ---- tool modes and the mouse
    def set_mode(self, mode: str) -> None:
        self.commit_pending()
        self._polygon = []
        self.preview.set_ellipses([])
        self.mode = mode
        self.canvas.unsetCursor() if mode == "pan" else self.canvas.setCursor(Qt.CrossCursor)
        self.canvas.setFocus()

    def _point(self, scene_pos):
        p = self.group.mapFromScene(scene_pos)
        return p.x(), p.y()

    def _event_point(self, event):
        return self._point(self.canvas.mapToScene(event.position().toPoint()))

    def _mouse_moved(self, pos) -> None:
        if self.image is None or not self.view.sceneBoundingRect().contains(pos):
            return
        x, y = self._point(pos)
        if self._pending is not None and self._drag is None:  # what a press would do
            part = self._pending_part(x, y)
            cursors = {"move": Qt.SizeAllCursor, "size": Qt.SizeFDiagCursor}
            cursors["rotate"] = Qt.PointingHandCursor
            self.canvas.setCursor(cursors[part[0]] if part else Qt.CrossCursor)
        if self.mode in ("circle", "brush", "eraser") and self._drag is None:
            self.preview.set_ellipses([(x, y, self.brush, self.brush, 0.0)])
        elif self.mode == "pick" and self.radius_from is not None:
            cx, cy = self.radius_from
            r = math.hypot(x - cx, y - cy)
            self.preview.set_ellipses([(cx, cy, r, r, 0.0)])
        elif self.mode == "polygon" and self._polygon:
            self._polygon_preview((x, y))
        elif self.mode == "crosshair" and self._drag is None:
            ny, nx = self.image.shape  # it stays on the image
            cx, cy = min(max(x, 0.0), nx), min(max(y, 0.0), ny)
            self.set_crosshair(cx, cy)
            self.crosshair_moved.emit(cx, cy)
        if QtWidgets.QApplication.keyboardModifiers() & Qt.ControlModifier:
            self.hover_patch.emit(x, y)
        self.moved.emit(x, y)

    def _mouse_clicked(self, event) -> None:
        if self.image is None or event.button() != Qt.LeftButton:
            return
        if self.mode in DRAW_MODES or event.modifiers() & Qt.ControlModifier:
            return  # handled on the press
        x, y = self._point(event.scenePos())
        if self.mode == "pick":
            self.picked.emit(x, y)
        else:  # the pan mode, or the crosshair mode (which pans too)
            self.clicked.emit(x, y)

    def _polygon_preview(self, cursor=None) -> None:
        points = self._polygon + ([cursor] if cursor else [])
        path = QtGui.QPainterPath()
        if points:
            path.moveTo(*points[0])
            for p in points[1:]:
                path.lineTo(*p)
            if len(points) > 2:
                path.lineTo(*points[0])
        self.preview.set_path(path)

    def _finish_polygon(self, erase: bool) -> None:
        points, self._polygon = list(self._polygon), []
        self.preview.set_ellipses([])
        if len(points) >= 3:  # waits, to be rotated, until applied
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            rel = [(x - cx, y - cy) for x, y in points]
            self._start_pending("polygon", erase, cx, cy, 0.0, points=rel)

    # ---- a shape drawn but not applied yet (ellipse, box, polygon): it can be moved (drag
    # it from inside), resized (its square handles: the ends of an ellipse's axes, a box's
    # corners, a polygon's corners) and rotated (its round handle, the [ and ] keys or the
    # angle box); it is applied with Enter, a double-click, another shape, another tool or
    # any command, and Esc cancels it
    def _start_pending(self, kind, erase, cx, cy, angle, **shape) -> None:
        self.commit_pending()
        self._pending = dict(kind=kind, erase=erase, cx=cx, cy=cy, angle=angle, **shape)
        self._draw_pending()
        self.pending_angle.emit(angle)
        self.hint.emit(
            "Drag the shape to move it, its squares to resize it, its round handle to "
            "rotate it (Shift: 15° steps; or [ and ]); Enter or a double-click applies it, "
            "Esc cancels"
        )

    def pending(self) -> Optional[dict]:
        return self._pending

    def _shape_points(self, d: dict) -> List[tuple]:
        """The pending shape's outline in its own frame (before its rotation)."""
        if d["kind"] == "ellipse":
            return [
                (d["a"] * math.cos(t), d["b"] * math.sin(t))
                for t in np.linspace(0, 2 * math.pi, 73)[:-1]
            ]
        return list(d["points"])

    def _turned(self, d: dict, points) -> List[tuple]:
        """Points of the shape's own frame, rotated by its angle about its centre."""
        t = math.radians(d["angle"])
        c, s = math.cos(t), math.sin(t)
        return [(d["cx"] + u * c - v * s, d["cy"] + u * s + v * c) for u, v in points]

    def _local(self, d: dict, x: float, y: float) -> tuple:
        """An image point in the shape's own frame (centred, not rotated)."""
        t = math.radians(d["angle"])
        dx, dy = x - d["cx"], y - d["cy"]
        return dx * math.cos(t) + dy * math.sin(t), -dx * math.sin(t) + dy * math.cos(t)

    @staticmethod
    def _size_points(d: dict) -> List[tuple]:
        """The square handles, in the shape's own frame: the ends of an ellipse's axes, a
        box's corners, a polygon's corners."""
        if d["kind"] == "ellipse":
            a, b = d["a"], d["b"]
            return [(a, 0.0), (0.0, b), (-a, 0.0), (0.0, -b)]
        return list(d["points"])

    def _handle(self, d: dict) -> tuple:
        """Where the rotation handle is (image coordinates), the top of the shape, and the
        handle's radius: above the shape, a fixed distance on the screen."""
        zoom = max(self.zoom_factor(), 1e-6)
        top = max(v for _u, v in self._shape_points(d))
        (hx, hy), (tx, ty) = self._turned(d, [(0.0, top + 24 / zoom), (0.0, top)])
        return hx, hy, tx, ty, 5 / zoom

    def _draw_pending(self) -> None:
        d = self._pending
        if d is None:
            self.pending_item.set_path(QtGui.QPainterPath())
            if self.mode in DRAW_MODES:  # the move/resize cursor goes with the shape
                self.canvas.setCursor(Qt.CrossCursor)
            return
        # a shape that will unmask is drawn dashed
        self.pending_item._pen.setStyle(Qt.DashLine if d["erase"] else Qt.SolidLine)
        outline = self._turned(d, self._shape_points(d))
        path = QtGui.QPainterPath()
        path.moveTo(*outline[0])
        for p in outline[1:]:
            path.lineTo(*p)
        path.closeSubpath()
        hx, hy, tx, ty, r = self._handle(d)
        path.moveTo(tx, ty)  # the stem, then the round handle
        path.lineTo(hx, hy)
        path.addEllipse(QtCore.QPointF(hx, hy), r, r)
        for px, py in self._turned(d, self._size_points(d)):  # the square handles
            path.addRect(QtCore.QRectF(px - r * 0.8, py - r * 0.8, r * 1.6, r * 1.6))
        self.pending_item.set_path(path)

    def set_pending_angle(self, angle: float) -> None:
        """Turn the pending shape to ``angle`` degrees (the angle box, the keys)."""
        if self._pending is None:
            return
        angle = (float(angle) + 180.0) % 360.0 - 180.0
        if abs(angle - self._pending["angle"]) < 1e-9:
            return
        self._pending["angle"] = angle
        self._draw_pending()
        self.pending_angle.emit(angle)

    def cancel_pending(self) -> None:
        self._pending = None
        self._draw_pending()

    def commit_pending(self) -> None:
        """Apply the pending shape to the mask (as the commands of its kind)."""
        d, self._pending = self._pending, None
        if d is None:
            return
        self._draw_pending()
        if d["kind"] == "ellipse":
            self.ellipse_drawn.emit(d["cx"], d["cy"], d["a"], d["b"], d["angle"], d["erase"])
            return
        points = self._turned(d, d["points"])
        if d["kind"] == "box" and abs(d["angle"]) % 90.0 < 1e-9:  # still upright: a box
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            self.box_drawn.emit(min(xs), min(ys), max(xs), max(ys), d["erase"])
        else:  # a turned box is a polygon of its four corners
            self.polygon_drawn.emit(points, d["erase"])

    def _on_handle(self, x: float, y: float) -> bool:
        if self._pending is None:
            return False
        hx, hy, _tx, _ty, r = self._handle(self._pending)
        return math.hypot(x - hx, y - hy) <= 2.2 * r

    def _pending_part(self, x: float, y: float):
        """What of the pending shape is under (x, y): ("rotate",), ("size", k) for the
        k-th square handle, ("move",) inside the shape, or None."""
        d = self._pending
        if d is None:
            return None
        if self._on_handle(x, y):
            return ("rotate",)
        r = self._handle(d)[4]
        for k, (px, py) in enumerate(self._turned(d, self._size_points(d))):
            if math.hypot(x - px, y - py) <= 2.2 * r:
                return ("size", k)
        outline = self._turned(d, self._shape_points(d))
        if QtGui.QPolygonF([QtCore.QPointF(*p) for p in outline]).containsPoint(
            QtCore.QPointF(x, y), Qt.OddEvenFill
        ):
            return ("move",)
        return None

    def _resize_pending(self, k: int, x: float, y: float, fixed) -> None:
        """Drag the k-th square handle to (x, y): an ellipse's axis about the centre, a
        box's corner with the opposite corner (``fixed``, image coordinates) kept, or a
        polygon's corner."""
        d = self._pending
        u, v = self._local(d, x, y)
        if d["kind"] == "ellipse":
            if k % 2 == 0:
                d["a"] = max(abs(u), 0.5)
            else:
                d["b"] = max(abs(v), 0.5)
        elif d["kind"] == "box":
            fu, fv = self._local(d, *fixed)
            d["cx"], d["cy"] = self._turned(d, [((u + fu) / 2, (v + fv) / 2)])[0]
            w, h = max(abs(u - fu) / 2, 0.5), max(abs(v - fv) / 2, 0.5)
            d["points"] = [(-w, -h), (w, -h), (w, h), (-w, h)]
        else:
            d["points"][k] = (u, v)
        self._draw_pending()

    def eventFilter(self, obj, event):
        kind = event.type()
        if obj is self.canvas:
            if kind == QtCore.QEvent.KeyPress:
                return self._key(event)
            if kind == QtCore.QEvent.KeyRelease and event.key() == Qt.Key_Control:
                self.hover_patch.emit(float("nan"), float("nan"))
            return False
        if self.image is None:
            return False
        if kind == QtCore.QEvent.MouseButtonPress:
            return self._press(event)
        if kind == QtCore.QEvent.MouseMove and self._drag is not None:
            return self._move(event)
        if kind == QtCore.QEvent.MouseButtonRelease and self._drag is not None:
            return self._release(event)
        if kind == QtCore.QEvent.MouseButtonDblClick and self.mode == "polygon" and self._polygon:
            self._polygon = self._polygon[:-1]  # the double-click's second press added one
            self._finish_polygon(bool(event.modifiers() & MODIFIERS_ERASE))
            return True
        if kind == QtCore.QEvent.MouseButtonDblClick and self._pending is not None:
            self.commit_pending()
            return True
        return False

    def _key(self, event) -> bool:
        arrows = {
            Qt.Key_Left: (-1, 0),
            Qt.Key_Right: (1, 0),
            Qt.Key_Up: (0, 1),
            Qt.Key_Down: (0, -1),
        }
        alt = bool(event.modifiers() & Qt.AltModifier)
        if event.key() in (Qt.Key_Left, Qt.Key_Right) and not alt and self.blink_box.isChecked():
            self.blink_step.emit(-1 if event.key() == Qt.Key_Left else 1)
            return True
        if self.mode == "crosshair" and self.crosshair_at and event.key() in arrows and alt:
            step = 10.0 if event.modifiers() & Qt.ShiftModifier else 1.0
            dx, dy = arrows[event.key()]
            # the arrows move it on the screen, whatever the display's turn
            t = self.group.transform()
            p = t.map(QtCore.QPointF(*self.crosshair_at))
            back = t.inverted()[0].map(QtCore.QPointF(p.x() + dx * step, p.y() + dy * step))
            self.set_crosshair(back.x(), back.y())
            self.crosshair_moved.emit(back.x(), back.y())
            return True
        if self._pending is not None and not self._polygon:
            steps = {Qt.Key_BracketLeft: 5.0, Qt.Key_BracketRight: -5.0}
            steps.update({Qt.Key_BraceLeft: 1.0, Qt.Key_BraceRight: -1.0})  # with Shift
            if event.key() in steps:
                self.set_pending_angle(self._pending["angle"] + steps[event.key()])
            elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.commit_pending()
            elif event.key() == Qt.Key_Escape:
                self.cancel_pending()
            else:
                return False
            return True
        if self.mode != "polygon" or not self._polygon:
            return False
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._finish_polygon(bool(event.modifiers() & MODIFIERS_ERASE))
        elif event.key() == Qt.Key_Escape:
            self._polygon = []
            self.preview.set_ellipses([])
        elif event.key() == Qt.Key_Backspace:
            self._polygon.pop()
            self._polygon_preview()
        else:
            return False
        return True

    def set_unmask(self, on: bool) -> None:
        """The Unmask toggle; a shape waiting to be applied follows it."""
        self.unmask = bool(on)
        if self._pending is not None:
            self._pending["erase"] = self.unmask
            self._draw_pending()

    def _press(self, event) -> bool:
        x, y = self._event_point(event)
        # Alt or Shift does the opposite of the Unmask toggle
        erase = bool(event.modifiers() & MODIFIERS_ERASE) != self.unmask
        if event.button() == Qt.RightButton:
            self._drag = ("contrast", event.position(), self.contrast, self.bias, x, y)
            self._drag_click = "menu"
            return True
        if event.button() != Qt.LeftButton:
            return False
        if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            # a click removes the masked patch (Ctrl / ⌘) or opens the menu (Control on a
            # Mac); a drag changes contrast and bias
            self._drag = ("contrast", event.position(), self.contrast, self.bias, x, y)
            self._drag_click = "menu" if event.modifiers() & Qt.MetaModifier else "unpatch"
            return True
        self._drag_click = "menu"
        part = self._pending_part(x, y) if self.mode in DRAW_MODES + ("pan",) else None
        if part is not None:  # rotate, resize or move the pending shape
            d = self._pending
            if part[0] == "size":
                fixed = None
                if d["kind"] == "box":  # the corner opposite the one dragged stays put
                    fixed = self._turned(d, [d["points"][(part[1] + 2) % 4]])[0]
                self._drag = ("size", part[1], fixed)
            elif part[0] == "move":
                self._drag = ("move", x, y, d["cx"], d["cy"])
            else:
                self._drag = ("rotate",)
            return True
        if self.mode in DRAW_MODES and not self._polygon:
            self.commit_pending()  # another shape: the pending one is applied first
        if self.mode == "polygon":
            self._polygon.append((x, y))
            self._polygon_preview()
            return True
        if self.mode in ("circle", "ellipse", "box"):
            self._drag = (self.mode, event.position(), x, y, erase)
            return True
        if self.mode in ("brush", "eraser"):
            self._drag = ("brush", [(x, y)], erase or self.mode == "eraser")
            self.preview.set_ellipses([(x, y, self.brush, self.brush, 0.0)])
            return True
        return False

    def _move(self, event) -> bool:
        x, y = self._event_point(event)
        kind = self._drag[0]
        if kind == "contrast":
            _, start, contrast, bias, _x, _y = self._drag
            d = event.position() - start
            w, h = max(self.canvas.width(), 1), max(self.canvas.height(), 1)
            self.bias = float(np.clip(bias + d.x() / w, 0.0, 1.0))
            self.contrast = float(np.clip(contrast * math.exp(-2.0 * d.y() / h), 0.05, 20.0))
            self.apply_colours()
            self.contrast_changed.emit(self.contrast, self.bias)
        elif kind == "crosshair":
            self.set_crosshair(x, y)
            self.crosshair_moved.emit(x, y)
        elif kind == "move" and self._pending is not None:
            _, x0, y0, cx, cy = self._drag
            self._pending["cx"], self._pending["cy"] = cx + x - x0, cy + y - y0
            self._draw_pending()
        elif kind == "size" and self._pending is not None:
            self._resize_pending(self._drag[1], x, y, self._drag[2])
        elif kind == "rotate" and self._pending is not None:
            d = self._pending
            # the handle points along the shape's own +y: 90° from its angle
            angle = math.degrees(math.atan2(y - d["cy"], x - d["cx"])) - 90.0
            if event.modifiers() & Qt.ShiftModifier:
                angle = round(angle / 15.0) * 15.0
            self.set_pending_angle(angle)
        elif kind == "circle":
            x0, y0 = self._drag[2], self._drag[3]
            r = math.hypot(x - x0, y - y0)
            self.preview.set_ellipses([(x0, y0, r, r, 0.0)])
        elif kind == "ellipse":
            x0, y0 = self._drag[2], self._drag[3]
            a, b = self._ellipse_axes(x0, y0, x, y)
            self.preview.set_ellipses([(x0, y0, a, b, self.ellipse_angle)])
        elif kind == "box":
            x0, y0 = self._drag[2], self._drag[3]
            path = QtGui.QPainterPath()
            path.addRect(QtCore.QRectF(QtCore.QPointF(x0, y0), QtCore.QPointF(x, y)).normalized())
            self.preview.set_path(path)
        elif kind == "brush":
            points = self._drag[1]
            points.append((x, y))
            step = max(len(points) // 200, 1)
            r = self.brush
            self.preview.set_ellipses([(px, py, r, r, 0.0) for px, py in points[::step]])
        return True

    def _ellipse_axes(self, x0, y0, x, y):
        """Semi-axes from a drag, along the ellipse's own axes (at the tool's angle)."""
        t = math.radians(self.ellipse_angle)
        dx, dy = x - x0, y - y0
        return abs(dx * math.cos(t) + dy * math.sin(t)), abs(-dx * math.sin(t) + dy * math.cos(t))

    def _release(self, event) -> bool:
        x, y = self._event_point(event)
        drag, self._drag = self._drag, None
        kind = drag[0]
        if kind == "contrast":
            if (event.position() - drag[1]).manhattanLength() < 4:  # a click, not a drag
                if self._drag_click == "unpatch":
                    self.unpatch.emit(drag[4], drag[5])
                else:
                    self.context.emit(drag[4], drag[5], event.globalPosition().toPoint())
            return True
        if kind in ("crosshair", "rotate", "move", "size"):
            return True
        self.preview.set_ellipses([])
        if kind == "brush":
            self.brush_drawn.emit(drag[1], self.brush, drag[2])
            return True
        _, start, x0, y0, erase = drag
        dragged = (event.position() - start).manhattanLength() >= 4
        if kind == "circle":
            r = math.hypot(x - x0, y - y0) if dragged else self.brush
            self.circle_drawn.emit(x0, y0, r, erase)
        elif kind == "ellipse" and dragged:
            a, b = self._ellipse_axes(x0, y0, x, y)
            if a >= 0.5 and b >= 0.5:  # waits, to be rotated, until applied
                self._start_pending("ellipse", erase, x0, y0, self.ellipse_angle, a=a, b=b)
        elif kind == "box" and dragged:
            cx, cy = (x0 + x) / 2, (y0 + y) / 2
            w, h = abs(x - x0) / 2, abs(y - y0) / 2
            corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
            self._start_pending("box", erase, cx, cy, 0.0, points=corners)
        return True
