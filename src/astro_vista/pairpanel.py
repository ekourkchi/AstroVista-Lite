"""The Pair fit tool (two overlapping galaxies, fitted in turn) and its results.

The panel only builds command lines (``sky N SEGMENT ...``, ``pair N ...``); the window
runs them, as it runs every other tool's.  Galaxy A is drawn in cyan and B in orange,
everywhere.
"""

from __future__ import annotations

import math
import shlex
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import guide
from .fitting import MODES
from .flow import FlowLayout
from .panels import heading, note, number, plain_table, tip, title_row, whole

COLOURS = {"A": "#22d3ee", "B": "#ff9f1c"}  # cyan and orange, as in the examples
STATE_COLOURS = {
    "usable_support": "#3fb950",
    "candidate_inspect_support": "#d29922",
    "no_estimate": "#f85149",
}
STATUS_COLOURS = {
    "stable_valid_pair": "#3fb950",
    "valid_pair_not_stable": "#d29922",
    "retained_after_failed_update": "#d29922",
    "incomplete_initialization": "#f85149",
    "cancelled": "#8b919c",
}


class SkyMap(QtWidgets.QWidget):
    """The cell medians of a sky measurement as a small map (unusable cells hatched), with
    an arrow along the gradient (an image-axis direction; up is +y, as on the image)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.setMinimumHeight(110)
        self.setToolTip(
            "The median of each cell's sky pixels (dark: low, light: high); hatched cells "
            "have too little sky. The arrow points up the fitted gradient."
        )

    def set_result(self, result) -> None:
        self.result = result
        self.setVisible(result is not None)
        self.update()

    def sizeHint(self):
        return QtCore.QSize(200, 120)

    def paintEvent(self, _event):
        r = self.result
        if r is None or not r.cells:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        n = int(r.settings["CELLS"])
        ny, nx = r.support.shape
        side = min(self.width() - 8, (self.height() - 8) * nx / ny)
        w, h = side, side * ny / nx
        x0, y0 = (self.width() - w) / 2, (self.height() - h) / 2
        usable = [c.median for c in r.cells if c.usable]
        lo, hi = (min(usable), max(usable)) if usable else (0.0, 1.0)
        span = (hi - lo) or 1.0
        for c in r.cells:
            rect = QtCore.QRectF(
                x0 + w * c.x0 / nx,
                y0 + h * (1 - c.y1 / ny),
                w * (c.x1 - c.x0) / nx,
                h * (c.y1 - c.y0) / ny,
            )
            if c.usable:
                g = int(40 + 200 * (c.median - lo) / span)
                p.fillRect(rect, QtGui.QColor(g, g, g))
                if c.rejected:
                    p.setPen(QtGui.QPen(QtGui.QColor("#f85149"), 1.5))
                    p.drawRect(rect.adjusted(1, 1, -1, -1))
            else:
                p.fillRect(rect, QtGui.QBrush(QtGui.QColor("#555a63"), Qt.BDiagPattern))
        p.setPen(QtGui.QPen(QtGui.QColor("#8b919c"), 1))
        p.drawRect(QtCore.QRectF(x0, y0, w, h))
        g = r.gradient()
        if g is not None and g["size"] > 0:
            cx, cy = x0 + w / 2, y0 + h / 2
            t = math.radians(g["angle"])
            length = 0.4 * min(w, h)
            ex, ey = cx + length * math.cos(t), cy - length * math.sin(t)
            pen = QtGui.QPen(QtGui.QColor("#ffd200"), 2)
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(ex, ey))
            for d in (150, -150):
                a = t + math.radians(d)
                p.drawLine(
                    QtCore.QPointF(ex, ey),
                    QtCore.QPointF(ex + 8 * math.cos(a), ey - 8 * math.sin(a)),
                )
        p.end()


class GalaxyCard(QtWidgets.QGroupBox):
    """One galaxy of the pair: its centre, radii and how it is fitted."""

    pick_requested = QtCore.Signal(str)  # "centre", "r0" or "r1"
    changed = QtCore.Signal()

    def __init__(self, cid: str, parent=None):
        super().__init__(parent)
        self.cid = cid
        colour = COLOURS[cid]
        self.setTitle(f"Galaxy {cid}")
        self.setStyleSheet(
            f"QGroupBox {{ border: 2px solid {colour}; border-radius: 6px; margin-top: 10px; "
            f"padding-top: 6px }} QGroupBox::title {{ color: {colour}; subcontrol-origin: "
            "margin; left: 8px; font-weight: 700 }"
        )
        form = QtWidgets.QFormLayout(self)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText(f"a name for {cid} (optional)")
        form.addRow("Name", self.name)
        self.x0 = number(
            0, 1e6, 0, tip="X0: ELLIPROF's coordinates; the first pixel's centre is 0.5"
        )
        self.y0 = number(
            0, 1e6, 0, tip="Y0: ELLIPROF's coordinates; the first pixel's centre is 0.5"
        )
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.x0, 1)
        row.addWidget(self.y0, 1)
        pick = QtWidgets.QToolButton(text=f"Pick {cid}")
        pick.setToolTip(
            f"Click {cid}'s centre on the image. A source found within 3 px takes its centre "
            "(the panel says so); Alt-click places it exactly."
        )
        pick.clicked.connect(lambda: self.pick_requested.emit("centre"))
        row.addWidget(pick)
        form.addRow("Centre", tip(row, "pair_centre"))
        self.match = note()
        form.addRow(self.match)
        self.r0 = number(0.1, 1e5, 3, tip="R0: semi-major axis of the first isophote, pixels")
        self.r1 = number(0.2, 1e5, 100, tip="R1: the final (target) outer radius, pixels")
        self.start = number(
            0, 1e5, 0, tip="START: the R1 of the first fit; it grows to the final R1"
        )
        self.start.setSpecialValueText("= final R1")
        for key, box, label in (
            ("r0", self.r0, "R0"),
            ("start", self.start, "Initial R1"),
            ("r1", self.r1, "Final R1"),
        ):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(box, 1)
            b = QtWidgets.QToolButton(text="Pick")
            b.setToolTip(f"Click on the image at the distance {label} from {cid}'s centre")
            b.clicked.connect(lambda _c=False, k=key: self.pick_requested.emit(k))
            row.addWidget(b)
            form.addRow(label, tip(row, "pair_grow" if key == "start" else key))
        self.growth = QtWidgets.QComboBox()
        self.growth.addItems(["× factor", "+ pixels", "list of radii"])
        self.growth_value = number(0.01, 1e4, 1.5, decimals=2, step=0.1, tip="the factor or step")
        self.radii = QtWidgets.QLineEdit()
        self.radii.setPlaceholderText("40, 60, 90, 130, 180")
        self.radii.setToolTip("RADII: the R1 of each fit, increasing; the final R1 ends it")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.growth)
        row.addWidget(self.growth_value, 1)
        row.addWidget(self.radii, 2)
        form.addRow("Growth", tip(row, "pair_grow"))
        self.schedule_note = note()
        form.addRow(self.schedule_note)
        self.growth.currentIndexChanged.connect(self._growth_mode)
        self._growth_mode(0)
        for w in (self.start, self.r1, self.r0, self.growth_value):
            w.valueChanged.connect(lambda _v: self._show_schedule())
        self.radii.textChanged.connect(lambda _t: self._show_schedule())
        self.nr = whole(2, 100, 25, tip="NR: the number of isophotes from R0 to R1")
        self.coarse = note()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.nr)
        row.addWidget(self.coarse, 1)
        form.addRow("NR", tip(row, "nr"))
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem("elliprof's default (3, 4)", "")
        for name in MODES:
            self.mode.addItem(name, name)
        self.mode.setToolTip(
            "The harmonics fitted and put in the model: 3 and 4 are the cos 3θ and cos 4θ "
            "terms; 6 replaces 3 (a 6th-order model is checked for elliprof's PA-wrap "
            "warning, and fails with it); m: the median of each term over all ellipses"
        )
        form.addRow("Harmonics", tip(self.mode, "pair_mode"))
        self.fixctr = QtWidgets.QComboBox()
        self.fixctr.addItems(["Free", "Fixed at X0, Y0", "Median of the fitted centres"])
        self.fixctr.setCurrentIndex(1)  # fixed centres: the more stable decomposition
        form.addRow("Centres", tip(self.fixctr, "pair_fixctr"))
        self.core_mode = QtWidgets.QComboBox()
        self.core_mode.addItems(["Auto", "This radius", "Off"])
        self.core = number(0, 1e4, 30, decimals=1, tip="the semi-major axis of the core, pixels")
        self.core.setEnabled(False)
        self.core_mode.currentIndexChanged.connect(
            lambda i: self.core.setEnabled(i == 1) or self.changed.emit()
        )
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.core_mode)
        row.addWidget(self.core, 1)
        other = "B" if cid == "A" else "A"
        form.addRow(f"Core of {cid}", tip(row, "pair_core"))
        self.core_note = note(f"left out of the fit of {other}")
        form.addRow(self.core_note)
        self.adaptive = QtWidgets.QCheckBox("Adaptive outer radius")
        self.adaptive.setToolTip(
            "If the outer isophotes fail their checks, try a smaller R1 (at most 3 times, "
            "never below half the R1 asked for); every try is logged"
        )
        self.visible = QtWidgets.QCheckBox("Show on the image")
        self.visible.setChecked(True)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.adaptive)
        row.addWidget(self.visible)
        form.addRow(tip(row, "pair_adaptive"))
        for box in (self.x0, self.y0, self.r0, self.r1, self.start):
            box.valueChanged.connect(lambda _v: self.changed.emit())
        self.nr.valueChanged.connect(self._nr)
        self.visible.toggled.connect(lambda _v: self.changed.emit())
        self.core.valueChanged.connect(lambda _v: self.changed.emit())
        self._nr(self.nr.value())

    def _growth_mode(self, index: int) -> None:
        self.growth_value.setVisible(index < 2)
        self.radii.setVisible(index == 2)
        if index == 0 and self.growth_value.value() < 1.01:
            self.growth_value.setValue(1.5)
        if index == 1 and self.growth_value.value() < 2:
            self.growth_value.setValue(20)
        self._show_schedule()

    def growth_keywords(self) -> List[str]:
        index = self.growth.currentIndex()
        if index == 2:
            text = ",".join(x for x in self.radii.text().replace(" ", ",").split(",") if x)
            return [f"RADII={text}"] if text else []
        if not self.r0.value() < self.start.value() < self.r1.value():
            return []
        out = [f"START={self.start.value():g}"]
        out.append(
            f"FACTOR={self.growth_value.value():g}"
            if index == 0
            else f"STEP={self.growth_value.value():g}"
        )
        return out

    def schedule(self) -> List[float]:
        """The R1 of each fit, as the engine will make it (empty if not valid)."""
        from .pair import Component, PairError, components_from

        keys = {"X0": "1", "Y0": "1", "R0": f"{self.r0.value():g}", "R1": f"{self.r1.value():g}"}
        for k in self.growth_keywords():
            key, value = k.split("=", 1)
            keys[key] = value
        try:
            other = {"X0": "500", "Y0": "1", "R0": "1", "R1": "2"}
            return components_from({"A": keys, "B": other})["A"].schedule()
        except PairError:
            return []

    def _show_schedule(self) -> None:
        radii = self.schedule()
        self.schedule_note.setText(
            "R1 per cycle: " + ", ".join(f"{r:g}" for r in radii) if radii else "not valid"
        )
        self.changed.emit()

    def _nr(self, value: int) -> None:
        self.coarse.setText("coarse (teaching) fit" if value < 15 else "")

    def centre(self):
        return self.x0.value(), self.y0.value()

    def set_centre(self, x: float, y: float) -> None:
        self.x0.setValue(round(x, 2))
        self.y0.setValue(round(y, 2))

    def keywords(self) -> str:
        parts = []
        name = self.name.text().strip()
        if name:
            parts.append("NAME=" + shlex.quote(name))
        parts += [f"X0={self.x0.value():g}", f"Y0={self.y0.value():g}"]
        parts += [f"R0={self.r0.value():g}", f"R1={self.r1.value():g}"]
        if self.nr.value() != 25:
            parts.append(f"NR={self.nr.value()}")
        mode = self.mode.currentData()
        if mode:
            parts.append(f"MODE={mode}")
        if self.fixctr.currentIndex():
            parts.append(f"FIXCTR={self.fixctr.currentIndex()}")
        if self.core_mode.currentIndex() == 1:
            parts.append(f"CORE={self.core.value():g}")
        elif self.core_mode.currentIndex() == 2:
            parts.append("CORE=0")
        if self.adaptive.isChecked():
            parts.append("ADAPTIVE")
        parts += self.growth_keywords()
        return " ".join(parts)

    def values(self) -> dict:
        return {
            "name": self.name.text(),
            "x0": self.x0.value(),
            "y0": self.y0.value(),
            "r0": self.r0.value(),
            "r1": self.r1.value(),
            "nr": self.nr.value(),
            "mode": self.mode.currentIndex(),
            "fixctr": self.fixctr.currentIndex(),
            "core_mode": self.core_mode.currentIndex(),
            "core": self.core.value(),
            "adaptive": self.adaptive.isChecked(),
            "start": self.start.value(),
            "growth": self.growth.currentIndex(),
            "growth_value": self.growth_value.value(),
            "radii": self.radii.text(),
        }

    def set_values(self, v: dict) -> None:
        self.name.setText(v.get("name", ""))
        for key in ("x0", "y0", "r0", "r1", "core", "start"):
            if key in v:
                getattr(self, key).setValue(float(v[key]))
        if "nr" in v:
            self.nr.setValue(int(v["nr"]))
        for key in ("mode", "fixctr", "core_mode"):
            if key in v:
                getattr(self, key).setCurrentIndex(int(v[key]))
        if "adaptive" in v:
            self.adaptive.setChecked(bool(v["adaptive"]))
        if "growth" in v:
            self.growth.setCurrentIndex(int(v["growth"]))
        if "growth_value" in v:
            self.growth_value.setValue(float(v["growth_value"]))
        if "radii" in v:
            self.radii.setText(str(v["radii"]))


class PairPanel(QtWidgets.QWidget):
    """The settings of a pair fit, and the sky it uses."""

    sky_requested = QtCore.Signal()
    sky_mask_toggled = QtCore.Signal(bool)  # show the sky-exclusion layer on the image
    pick_requested = QtCore.Signal(str, str)  # "A" or "B"; "centre", "r0" or "r1"
    run_requested = QtCore.Signal()
    stop_requested = QtCore.Signal()
    export_requested = QtCore.Signal()
    blink_requested = QtCore.Signal()
    report_requested = QtCore.Signal()
    iteration_requested = QtCore.Signal(int)  # a call's number (1, 2, ...): show its images
    changed = QtCore.Signal()  # something drawn on the image changed

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.addRow(title_row("Pair fit", "pair", "elliprof"))
        form.addRow(
            note(
                "Two overlapping galaxies, fitted in turn with ELLIPROF: each with the other's "
                "model taken out. Measure the sky, place A and B, then Fit pair."
            )
        )

        # ---- the sky
        form.addRow(heading("1. Sky"))
        self.sky_more = QtWidgets.QGroupBox("Segmentation settings")
        self.sky_more.setCheckable(True)
        self.sky_more.setChecked(True)
        more = QtWidgets.QFormLayout(self.sky_more)
        self.thresh = number(
            0.01,
            50,
            0.5,
            decimals=3,
            step=0.05,
            tip="THRESH: sigma above the level; lower masks more (conservative)",
        )
        self.minarea = whole(1, 10000, 12, tip="MINAREA: connected pixels above the threshold")
        self.grow = number(0, 100, 2, decimals=1, tip="GROW: pixels added around every source")
        self.cells = whole(2, 64, 8, tip="CELLS: boxes per side for the cells and the plane")
        self.back_size = whole(0, 4096, 0, tip="BACK_SIZE: 0 = one level, iterated; > 0 a mesh")
        self.fwhm = number(
            0, 20, 0, decimals=1, step=0.5, tip="FWHM: Gaussian detection filter; 0 = 3 x 3"
        )
        for label, box in (
            ("Threshold (σ)", self.thresh),
            ("Min. area", self.minarea),
            ("Grow (px)", self.grow),
            ("Filter FWHM (px)", self.fwhm),
            ("Mesh (BACK_SIZE)", self.back_size),
            ("Cells per side", self.cells),
        ):
            more.addRow(label, box)
        self.sky_more.toggled.connect(self._fold_sky)
        form.addRow(self.sky_more)
        measure = QtWidgets.QPushButton("Measure sky")
        measure.setToolTip(
            "Every source found down to the threshold is left out (with the ticked fitting "
            "layers), and the sky is the median of the rest (sky N SEGMENT)"
        )
        measure.clicked.connect(self.sky_requested)
        self.show_mask = QtWidgets.QCheckBox("Show the sky mask")
        self.show_mask.setToolTip("Draw the pixels left out of the sky (the layer skymask)")
        self.show_mask.toggled.connect(self.sky_mask_toggled)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(tip(measure, "pair_sky"), 2)
        row.addWidget(self.show_mask, 1)
        form.addRow(row)
        self.sky_card = QtWidgets.QLabel()
        self.sky_card.setWordWrap(True)
        self.sky_card.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.sky_card.setTextFormat(Qt.RichText)
        form.addRow(self.sky_card)
        self.sky_hist = SkyHistogram()
        self.sky_hist.setToolTip(
            "The sky pixels (those the mask leaves): their median in red, ± 1 σ dotted"
        )
        form.addRow(self.sky_hist)
        self.sky_map = SkyMap()
        self.sky_map.set_result(None)
        form.addRow(self.sky_map)
        self.use_sky = QtWidgets.QPushButton("Use this sky")
        self.use_sky.setEnabled(False)
        self.use_sky.setToolTip(
            "The median of the sky pixels as the base level: your decision; the verdict is "
            "advice and is kept with the fit"
        )
        self.use_sky.clicked.connect(self._use_measured)
        enter = QtWidgets.QPushButton("Enter a value…")
        enter.setToolTip("Type a sky level, with a note on where it comes from")
        enter.clicked.connect(self._enter_sky)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.use_sky)
        row.addWidget(enter)
        form.addRow(row)
        self.sky = number(-1e12, 1e12, 0, decimals=4, tip="SKY: the base level")
        self.sky.setReadOnly(True)
        self.sky_source = note("no sky yet")
        form.addRow("Base level", self.sky)
        form.addRow(self.sky_source)
        self.factor = number(
            0.5, 1.5, 1.0, decimals=5, step=0.001, tip="SKYFACTOR: the sky used = factor × base"
        )
        form.addRow("Sky factor", tip(self.factor, "pair_skyfactor"))
        self.sky_used = note()
        form.addRow(self.sky_used)
        self.sky_iter = whole(
            0, 50, 0, tip="SKYITER: refit, each time with the factor from the residual sky"
        )
        self.sky_iter.setSpecialValueText("off")
        form.addRow("Refine the factor", tip(self.sky_iter, "pair_skyfactor"))
        form.addRow(
            note(
                "Refine: after each pair fit, the median m of image − sky − A − B on the sky "
                "pixels gives factor + m / base; up to this many fits, the one with the "
                "smallest |m| kept."
            )
        )
        self.sky.valueChanged.connect(self._sky_used_line)
        self.factor.valueChanged.connect(self._sky_used_line)
        self._sky_used_line()
        self.sky_id = ""  # the measurement the sky comes from ("" = typed)
        self._measurement = None

        # ---- the galaxies
        form.addRow(heading("2. The two galaxies"))
        self.cards = {cid: GalaxyCard(cid) for cid in ("A", "B")}
        for cid, card in self.cards.items():
            card.pick_requested.connect(lambda what, c=cid: self.pick_requested.emit(c, what))
            card.changed.connect(self.changed)
            card.changed.connect(self._summary_line)
            form.addRow(card)
        swap = QtWidgets.QPushButton("⇄ Swap A and B")
        swap.setToolTip(
            "A is fitted first. Swapping changes which galaxy that is; with a finite number of "
            "cycles the result can depend on the order."
        )
        swap.clicked.connect(self.swap)
        form.addRow(swap)

        # ---- fitting masks
        form.addRow(heading("3. Fitting masks"))
        self.layers = QtWidgets.QListWidget()
        self.layers.setMinimumHeight(90)
        self.layers.setToolTip(
            "Layers left out of every fit: those of role fit, and those without a role. Sky "
            "layers are not listed: they cover the galaxies too."
        )
        self._ticked = set()
        self.layers.itemChanged.connect(self._tick_changed)
        form.addRow(tip(self.layers, "pair_masks"))

        # ---- the run
        form.addRow(heading("4. Run"))
        form.addRow(
            note(
                "A on D, B on D − M_A, then A on D − M_B and B on D − M_A, cycle after cycle, "
                "the radii growing to their final R1. The galaxies are not masked from each "
                "other: the companion's current model is subtracted from the original image."
            )
        )
        self.cycles = whole(
            1, 50, 6, tip="CYCLES: the most A → B cycles at the final radii (after growing)"
        )
        self.niter = whole(1, 1000, 10, tip="NITER: iterations inside each elliprof call")
        form.addRow("Max cycles at final R1", tip(self.cycles, "pair_cycles"))
        self.converge = number(
            0.01, 50, 2.0, decimals=2, step=0.5, tip="CONVERGE: largest change of A, B and A + B, %"
        )
        self.consecutive = whole(0, 20, 2, tip="CONSECUTIVE: converged cycles in a row to stop")
        self.consecutive.setSpecialValueText("never stop early")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("change <"))
        row.addWidget(self.converge, 1)
        row.addWidget(QtWidgets.QLabel("% for"))
        row.addWidget(self.consecutive, 1)
        row.addWidget(QtWidgets.QLabel("cycles"))
        form.addRow("Converged", tip(row, "pair_converge"))
        self.on_fail = QtWidgets.QComboBox()
        self.on_fail.addItems(
            [
                "keep the last valid model, try smaller steps up",
                "keep the last valid model, stay at its radius",
                "keep the last valid model, try the radius again",
                "stop and keep the last valid models",
            ]
        )
        form.addRow("If a fit fails", tip(self.on_fail, "pair_onfail"))
        self.damp = number(0.05, 1.0, 0.5, decimals=2, step=0.1, tip="DAMP: share of the new model")
        self.damp.setSpecialValueText("off (1)")
        form.addRow("Damping", tip(self.damp, "pair_damp"))
        self.search = QtWidgets.QGroupBox("Search for the best fit")
        self.search.setCheckable(True)
        self.search.setChecked(True)
        grid = QtWidgets.QGridLayout(self.search)
        self.try_damp = QtWidgets.QLineEdit("1, 0.5")
        self.try_fix = QtWidgets.QComboBox()
        self.try_fix.addItems(["fixed and free", "fixed only", "free only"])
        self.try_factor = QtWidgets.QLineEdit("1.5, 1.25")
        for row, (label, w, tipkey) in enumerate(
            (
                ("Damping values", self.try_damp, "TRYDAMP"),
                ("Centres", self.try_fix, "TRYFIX"),
                ("Growth factors", self.try_factor, "TRYFACTOR"),
            )
        ):
            w.setToolTip(tipkey)
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            grid.addWidget(w, row, 1)
        grid.addWidget(
            note(
                "Every combination is fitted and one is kept: the stable fit with the smallest "
                "residual outside the cores, with fixed centres unless free centres lower it by "
                "more than 0.01 of the noise. On simulated pairs this gave the best residual "
                "and the most accurate A and B of the rules tried."
            ),
            3,
            0,
            1,
            2,
        )
        form.addRow(tip(self.search, "pair_search"))
        form.addRow("NITER (each call)", tip(self.niter, "niter"))
        self.masking = QtWidgets.QCheckBox("Also mask a core around the companion (older method)")
        self.masking.setToolTip("CORES: by default nothing is masked around the companion")
        form.addRow(self.masking)
        self.calls = note()
        form.addRow(self.calls)
        self.cycles.valueChanged.connect(self._summary_line)
        for card in self.cards.values():
            card.changed.connect(self._summary_line)
        self.advanced = QtWidgets.QGroupBox("Advanced")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        adv = QtWidgets.QFormLayout(self.advanced)
        self.rlaw = QtWidgets.QComboBox()
        self.rlaw.addItems(["0: linear", "1: logarithmic", "2: r^¼ (default)"])
        self.rlaw.setCurrentIndex(2)
        self.gain = number(0.01, 10, 1.0, decimals=2, step=0.1, tip="GAIN: 1 by default")
        self.tie = whole(-50, 10, -1, tip="TIE: -1 = no smoothing of the parameters with radius")
        self.rmstar = QtWidgets.QCheckBox("RMSTAR")
        self.corecap = number(0, 1e4, 50, decimals=1, tip="the largest core, pixels")
        self.timeout = number(1, 36000, 300, decimals=0, tip="seconds per elliprof call")
        self.bootstrap = QtWidgets.QCheckBox("Partial bootstrap start if a first fit fails")
        self.bootstrap.setChecked(True)
        self.views = QtWidgets.QCheckBox("Also image − sky − B and image − sky − A")
        self.views.setChecked(True)
        for label, w in (
            ("RLAW", self.rlaw),
            ("GAIN", self.gain),
            ("TIE", self.tie),
            ("", self.rmstar),
            ("Core cap (px)", self.corecap),
            ("Timeout (s)", self.timeout),
            ("", self.bootstrap),
            ("", self.views),
        ):
            adv.addRow(label, w)
        self.advanced.toggled.connect(self._fold_advanced)
        self._fold_advanced(False)
        form.addRow(self.advanced)
        buttons = QtWidgets.QHBoxLayout()
        self.run = QtWidgets.QPushButton("Fit pair")
        self.run.setDefault(True)
        self.run.clicked.connect(self.run_requested)
        self.stop = QtWidgets.QPushButton("Stop")
        self.stop.setEnabled(False)
        self.stop.setToolTip(
            "Stop after the elliprof call running now; the last cycle in "
            "which both fits passed is kept"
        )
        self.stop.clicked.connect(self.stop_requested)
        buttons.addWidget(self.run, 2)
        buttons.addWidget(self.stop, 1)
        form.addRow(buttons)
        self.progress = note()
        form.addRow(self.progress)
        self.iterations = plain_table()
        self.iterations.setColumnCount(6)
        self.iterations.setHorizontalHeaderLabels(
            ["cycle", "galaxy", "input", "R1", "status", "centre drift"]
        )
        self.iterations.setMinimumHeight(170)
        self.iterations.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        self.iterations.verticalHeader().setDefaultSectionSize(22)
        self.iterations.setToolTip(
            "Every elliprof call, as it runs. Double-click a row to see that fit's model and "
            "the image it was fitted on (D − the companion's model then)."
        )
        self.iterations.cellDoubleClicked.connect(
            lambda row, _col: self.iteration_requested.emit(row + 1)
        )
        form.addRow(self.iterations)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.RichText)
        form.addRow(self.status)
        # ---- the residual sky
        form.addRow(heading("5. Residual sky"))
        self.resid_card = note("after a fit: the residual on the sky pixels")
        self.resid_card.setTextFormat(Qt.RichText)
        form.addRow(self.resid_card)
        self.resid_hist = SkyHistogram()
        self.resid_hist.setToolTip(
            "image − sky − A − B on the sky pixels: its median in red, zero dashed. The "
            "models extend beyond R1, so their outer light counts."
        )
        form.addRow(self.resid_hist)
        self.next_factor = QtWidgets.QPushButton("Use the next factor")
        self.next_factor.setEnabled(False)
        self.next_factor.setToolTip("Set the factor to factor + median / base; then Fit pair")
        self.next_factor.clicked.connect(self._use_next_factor)
        form.addRow(self.next_factor)
        self._next = None
        row = QtWidgets.QHBoxLayout()
        self.blink = QtWidgets.QPushButton("Blink the result")
        self.blink.setToolTip(
            "Blink image − sky, A + B, image − sky − B and image − sky − A on one scale "
            "(the arrow keys step through them)"
        )
        self.blink.clicked.connect(self.blink_requested)
        self.export = QtWidgets.QPushButton("Export pair…")
        self.export.setToolTip("Write everything about the last pair fit to a folder")
        self.export.clicked.connect(self.export_requested)
        row.addWidget(self.blink)
        row.addWidget(self.export)
        form.addRow(row)
        self.report = QtWidgets.QPushButton("Copy report")
        self.report.setToolTip(
            "Copy a plain-text report of the last pair fit (settings, every call, growth, "
            "cycles, sky, residual rings around each galaxy and hints) to paste anywhere"
        )
        self.report.clicked.connect(self.report_requested)
        form.addRow(self.report)
        self._summary_line()

    # ---- folding
    def _fold_sky(self, on: bool) -> None:
        for w in self.sky_more.findChildren(QtWidgets.QWidget):
            w.setVisible(on)

    def _fold_advanced(self, on: bool) -> None:
        for w in self.advanced.findChildren(QtWidgets.QWidget):
            w.setVisible(on)

    def _summary_line(self) -> None:
        grow = max(len(c.schedule()) for c in self.cards.values()) - 1 if self.cards else 0
        grow = max(grow, 0)
        n = 2 + 2 * (grow + self.cycles.value())
        self.calls.setText(
            f"at most {n} elliprof calls: 2 to start, {2 * grow} while the radii grow, then "
            f"A and B in each cycle at the final radii"
        )

    # ---- the sky
    def _sky_used_line(self) -> None:
        f = self.factor.value()
        self.sky_used.setText(
            f"sky used: {f:.5f} × {self.sky.value():.6g} = {f * self.sky.value():.6g}"
        )

    def sky_command(self, n: int) -> str:
        parts = [f"sky {n} SEGMENT"]
        defaults = {
            "THRESH": 0.5,
            "MINAREA": 12,
            "GROW": 2.0,
            "CELLS": 8,
            "BACK_SIZE": 0,
            "FWHM": 0.0,
        }
        values = {
            "THRESH": self.thresh.value(),
            "MINAREA": self.minarea.value(),
            "GROW": self.grow.value(),
            "CELLS": self.cells.value(),
            "BACK_SIZE": self.back_size.value(),
            "FWHM": self.fwhm.value(),
        }
        parts += [f"{k}={v:g}" for k, v in values.items() if v != defaults[k]]
        layers = self.ticked_layers()
        if layers:
            parts.append("MASK=" + ",".join(layers))
        return " ".join(parts)

    def show_sky(self, result, stale: Optional[str], values=None) -> None:
        """Show a sky measurement (None: none), whether it still fits the image, and the
        histogram of its sky pixels (``values``)."""
        self._measurement = result
        self.sky_map.set_result(result)
        if result is None or values is None:
            self.sky_hist.show_values([], float("nan"), float("nan"))
        else:
            self.sky_hist.show_values(values, result.level, result.sigma)
        if result is None:
            self.sky_card.setText("")
            self.use_sky.setEnabled(False)
            return
        colour = STATE_COLOURS.get(result.state, "#8b919c")
        lines = [f"<b style='color:{colour}'>{result.state}</b> · measurement {result.id}"]
        if stale:
            lines.append(f"<span style='color:#f85149'>stale: {stale}</span>")
        if np.isfinite(result.level):
            lines.append(
                f"<b>median {result.level:.6g}</b>, clipped mean {result.mean:.6g}, "
                f"σ {result.sigma:.4g}"
            )
        if result.state != "no_estimate":
            lines.append(
                f"sampling ±{result.sampling_error:.3g} (approximate); spatial: cell range "
                f"{result.cell_range:.3g}"
            )
            g = result.gradient()
            if g is not None:
                lines.append(
                    f"gradient {g['size']:.3g} per 1000 px toward {g['angle']:.0f}° "
                    f"(image-axis angle); {result.plane_change:.3g} across the frame"
                )
        lines.append(
            f"{100 * result.support_fraction:.1f} % of the pixels used; "
            f"{len(result.usable_cells)} of {len(result.cells)} cells usable"
        )
        for reason in result.reasons:
            lines.append(f"• {reason}")
        lines.append(
            "<i>The verdict is advice. Usable support is needed, not proof: a halo too faint "
            "to detect can remain.</i>"
        )
        self.sky_card.setText("<br>".join(lines))
        self.use_sky.setEnabled(bool(np.isfinite(result.level)) and not stale)

    def _use_measured(self) -> None:
        r = self._measurement
        if r is None:
            return
        self.sky.setValue(r.level)
        self.factor.setValue(1.0)
        self.sky_id = r.id
        s = r.settings
        self.sky_source.setText(
            f"measured: median of the sky pixels, {s['THRESH']:g}σ segmentation; {r.id} "
            f"({r.state})"
        )

    def _enter_sky(self) -> None:
        value, ok = QtWidgets.QInputDialog.getDouble(
            self, "The sky", "Sky level (the image's units):", self.sky.value(), -1e12, 1e12, 4
        )
        if not ok:
            return
        why, ok = QtWidgets.QInputDialog.getText(
            self, "The sky", "Where does this value come from? (kept with the fit)"
        )
        if not ok:
            return
        self.sky.setValue(value)
        self.factor.setValue(1.0)
        self.sky_id = ""
        self.sky_note = why.strip()
        self.sky_source.setText("typed" + (f": {why.strip()}" if why.strip() else ""))

    def set_sky_value(self, value: float, note_text: str) -> None:
        self.sky.setValue(value)
        self.factor.setValue(1.0)
        self.sky_id = ""
        self.sky_source.setText(note_text)

    def show_residual_sky(self, check: Optional[dict], values=None, iterations=()) -> None:
        """After a fit: the residual on the sky pixels, its histogram, the next factor."""
        self._next = None
        self.next_factor.setEnabled(False)
        if not check or "clipped_median" not in check:
            self.resid_card.setText(
                "after a fit: the residual on the sky pixels (it needs a current sky "
                "measurement of the image)"
            )
            self.resid_hist.show_values([], float("nan"), float("nan"))
            return
        m, s = check["clipped_median"], check["clipped_sigma"]
        text = (
            f"median <b>{m:+.4g}</b> (σ {s:.4g}, {check['clipped_pixels']} px; "
            f"{m / s:+.3f} σ) with factor {check.get('clipped_factor', 1.0):.5f}"
            if s
            else f"median {m:+.4g}"
        )
        if "clipped_next_factor" in check:
            self._next = check["clipped_next_factor"]
            text += f"<br>next factor {self._next:.5f} → sky {check['clipped_next_sky']:.6g}"
            self.next_factor.setEnabled(0.5 <= self._next <= 1.5)
        if iterations:
            text += f"<br>{len(iterations)} pair fits refined the factor (Results ▸ Pair ▸ Sky)"
        self.resid_card.setText(text)
        self.resid_hist.show_values(values if values is not None else [], m, s, zero=True)

    def _use_next_factor(self) -> None:
        if self._next is not None:
            self.factor.setValue(self._next)

    # ---- masks
    def set_mask_layers(self, layers) -> None:
        """List the layers that can be used for fitting (role fit, or none)."""
        self.layers.blockSignals(True)
        self.layers.clear()
        for m in layers:
            if m.role not in ("fit", ""):
                continue
            item = QtWidgets.QListWidgetItem(m.name + ("" if m.role else "  (no role)"))
            item.setData(Qt.UserRole, m.name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if m.name in self._ticked else Qt.Unchecked)
            self.layers.addItem(item)
        self.layers.blockSignals(False)

    def tick(self, *names) -> None:
        self._ticked |= set(names)
        for k in range(self.layers.count()):
            item = self.layers.item(k)
            if item.data(Qt.UserRole) in names:
                item.setCheckState(Qt.Checked)

    def _tick_changed(self, item) -> None:
        name = item.data(Qt.UserRole)
        if item.checkState() == Qt.Checked:
            self._ticked.add(name)
        else:
            self._ticked.discard(name)

    def ticked_layers(self) -> List[str]:
        return [
            self.layers.item(k).data(Qt.UserRole)
            for k in range(self.layers.count())
            if self.layers.item(k).checkState() == Qt.Checked
        ]

    # ---- the command
    def command(self, n: int) -> str:
        parts = [f"pair {n}", f"SKY={self.sky.value():.10g}"]
        if self.factor.value() != 1.0:
            parts.append(f"SKYFACTOR={self.factor.value():.6g}")
        if self.sky_iter.value():
            parts.append(f"SKYITER={self.sky_iter.value()}")
        layers = self.ticked_layers()
        if layers:
            parts.append("MASK=" + ",".join(layers))
        if not self.masking.isChecked():
            parts.append("CORES=0")
        parts.append(f"CYCLES={self.cycles.value()}")
        if self.consecutive.value():
            parts.append(f"CONVERGE={self.converge.value() / 100:g}")
            parts.append(f"CONSECUTIVE={self.consecutive.value()}")
        on_fail = ("reduce", "hold", "retry", "stop")[self.on_fail.currentIndex()]
        if on_fail != "reduce":
            parts.append(f"ONFAIL={on_fail}")
        if self.damp.value() < 1:
            parts.append(f"DAMP={self.damp.value():g}")
        if self.search.isChecked():
            damps = ",".join(x.strip() for x in self.try_damp.text().split(",") if x.strip())
            factors = ",".join(x.strip() for x in self.try_factor.text().split(",") if x.strip())
            if damps:
                parts.append(f"TRYDAMP={damps}")
            parts.append("TRYFIX=" + ("1,0", "1", "0")[self.try_fix.currentIndex()])
            if factors:
                parts.append(f"TRYFACTOR={factors}")
        if self.niter.value() != 10:
            parts.append(f"NITER={self.niter.value()}")
        if self.rlaw.currentIndex() != 2:
            parts.append(f"RLAW={self.rlaw.currentIndex()}")
        if self.gain.value() != 1.0:
            parts.append(f"GAIN={self.gain.value():g}")
        if self.tie.value() != -1:
            parts.append(f"TIE={self.tie.value()}")
        if self.rmstar.isChecked():
            parts.append("RMSTAR")
        if self.corecap.value() != 50:
            parts.append(f"CORECAP={self.corecap.value():g}")
        if self.timeout.value() != 300:
            parts.append(f"TIMEOUT={self.timeout.value():g}")
        if not self.bootstrap.isChecked():
            parts.append("BOOTSTRAP=0")
        if self.views.isChecked():
            parts.append("VIEWS")
        parts.append("A: " + self.cards["A"].keywords())
        parts.append("B: " + self.cards["B"].keywords())
        return " ".join(parts)

    def swap(self) -> None:
        a, b = self.cards["A"].values(), self.cards["B"].values()
        self.cards["A"].set_values(b)
        self.cards["B"].set_values(a)
        self.changed.emit()

    def clear_iterations(self) -> None:
        self.iterations.setRowCount(0)

    def add_iteration(self, info: dict) -> None:
        """One row of the iteration table (from a call's progress message)."""
        row = self.iterations.rowCount()
        self.iterations.insertRow(row)
        status = "done" if info.get("passed") else f"FAILED: {info.get('reason', '')}"
        drift = info.get("drift")
        cells = [
            str(info.get("cycle", "")),
            info.get("component", ""),
            info.get("input", ""),
            f"{info.get('r1', 0):.1f}",
            status,
            "" if drift is None else f"{drift:.2f} px",
        ]
        for col, text in enumerate(cells):
            item = QtWidgets.QTableWidgetItem(text)
            if col == 1:
                item.setForeground(QtGui.QColor(COLOURS.get(text, "#c9d1d9")))
            if col == 4 and not info.get("passed"):
                item.setForeground(QtGui.QColor("#f85149"))
            self.iterations.setItem(row, col, item)
        self.iterations.scrollToBottom()

    def show_history(self, result) -> None:
        """The iteration table of a finished fit: the calls of the fit kept."""
        self.clear_iterations()
        if result is None:
            return
        for h, c in zip(result.history, result.calls):
            self.add_iteration(
                {
                    "cycle": h["cycle"],
                    "component": h["galaxy"],
                    "input": h["input"],
                    "r1": h["r1"],
                    "passed": h["passed"],
                    "reason": h["reason"],
                    "drift": c.centre_drift,
                }
            )

    def set_running(self, on: bool) -> None:
        self.run.setEnabled(not on)
        self.stop.setEnabled(on)
        if on:
            self.status.setText("")
            self.clear_iterations()

    def show_result(self, result) -> None:
        if result is None:
            self.status.setText("")
            return
        colour = STATUS_COLOURS.get(result.status, "#8b919c")
        text = f"<b style='color:{colour}'>{result.status}</b>"
        if result.stability_reason:
            text += f"<br>{result.stability_reason}"
        if result.checkpoint_cycle is not None:
            text += f"<br>result: the checkpoint of cycle {result.checkpoint_cycle}"
        for cid, c in result.commits.items():
            if c.r1_call < c.requested_r1 - 1e-6:
                text += (
                    f"<br><span style='color:#d29922'>{cid} reached R1 {c.r1_call:.1f} of "
                    f"{c.requested_r1:g}</span>"
                )
        for note_text in result.notes_growth:
            text += f"<br>• {note_text}"
        self.status.setText(text)


class SkyHistogram(pg.PlotWidget):
    """A histogram of sky pixels (or of the residual on them), median ± 5 sigma, with the
    median in red, ± 1 sigma dotted and, for a residual, zero dashed."""

    def __init__(self, parent=None, height: int = 150):
        super().__init__(parent)
        self.setMinimumHeight(height)
        self.setMaximumHeight(height + 60)
        self.showGrid(x=True, y=False, alpha=0.2)
        self.getAxis("left").enableAutoSIPrefix(False)
        self.getAxis("bottom").enableAutoSIPrefix(False)
        self.getAxis("left").setStyle(showValues=False)
        self.setMouseEnabled(x=True, y=False)
        self.setVisible(False)

    def show_values(self, values, median: float, sigma: float, zero: bool = False) -> None:
        self.clear()
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        if values.size < 10 or not (np.isfinite(median) and np.isfinite(sigma) and sigma > 0):
            self.setVisible(False)
            return
        lo, hi = median - 5 * sigma, median + 5 * sigma
        if zero:
            lo, hi = min(lo, -1 * sigma), max(hi, 1 * sigma)
        counts, edges = np.histogram(values, bins=60, range=(lo, hi))
        self.plot(
            edges,
            counts,
            stepMode="center",
            fillLevel=0,
            brush=pg.mkBrush(63, 185, 80, 110),
            pen=pg.mkPen("#3fb950", width=1),
        )
        self.addItem(pg.InfiniteLine(median, angle=90, pen=pg.mkPen("#f85149", width=2)))
        for s in (-sigma, sigma):
            self.addItem(
                pg.InfiniteLine(median + s, angle=90, pen=pg.mkPen("#f85149", style=Qt.DotLine))
            )
        if zero:
            self.addItem(
                pg.InfiniteLine(
                    0.0, angle=90, pen=pg.mkPen("#c9d1d9", width=1.5, style=Qt.DashLine)
                )
            )
        self.setXRange(lo, hi, padding=0)
        self.setVisible(True)


class PairResultsPanel(QtWidgets.QWidget):
    """A pair fit's results: the profiles of A and B, the change from cycle to cycle, and
    the summary."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.record = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        holder = QtWidgets.QWidget()
        flow = FlowLayout(holder, spacing=10)
        self.quantity = QtWidgets.QComboBox()
        self.quantity.addItems(
            [
                "Intensity I0",
                "Ellipticity",
                "Position angle",
                "Centre offset",
                "cos 4θ amplitude I4",
            ]
        )
        self.quarter = QtWidgets.QCheckBox("R^¼ axis")
        for w in (self.quantity, self.quarter):
            flow.addWidget(w)
        flow.addWidget(guide.button("pair"))
        lay.addWidget(holder)
        self.tabs = QtWidgets.QTabWidget()
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend(offset=(-10, 10))
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.tabs.addTab(self.plot, "Profiles")
        cycles = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(cycles)
        cl.setContentsMargins(0, 0, 0, 0)
        self.cycle_plot = pg.PlotWidget()
        self.cycle_plot.showGrid(x=True, y=True, alpha=0.25)
        self.cycle_plot.addLegend(offset=(-10, 10))
        self.cycle_plot.getAxis("left").enableAutoSIPrefix(False)
        self.cycle_table = plain_table()
        cl.addWidget(self.cycle_plot, 2)
        cl.addWidget(self.cycle_table, 1)
        self.tabs.addTab(cycles, "Cycles")
        sky = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(sky)
        sl.setContentsMargins(0, 0, 0, 0)
        self.sky_note = QtWidgets.QLabel()
        self.sky_note.setWordWrap(True)
        self.resid_hist = SkyHistogram(height=200)
        self.resid_hist.setMaximumHeight(16777215)
        self.sky_table = plain_table()
        sl.addWidget(self.sky_note)
        sl.addWidget(self.resid_hist, 2)
        sl.addWidget(self.sky_table, 1)
        self.tabs.addTab(sky, "Sky")
        self.search_table = plain_table()
        self.tabs.addTab(self.search_table, "Search")
        lay.addWidget(self.tabs, 1)
        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(170)
        lay.addWidget(self.summary)
        self.quantity.currentTextChanged.connect(lambda _t: self.draw())
        self.quarter.toggled.connect(lambda _t: self.draw())

    def set_record(self, record, summary: str = "", residual=None) -> None:
        self.record = record
        self.summary.setPlainText(summary)
        self.draw()
        self._cycles()
        self._sky(residual)
        self._search()

    def _search(self) -> None:
        rows = self.record.result.search if self.record else []
        headers = ["variant", "status", "score", "rings A", "rings B", "R1 A", "R1 B", "sky", ""]
        self.search_table.setColumnCount(len(headers))
        self.search_table.setHorizontalHeaderLabels(headers)
        self.search_table.setRowCount(len(rows))

        def f(v, fmt):
            return "" if v is None else format(v, fmt)

        for i, r in enumerate(rows):
            cells = [
                r["variant"],
                r["status"],
                f(r["score"], ".4f"),
                f(r["A"], ".4f"),
                f(r["B"], ".4f"),
                f(r["A_r1"], ".1f"),
                f(r["B_r1"], ".1f"),
                f(r["sky"], ".6g"),
                ("kept: " + self.record.result.search_choice) if r.get("chosen") else "",
            ]
            for j, text in enumerate(cells):
                self.search_table.setItem(i, j, QtWidgets.QTableWidgetItem(text))

    def _sky(self, residual) -> None:
        """The residual on the sky pixels, and the sky-factor fits."""
        check = self.record.result.sky_check if self.record else {}
        if "clipped_median" in check and residual is not None:
            m, s = check["clipped_median"], check["clipped_sigma"]
            self.sky_note.setText(
                f"image − sky − A − B on the {check['clipped_pixels']} sky pixels: median "
                f"{m:+.4g} (red), σ {s:.4g}; zero dashed"
            )
            self.resid_hist.show_values(residual, m, s, zero=True)
        else:
            self.sky_note.setText("No residual on sky pixels: measure the sky of this image first.")
            self.resid_hist.show_values([], float("nan"), float("nan"))
        rows = self.record.result.sky_iterations if self.record else []
        headers = ["fit", "factor", "sky", "median", "σ", "status", ""]
        self.sky_table.setColumnCount(len(headers))
        self.sky_table.setHorizontalHeaderLabels(headers)
        self.sky_table.setRowCount(len(rows))
        for i, e in enumerate(rows):
            cells = [
                str(e["iteration"]),
                f"{e['factor']:.5f}",
                f"{e['sky']:.6g}",
                "" if e.get("median") is None else f"{e['median']:+.4g}",
                "" if e.get("sigma") is None else f"{e['sigma']:.4g}",
                e["status"],
                ("kept" if e.get("chosen") else "") + (f" {e['note']}" if e.get("note") else ""),
            ]
            for j, text in enumerate(cells):
                self.sky_table.setItem(i, j, QtWidgets.QTableWidgetItem(text))
        self.sky_table.setVisible(bool(rows))

    def draw(self) -> None:
        self.plot.clear()
        if self.record is None:
            return
        result, snap = self.record.result, self.record.job.snap
        what = self.quantity.currentText()
        logarithmic = what == "Intensity I0"
        self.plot.setLogMode(x=False, y=logarithmic)
        for cid in ("A", "B"):
            commit = result.commits.get(cid)
            if commit is None:
                continue
            p = commit.profile
            r = p["Rmaj"].to_numpy(float)
            if what == "Intensity I0":
                y = p["I0"].to_numpy(float)
            elif what == "Ellipticity":
                y = p["ellip"].to_numpy(float)
            elif what == "Position angle":
                y = (p["alpha"].to_numpy(float) - 90.0) % 180.0  # from +x on the image
            elif what == "Centre offset":
                comp = snap.components[cid]
                y = np.hypot(p["x0"].to_numpy(float) - comp.x0, p["y0"].to_numpy(float) - comp.y0)
            else:
                y = p["I4"].to_numpy(float)
            x = r**0.25 if self.quarter.isChecked() else r
            if logarithmic:
                keep = y > 0
                x, y = x[keep], y[keep]
            name = snap.components[cid].label
            self.plot.plot(
                x,
                y,
                pen=pg.mkPen(COLOURS[cid], width=1.6),
                symbol="o",
                symbolSize=5,
                symbolBrush=COLOURS[cid],
                name=f"{cid}: {name}" if name != cid else cid,
            )
        self.plot.setLabel(
            "bottom", "Rmaj^¼ (pixels^¼)" if self.quarter.isChecked() else "Rmaj (pixels)"
        )
        label = {
            "Position angle": "angle of the major axis from +x (deg)",
            "Centre offset": "fitted centre − picked centre (px)",
        }.get(what, what)
        self.plot.setLabel("left", label)

    def _cycles(self) -> None:
        self.cycle_plot.clear()
        if self.record is None:
            self.cycle_table.setRowCount(0)
            return
        result = self.record.result
        m = result.metrics
        if m:
            k = np.array([x["cycle"] for x in m], float)
            for key, colour, label in (
                ("dA", COLOURS["A"], "A"),
                ("dB", COLOURS["B"], "B"),
                ("dT", "#e6e6e6", "A + B"),
            ):
                v = 100 * np.array([x[key] if x.get(key) is not None else np.nan for x in m], float)
                ok = np.isfinite(v)
                if ok.any():
                    self.cycle_plot.plot(
                        k[ok],
                        v[ok],
                        pen=pg.mkPen(colour, width=1.6),
                        symbol="o",
                        symbolSize=6,
                        symbolBrush=colour,
                        name=label,
                    )
            line = pg.InfiniteLine(
                pos=100 * 0.02, angle=0, pen=pg.mkPen("#d29922", style=Qt.DashLine)
            )
            self.cycle_plot.addItem(line)
            self.cycle_plot.getAxis("bottom").setTicks([[(c, str(int(c))) for c in k]])
            self.cycle_plot.setLabel("bottom", "cycle")
            self.cycle_plot.setLabel("left", "change of the model (%), 2 % dashed")
        columns = ["cycle", "A", "B", "A model", "B model", "ΔA", "ΔB", "ΔA+B", "why"]

        def pct(v):
            return "n/a" if v is None else f"{100 * v:.2f} %"

        rows = [
            [
                x["cycle"],
                "ok" if x["A_ok"] else "failed",
                "ok" if x["B_ok"] else "failed",
                x["A_tag"],
                x["B_tag"],
                pct(x.get("dA")),
                pct(x.get("dB")),
                pct(x.get("dT")),
                x.get("reason", ""),
            ]
            for x in m
        ]
        t = self.cycle_table
        t.clear()
        t.setColumnCount(len(columns))
        t.setRowCount(len(rows))
        t.setHorizontalHeaderLabels(columns)
        for r, values in enumerate(rows):
            for c, value in enumerate(values):
                t.setItem(r, c, QtWidgets.QTableWidgetItem(str(value)))
        t.resizeColumnsToContents()
