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
        self.r1 = number(0.2, 1e5, 100, tip="R1: semi-major axis of the last isophote, pixels")
        for key, box in (("r0", self.r0), ("r1", self.r1)):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(box, 1)
            b = QtWidgets.QToolButton(text="Pick")
            b.setToolTip(f"Click on the image at the distance {key.upper()} from {cid}'s centre")
            b.clicked.connect(lambda _c=False, k=key: self.pick_requested.emit(k))
            row.addWidget(b)
            form.addRow(key.upper(), tip(row, key))
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
        for box in (self.x0, self.y0, self.r0, self.r1):
            box.valueChanged.connect(lambda _v: self.changed.emit())
        self.nr.valueChanged.connect(self._nr)
        self.visible.toggled.connect(lambda _v: self.changed.emit())
        self.core.valueChanged.connect(lambda _v: self.changed.emit())
        self._nr(self.nr.value())

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
        }

    def set_values(self, v: dict) -> None:
        self.name.setText(v.get("name", ""))
        for key in ("x0", "y0", "r0", "r1", "core"):
            if key in v:
                getattr(self, key).setValue(float(v[key]))
        if "nr" in v:
            self.nr.setValue(int(v["nr"]))
        for key in ("mode", "fixctr", "core_mode"):
            if key in v:
                getattr(self, key).setCurrentIndex(int(v[key]))
        if "adaptive" in v:
            self.adaptive.setChecked(bool(v["adaptive"]))


class PairPanel(QtWidgets.QWidget):
    """The settings of a pair fit, and the sky it uses."""

    sky_requested = QtCore.Signal()
    pick_requested = QtCore.Signal(str, str)  # "A" or "B"; "centre", "r0" or "r1"
    run_requested = QtCore.Signal()
    stop_requested = QtCore.Signal()
    export_requested = QtCore.Signal()
    blink_requested = QtCore.Signal()
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
        measure = QtWidgets.QPushButton("Measure sky (0.5σ segmentation)")
        measure.setToolTip(
            "Every source found down to the threshold is left out (with the ticked fitting "
            "layers), and the sky is measured on the rest (sky N SEGMENT)"
        )
        measure.clicked.connect(self.sky_requested)
        form.addRow(tip(measure, "pair_sky"))
        self.sky_more = QtWidgets.QGroupBox("Segmentation settings")
        self.sky_more.setCheckable(True)
        self.sky_more.setChecked(False)
        more = QtWidgets.QFormLayout(self.sky_more)
        self.thresh = number(
            0.05, 50, 0.5, decimals=2, step=0.1, tip="THRESH: sigma above the level"
        )
        self.minarea = whole(1, 10000, 12, tip="MINAREA: connected pixels above the threshold")
        self.grow = number(0, 100, 2, decimals=1, tip="GROW: pixels added around every source")
        self.cells = whole(2, 64, 8, tip="CELLS: boxes per side for the cells and the plane")
        self.back_size = whole(0, 4096, 0, tip="BACK_SIZE: 0 = one level, iterated; > 0 a mesh")
        for label, box in (
            ("Threshold (σ)", self.thresh),
            ("Min. area", self.minarea),
            ("Grow (px)", self.grow),
            ("Cells per side", self.cells),
            ("Mesh (BACK_SIZE)", self.back_size),
        ):
            more.addRow(label, box)
        self.sky_more.toggled.connect(self._fold_sky)
        self._fold_sky(False)
        form.addRow(self.sky_more)
        self.sky_card = QtWidgets.QLabel()
        self.sky_card.setWordWrap(True)
        self.sky_card.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.sky_card.setTextFormat(Qt.RichText)
        form.addRow(self.sky_card)
        self.sky_map = SkyMap()
        self.sky_map.set_result(None)
        form.addRow(self.sky_map)
        self.use_sky = QtWidgets.QPushButton("Use this sky")
        self.use_sky.setEnabled(False)
        self.use_sky.setToolTip("Only a measurement with usable support, still up to date")
        self.use_sky.clicked.connect(self._use_measured)
        enter = QtWidgets.QPushButton("Enter a value…")
        enter.setToolTip("Type a sky level, with a note on where it comes from")
        enter.clicked.connect(self._enter_sky)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.use_sky)
        row.addWidget(enter)
        form.addRow(row)
        self.sky = number(-1e12, 1e12, 0, decimals=4, tip="the sky, frozen for the whole pair fit")
        self.sky.setReadOnly(True)
        self.sky_source = note("no sky yet")
        form.addRow("Sky", self.sky)
        form.addRow(self.sky_source)
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
        self.cycles = whole(1, 50, 4, tip="CYCLES: A then B, this many times after the start")
        self.niter = whole(1, 1000, 10, tip="NITER: iterations inside each elliprof call")
        form.addRow("Cycles (A → B)", tip(self.cycles, "pair_cycles"))
        form.addRow("NITER (each call)", tip(self.niter, "niter"))
        self.calls = note()
        form.addRow(self.calls)
        self.cycles.valueChanged.connect(self._summary_line)
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
        self.early = QtWidgets.QCheckBox("Stop early once stable")
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
            ("", self.early),
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
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.RichText)
        form.addRow(self.status)
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
        self._summary_line()

    # ---- folding
    def _fold_sky(self, on: bool) -> None:
        for w in self.sky_more.findChildren(QtWidgets.QWidget):
            w.setVisible(on)

    def _fold_advanced(self, on: bool) -> None:
        for w in self.advanced.findChildren(QtWidgets.QWidget):
            w.setVisible(on)

    def _summary_line(self) -> None:
        n = 2 + 2 * self.cycles.value()
        self.calls.setText(f"{n} elliprof calls: 2 to start, then A and B in each cycle")

    # ---- the sky
    def sky_command(self, n: int) -> str:
        parts = [f"sky {n} SEGMENT"]
        defaults = {"THRESH": 0.5, "MINAREA": 12, "GROW": 2.0, "CELLS": 8, "BACK_SIZE": 0}
        values = {
            "THRESH": self.thresh.value(),
            "MINAREA": self.minarea.value(),
            "GROW": self.grow.value(),
            "CELLS": self.cells.value(),
            "BACK_SIZE": self.back_size.value(),
        }
        parts += [f"{k}={v:g}" for k, v in values.items() if v != defaults[k]]
        layers = self.ticked_layers()
        if layers:
            parts.append("MASK=" + ",".join(layers))
        return " ".join(parts)

    def show_sky(self, result, stale: Optional[str]) -> None:
        """Show a sky measurement (None: none) and whether it still fits the image."""
        self._measurement = result
        self.sky_map.set_result(result)
        if result is None:
            self.sky_card.setText("")
            self.use_sky.setEnabled(False)
            return
        colour = STATE_COLOURS.get(result.state, "#8b919c")
        lines = [f"<b style='color:{colour}'>{result.state}</b> · measurement {result.id}"]
        if stale:
            lines.append(f"<span style='color:#f85149'>stale: {stale}</span>")
        if result.state != "no_estimate":
            lines.append(
                f"level {result.level:.6g} (median), clipped mean {result.mean:.6g}, "
                f"σ {result.sigma:.4g}"
            )
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
            "<i>Usable support is needed, not proof: a halo too faint to detect can remain.</i>"
        )
        self.sky_card.setText("<br>".join(lines))
        self.use_sky.setEnabled(result.state == "usable_support" and not stale)

    def _use_measured(self) -> None:
        r = self._measurement
        if r is None:
            return
        self.sky.setValue(r.level)
        self.sky_id = r.id
        self.sky_source.setText(f"measured: {r.id} ({r.state})")

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
        self.sky_id = ""
        self.sky_note = why.strip()
        self.sky_source.setText("typed" + (f": {why.strip()}" if why.strip() else ""))

    def set_sky_value(self, value: float, note_text: str) -> None:
        self.sky.setValue(value)
        self.sky_id = ""
        self.sky_source.setText(note_text)

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
        layers = self.ticked_layers()
        if layers:
            parts.append("MASK=" + ",".join(layers))
        if self.cycles.value() != 4:
            parts.append(f"CYCLES={self.cycles.value()}")
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
        if self.early.isChecked():
            parts.append("STOP")
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

    def set_running(self, on: bool) -> None:
        self.run.setEnabled(not on)
        self.stop.setEnabled(on)
        if on:
            self.status.setText("")

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
        self.status.setText(text)


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
        lay.addWidget(self.tabs, 1)
        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(170)
        lay.addWidget(self.summary)
        self.quantity.currentTextChanged.connect(lambda _t: self.draw())
        self.quarter.toggled.connect(lambda _t: self.draw())

    def set_record(self, record, summary: str = "") -> None:
        self.record = record
        self.summary.setPlainText(summary)
        self.draw()
        self._cycles()

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
