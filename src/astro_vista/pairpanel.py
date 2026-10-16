"""The Pair fit tool (two overlapping galaxies, fitted in turn) and its results.

The panel only builds command lines (``sky N SEGMENT ...``, ``pair N ...``); the window
runs them, as it runs every other tool's.  Galaxy A is drawn in magenta and B in orange,
everywhere.
"""

from __future__ import annotations

import os
import re
import shlex
from typing import List

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import guide
from .fitting import MODES
from .flow import FlowLayout
from .pair import (
    JOE_CORE,
    JOE_CORE_MAX,
    JOE_CORE_MIN,
    JOE_GROW,
    MAX_PARALLEL,
    PULL_FRAC,
    PULL_PX,
    PULL_R0,
    PULL_RUN,
    SHAPE_TOL,
)
from .skysection import SkyHistogram, SkySection
from .panels import (
    InputChoice,
    heading,
    note,
    number,
    plain_table,
    residual_keywords,
    residual_sources_row,
    tip,
    title_row,
    whole,
)

COLOURS = {"A": "#ff4dff", "B": "#ff9f1c"}
COS3X_SHOWN = [
    (2, "2: each isophote (default)"),
    (1, "1: median of all"),
    (0, "0: none"),
    (-2, "-2: 6th order, each"),
    (-1, "-1: 6th order, median"),
    (-3, "-3: 6th order, none"),
]
COS4X_SHOWN = [(2, "2: each isophote (default)"), (1, "1: median of all"), (0, "0: none")]
DEFAULT_PARALLEL = max(1, min(5, (os.cpu_count() or 1) - 1))
MODES_SHOWN = ["Automatic", "Automatic, adjustable", "Manual", "Joe's mode"]
AUTOMATIC, ADJUSTABLE, MANUAL, JOE = range(4)
# Joe's mode (METHOD=JOE): Joe Jensen's way of fitting a pair, with these settings
JOE_RUN = ["METHOD=JOE", "CYCLES=12", "CONVERGE=0.02", "TIE=-3", "RMSTAR", "TAIL=measured"]
# the run of the automatic settings (chosen on simulated pairs; see the changelog)
AUTO_RUN = [
    "CORES=0",
    "CYCLES=6",
    "CONVERGE=0.02",
    "CONSECUTIVE=2",
    "DAMP=0.5",
    "TRYDAMP=1,0.5",
    "TRYFIX=1,0",
    "TRYFACTOR=1.2,1.35",
    "TIE=-3",
    "RMSTAR",
    "TAIL=measured",
    "SKYPLANE=1",
    "SKYITER=3",
]
STATUS_COLOURS = {
    "stable_valid_pair": "#3fb950",
    "valid_pair_not_stable": "#d29922",
    "retained_after_failed_update": "#d29922",
    "incomplete_initialization": "#f85149",
    "cancelled": "#8b919c",
}


def _box(layout) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w


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
        self.form = form
        self._manual_rows = []  # rows hidden with the automatic settings
        self._row_of = {}  # the rows Joe's mode shows: key -> row

        def manual(*args, key=None):
            form.addRow(*args)
            self._manual_rows.append(form.rowCount() - 1)
            if key:
                self._row_of[key] = form.rowCount() - 1

        self.automatic = False
        self.joe = False
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
            if key == "start":
                self.start_pick = b
                self.start_auto = QtWidgets.QCheckBox("auto")
                self.start_auto.setChecked(True)
                self.start_auto.setToolTip(
                    "START=AUTO: a third of R1, at most 0.35 of the separation, at least 9 px "
                    "(worked out when the fit starts)"
                )
                self.start_auto.toggled.connect(self._start_auto)
                row.addWidget(self.start_auto)
            if key == "r1":
                self.r1_pick = b
                self.r1_auto = QtWidgets.QCheckBox("auto")
                self.r1_auto.setToolTip(
                    "R1=AUTO: where the light on the side away from the companion fades into "
                    "the noise (worked out when the fit starts)"
                )
                self.r1_auto.toggled.connect(self._r1_auto)
                row.addWidget(self.r1_auto)
                form.addRow(label, tip(row, "pair_r1auto"))
            else:
                manual(label, tip(row, "pair_grow" if key == "start" else key), key=key)
        self.start.setEnabled(False)
        self.start_pick.setEnabled(False)
        self.growth = QtWidgets.QComboBox()
        self.growth.addItems(["× factor", "+ pixels", "list of radii"])
        self.growth_value = number(0.01, 1e4, 1.2, decimals=2, step=0.05, tip="the factor or step")
        self.radii = QtWidgets.QLineEdit()
        self.radii.setPlaceholderText("40, 60, 90, 130, 180")
        self.radii.setToolTip("RADII: the R1 of each fit, increasing; the final R1 ends it")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.growth)
        row.addWidget(self.growth_value, 1)
        row.addWidget(self.radii, 2)
        manual("Growth", tip(row, "pair_grow"))
        self.schedule_note = note()
        manual(self.schedule_note)
        self.growth.currentIndexChanged.connect(self._growth_mode)
        self._growth_mode(0)
        for w in (self.start, self.r1, self.r0, self.growth_value):
            w.valueChanged.connect(lambda _v: self._show_schedule())
        self.radii.textChanged.connect(lambda _t: self._show_schedule())
        self.nr = whole(2, 100, 25, tip="NR: the number of isophotes from R0 to R1")
        self.coarse = note()
        self.nr_auto = QtWidgets.QCheckBox("auto")
        self.nr_auto.setChecked(True)
        self.nr_auto.setToolTip(
            "NR=AUTO: isophotes about 30 % apart in radius, for every R1 of the growth (6 to "
            "40); closer isophotes take up part of the SBF signal on simulated pairs"
        )
        self.nr_auto.toggled.connect(lambda on: self.nr.setEnabled(not on))
        self.nr.setEnabled(False)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.nr)
        row.addWidget(self.nr_auto)
        row.addWidget(self.coarse, 1)
        manual("NR", tip(row, "nr"), key="nr")
        self.cos3x = QtWidgets.QComboBox()
        for value, text in COS3X_SHOWN:
            self.cos3x.addItem(text, value)
        self.cos4x = QtWidgets.QComboBox()
        for value, text in COS4X_SHOWN:
            self.cos4x.addItem(text, value)
        self.cos3x.setCurrentIndex(self.cos3x.findData(2))
        self.cos4x.setCurrentIndex(self.cos4x.findData(2))
        self.cos3x.setToolTip(
            "COS3X: the cos 3θ term (lopsided, egg-shaped isophotes) in the model: 2 each "
            "isophote's own, 1 the median over all isophotes, 0 none; negative values fit the "
            "6th order instead (-2 each, -1 median, -3 none)"
        )
        self.cos4x.setToolTip(
            "COS4X: the cos 4θ term in the model (disky isophotes: positive; boxy: negative): "
            "2 each isophote's own, 1 the median over all isophotes, 0 none"
        )
        self.try_cos3x = QtWidgets.QLineEdit()
        self.try_cos4x = QtWidgets.QLineEdit()
        for box, key in ((self.try_cos3x, "COS3X"), (self.try_cos4x, "COS4X")):
            box.setPlaceholderText("try: e.g. 0, 1, 2")
            box.setToolTip(
                f"TRY{key}: values to try as well, on the fit kept; another is kept when it "
                "lowers the residual by more than 0.01 of the noise"
            )
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("COS3X"), 0, 0)
        grid.addWidget(self.cos3x, 0, 1)
        grid.addWidget(self.try_cos3x, 0, 2)
        grid.addWidget(QtWidgets.QLabel("COS4X"), 1, 0)
        grid.addWidget(self.cos4x, 1, 1)
        grid.addWidget(self.try_cos4x, 1, 2)
        grid.setColumnStretch(2, 1)
        manual("Harmonics", tip(grid, "pair_mode"), key="harmonics")
        self.fixctr = QtWidgets.QComboBox()
        self.fixctr.addItems(["Free", "Fixed at X0, Y0", "Median of the fitted centres"])
        self.fixctr.setCurrentIndex(1)  # fixed centres: the more stable decomposition
        manual("Centres", tip(self.fixctr, "pair_fixctr"))
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
        manual(f"Core of {cid}", tip(row, "pair_core"))
        self.core_note = note(f"left out of the fit of {other}")
        manual(self.core_note)
        self.adaptive = QtWidgets.QCheckBox("Adaptive outer radius")
        self.adaptive.setToolTip(
            "If the outer isophotes fail their checks, try a smaller R1 (at most 3 times, "
            "never below half the R1 asked for); every try is logged"
        )
        self.visible = QtWidgets.QCheckBox("Show on the image")
        self.visible.setChecked(True)
        manual(tip(self.adaptive, "pair_adaptive"))
        form.addRow(self.visible)
        for box in (self.x0, self.y0, self.r0, self.r1, self.start):
            box.valueChanged.connect(lambda _v: self.changed.emit())
        self.nr.valueChanged.connect(self._nr)
        self.visible.toggled.connect(lambda _v: self.changed.emit())
        self.core.valueChanged.connect(lambda _v: self.changed.emit())
        self._nr(self.nr.value())

    def _start_auto(self, on: bool) -> None:
        self.start.setEnabled(not on)
        self.start_pick.setEnabled(not on)
        self._show_schedule()

    def _r1_auto(self, on: bool) -> None:
        self.r1.setEnabled(not on)
        self.r1_pick.setEnabled(not on)
        self.changed.emit()

    def set_automatic(self, on: bool, joe: bool = False) -> None:
        """Show only the centre, R1 (automatic or typed) and the name (automatic settings),
        or every setting.  Joe's mode (``joe``) also shows R0, the first R1, NR and the
        harmonics (not the values to try: it has no search)."""
        self.automatic, self.joe = on, joe
        for row in self._manual_rows:
            self.form.setRowVisible(row, not on)
        if joe:
            for key in ("r0", "start", "nr", "harmonics"):
                self.form.setRowVisible(self._row_of[key], True)
        for box in (self.try_cos3x, self.try_cos4x):
            box.setVisible(not joe)

    def joe_keywords(self, fixctr: int = 1) -> str:
        """The keywords of Joe's mode: the centre, R0, R1 (or AUTO), the first R1 (or AUTO),
        NR (or AUTO), the harmonics and the centres of the model."""
        parts = []
        name = self.name.text().strip()
        if name:
            parts.append("NAME=" + shlex.quote(name))
        parts += [f"X0={self.x0.value():g}", f"Y0={self.y0.value():g}", f"R0={self.r0.value():g}"]
        parts.append("R1=AUTO" if self.r1_auto.isChecked() else f"R1={self.r1.value():g}")
        if self.start_auto.isChecked() or not self.start.value():
            parts.append("START=AUTO")
        else:
            parts.append(f"START={self.start.value():g}")
        parts.append("NR=AUTO" if self.nr_auto.isChecked() else f"NR={self.nr.value()}")
        for key, box in (("COS3X", self.cos3x), ("COS4X", self.cos4x)):
            if box.currentData() != 2:
                parts.append(f"{key}={box.currentData()}")
        parts.append(f"FIXCTR={fixctr}")
        return " ".join(parts)

    def auto_keywords(self) -> str:
        """The keywords of the automatic settings: the centre, R1 (or R1=AUTO), and START,
        NR and the centre fitting chosen on simulated pairs."""
        parts = []
        name = self.name.text().strip()
        if name:
            parts.append("NAME=" + shlex.quote(name))
        parts += [f"X0={self.x0.value():g}", f"Y0={self.y0.value():g}", "R0=3"]
        parts.append("R1=AUTO" if self.r1_auto.isChecked() else f"R1={self.r1.value():g}")
        parts += ["START=AUTO", "NR=AUTO", "FIXCTR=1"]
        return " ".join(parts)

    def _growth_mode(self, index: int) -> None:
        self.growth_value.setVisible(index < 2)
        self.radii.setVisible(index == 2)
        if index == 0 and self.growth_value.value() < 1.01:
            self.growth_value.setValue(1.2)
        if index == 1 and self.growth_value.value() < 2:
            self.growth_value.setValue(20)
        self._show_schedule()

    def growth_keywords(self) -> List[str]:
        index = self.growth.currentIndex()
        if index == 2:
            text = ",".join(x for x in self.radii.text().replace(" ", ",").split(",") if x)
            return [f"RADII={text}"] if text else []
        if self.start_auto.isChecked():
            out = ["START=AUTO"]
        elif not self.r0.value() < self.start.value() < self.r1.value():
            return []
        else:
            out = [f"START={self.start.value():g}"]
        out.append(
            f"FACTOR={self.growth_value.value():g}"
            if index == 0
            else f"STEP={self.growth_value.value():g}"
        )
        return out

    def schedule(self) -> List[float]:
        """The R1 of each fit, as the engine will make it (empty if not valid)."""
        from .pair import PairError, components_from

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
        if self.start_auto.isChecked() and self.growth.currentIndex() < 2:
            self.schedule_note.setText("initial R1: chosen when the fit starts (START=AUTO)")
            self.changed.emit()
            return
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
        if self.automatic:
            return self.auto_keywords()
        parts = []
        name = self.name.text().strip()
        if name:
            parts.append("NAME=" + shlex.quote(name))
        parts += [f"X0={self.x0.value():g}", f"Y0={self.y0.value():g}"]
        parts += [f"R0={self.r0.value():g}"]
        parts.append("R1=AUTO" if self.r1_auto.isChecked() else f"R1={self.r1.value():g}")
        if self.nr_auto.isChecked():
            parts.append("NR=AUTO")
        elif self.nr.value() != 25:
            parts.append(f"NR={self.nr.value()}")
        parts += self.harmonic_keywords()
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

    def harmonic_keywords(self) -> List[str]:
        """COS3X=, COS4X= (when not ELLIPROF's default) and the values to try."""
        out = []
        for key, box in (("COS3X", self.cos3x), ("COS4X", self.cos4x)):
            if box.currentData() != 2:
                out.append(f"{key}={box.currentData()}")
        for key, box in (("TRYCOS3X", self.try_cos3x), ("TRYCOS4X", self.try_cos4x)):
            values = [v for v in re.split(r"[,\s]+", box.text().strip()) if v]
            if values:
                out.append(f"{key}=" + ",".join(values))
        return out

    def values(self) -> dict:
        return {
            "name": self.name.text(),
            "x0": self.x0.value(),
            "y0": self.y0.value(),
            "r0": self.r0.value(),
            "r1": self.r1.value(),
            "nr": self.nr.value(),
            "cos3x": self.cos3x.currentData(),
            "cos4x": self.cos4x.currentData(),
            "try_cos3x": self.try_cos3x.text(),
            "try_cos4x": self.try_cos4x.text(),
            "fixctr": self.fixctr.currentIndex(),
            "core_mode": self.core_mode.currentIndex(),
            "core": self.core.value(),
            "adaptive": self.adaptive.isChecked(),
            "start": self.start.value(),
            "growth": self.growth.currentIndex(),
            "growth_value": self.growth_value.value(),
            "radii": self.radii.text(),
            "r1_auto": self.r1_auto.isChecked(),
            "nr_auto": self.nr_auto.isChecked(),
            "start_auto": self.start_auto.isChecked(),
        }

    def set_values(self, v: dict) -> None:
        self.name.setText(v.get("name", ""))
        for key in ("x0", "y0", "r0", "r1", "core", "start"):
            if key in v:
                getattr(self, key).setValue(float(v[key]))
        if "nr" in v:
            self.nr.setValue(int(v["nr"]))
        if "mode" in v:  # a named mode (older settings): its COS3X and COS4X
            name = v["mode"] if isinstance(v["mode"], str) else ([""] + list(MODES))[v["mode"]]
            v = dict(v, **dict(zip(("cos3x", "cos4x"), MODES.get(name, (2, 2)))))
        for key in ("cos3x", "cos4x"):
            if key in v:
                box = getattr(self, key)
                box.setCurrentIndex(max(box.findData(int(v[key])), 0))
        for key in ("try_cos3x", "try_cos4x"):
            if key in v:
                getattr(self, key).setText(str(v[key]))
        for key in ("fixctr", "core_mode"):
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
        if "r1_auto" in v:
            self.r1_auto.setChecked(bool(v["r1_auto"]))
        if "nr_auto" in v:
            self.nr_auto.setChecked(bool(v["nr_auto"]))
        if "start_auto" in v:
            self.start_auto.setChecked(bool(v["start_auto"]))


class PairPanel(SkySection, QtWidgets.QWidget):
    """The settings of a pair fit, and the sky it uses."""

    sky_requested = QtCore.Signal()
    sky_mask_toggled = QtCore.Signal(bool)  # show the sky-exclusion layer on the image
    pick_requested = QtCore.Signal(str, str)  # "A" or "B"; "centre", "r0" or "r1"
    run_requested = QtCore.Signal()
    stop_requested = QtCore.Signal()
    cancel_requested = QtCore.Signal()  # stop, and keep nothing of the fit
    export_requested = QtCore.Signal()
    blink_requested = QtCore.Signal()
    report_requested = QtCore.Signal()
    iteration_requested = QtCore.Signal(int)  # a call's number (1, 2, ...): show its images
    objects_requested = QtCore.Signal(float)  # mask every detected object but A and B (grow)
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
        self._manual_rows = []

        def manual(*args):
            form.addRow(*args)
            self._manual_rows.append(form.rowCount() - 1)

        self.form = form
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(MODES_SHOWN)
        self.mode.setToolTip(
            "Automatic: only the centres, the masks and the sky to choose; the radii, the "
            "isophotes and the recipe are those that worked best on simulated pairs.\n"
            "Automatic, adjustable: the same values, every one shown and open to change; the "
            "radii and isophotes stay automatic until you untick their auto boxes.\n"
            "Manual: every setting yours, the radii typed or picked.\n"
            "Joe's mode: Joe Jensen's way. Each galaxy's centre left out while the other is "
            "fitted; the smaller first, not far out; then the two in turn, each grown only as "
            "far as its outer ellipses are not pulled towards the other. Fast, no search."
        )
        self.input = InputChoice("pair fit")
        form.addRow("Image", tip(self.input, "tool_input"))
        form.addRow("Fit", tip(self.mode, "pair_auto"))
        # ---- the sky
        self.build_sky(
            form, manual, self.sky_requested, self.sky_mask_toggled, "1. Sky", "Fit pair"
        )
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
        row = QtWidgets.QHBoxLayout()
        self.mask_objects = QtWidgets.QPushButton("Mask the other objects")
        self.mask_objects.setToolTip(
            "Make the mask layer 'objects': every source the Source finder detects (its "
            "segmentation pixels), except the sources at A's and B's centres"
        )
        self.mask_objects.clicked.connect(
            lambda: self.objects_requested.emit(self.objects_grow.value())
        )
        row.addWidget(self.mask_objects)
        row.addWidget(QtWidgets.QLabel("grow"))
        self.objects_grow = number(
            0, 50, 2, decimals=1, tip="GROW: pixels added around each object"
        )
        row.addWidget(self.objects_grow)
        row.addStretch(1)
        form.addRow(tip(_box(row), "pair_objects"))

        # ---- the fit
        form.addRow(heading("4. Fitting"))
        self.parallel = whole(1, MAX_PARALLEL, DEFAULT_PARALLEL)
        self.parallel.setToolTip(
            "PARALLEL: N fits, all averaged: the fit kept and N - 1 repeats with each galaxy's "
            "R0 and R1 changed at random, run N at a time (each ELLIPROF call is a process of "
            "its own). Each refines its own sky. Buffers get the mean A, B and A + B models, "
            "the mean residual (image - mean sky - mean A + B), the models' spread and every "
            "fit's own model and residual. 1: one fit, no mean"
        )
        self.jitter = whole(1, 50, 15)
        self.jitter.setSuffix(" %")
        self.jitter.setToolTip(
            "JITTER: each repeat changes R0 and R1 by two thirds of this to this, up or down "
            "(15 %: 10 to 15 %)"
        )
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.parallel)
        row.addWidget(QtWidgets.QLabel("R0, R1 varied by"))
        row.addWidget(self.jitter)
        row.addStretch(1)
        form.addRow("Fits to average", tip(row, "pair_parallel"))
        self.parallel_note = note(f"this computer has {os.cpu_count() or 1} cores; 1: a single fit")
        form.addRow(self.parallel_note)
        self.all_harmonics = QtWidgets.QCheckBox("Search every harmonic (COS3X, COS4X)")
        self.all_harmonics.setToolTip(
            "HARMONICS=AUTO: on the fit kept, try every COS3X and COS4X for A, then for B, and "
            "keep the stable fit with the smallest residual"
        )
        self.wings = QtWidgets.QCheckBox("Mask star wings and spikes the first pass shows")
        self.wings.setToolTip(
            "WINGS=1: light beyond the masks of bright sources (a star's wings, its diffraction "
            "spikes) found in the residual of a first pass is added to the mask"
        )
        form.addRow(tip(self.all_harmonics, "pair_mode"))
        form.addRow(tip(self.wings, "pair_wings"))
        self.pull_check = QtWidgets.QCheckBox("Check the pull of the outer ellipses after the fit")
        self.pull_check.setChecked(True)
        self.pull_check.setToolTip(
            "Each galaxy fitted once more with free centres; the report says whether its outer "
            "ellipses move towards the other galaxy and from which radius (PULLCHECK=0: off)"
        )
        form.addRow(tip(self.pull_check, "pair_pull"))
        self.resid_rounds, self.resid_thresh, row = residual_sources_row()
        form.addRow("Residual sources", tip(row, "resid_sources"))

        self.joe_note = note(
            "Joe's mode: each galaxy's centre left out while the other is fitted; the smaller "
            "galaxy first, not far out; then the two in turn, each trying a 20 % larger R1 and "
            "keeping it only as far as its ellipses (fitted with free centres) do not move "
            "towards the other; the model kept with fixed centres, smoothed over 3 isophotes, "
            "the light beyond R1 measured in rings. No search and no sky refinement: about a "
            "tenth of the calls of Automatic."
        )
        self.auto_note = note(
            "Automatic: R1 where each galaxy's light fades into the noise; the first fits small "
            "and growing; isophotes about 30 % apart, smoothed over 3 (TIE=-3); bright outliers "
            "rejected along each isophote (RMSTAR); beyond R1, the light measured in rings; the sky "
            "factor refined 3 times from the residual sky; 8 variants of damping, centres and growth "
            "fitted, the best kept by the residual. Choose Automatic, adjustable to see and "
            "change any of it."
        )
        form.addRow(self.auto_note)
        form.addRow(self.joe_note)
        self.joe_box = self._joe_settings()
        form.addRow(self.joe_box)
        manual(
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
        manual("Max cycles at final R1", tip(self.cycles, "pair_cycles"))
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
        manual("Converged", tip(row, "pair_converge"))
        self.on_fail = QtWidgets.QComboBox()
        self.on_fail.addItems(
            [
                "keep the last valid model, try smaller steps up",
                "keep the last valid model, stay at its radius",
                "keep the last valid model, try the radius again",
                "stop and keep the last valid models",
            ]
        )
        manual("If a fit fails", tip(self.on_fail, "pair_onfail"))
        self.damp = number(0.05, 1.0, 0.5, decimals=2, step=0.1, tip="DAMP: share of the new model")
        self.damp.setSpecialValueText("off (1)")
        manual("Damping", tip(self.damp, "pair_damp"))
        self.tie = whole(-50, 10, -3, tip="TIE")
        self.tie.setToolTip(
            "TIE: smooth the ellipticity, angle and centre with radius. -3: a running average "
            "over 3 isophotes (chosen on simulated SBF pairs); -1: none; k >= 0: a polynomial "
            "of order k in r^1/4"
        )
        manual("Smoothing with radius", tip(self.tie, "pair_tie"))
        self.rmstar = QtWidgets.QCheckBox("Reject bright outliers along each isophote (RMSTAR)")
        self.rmstar.setChecked(True)
        self.rmstar.setToolTip(
            "Samples brighter than the median + 4 x (upper quartile - median) are ignored: "
            "globular clusters and faint stars that no mask covers"
        )
        manual(tip(self.rmstar, "pair_rmstar"))
        self.tail = QtWidgets.QComboBox()
        for label, key in (
            ("measured on the image (rings of the last isophote's shape)", "measured"),
            ("Sérsic law fitted to the outer profile", "sersic"),
            ("elliprof's extrapolation (r^¼ from the last two)", "elliprof"),
            ("exponential from the outer 30 %", "exp"),
            ("cut off (a taper over 0.1 R1)", "cut"),
        ):
            self.tail.addItem(label, key)
        self.tail.setToolTip(
            "TAIL: the model beyond the last isophote, which is subtracted from the "
            "companion's image"
        )
        manual("Model beyond R1", tip(self.tail, "pair_tail"))
        self.sky_plane = QtWidgets.QCheckBox("Remove a sky gradient (SKYPLANE)")
        self.sky_plane.setChecked(True)
        self.sky_plane.setToolTip(
            "Fit a plane to the residual of a first pass and, when its gradient is "
            "significant, fit the image minus its tilt (the sky level stays yours)"
        )
        manual(tip(self.sky_plane, "pair_skyplane"))
        self.polish = whole(0, 20, 2, tip="POLISH: undamped cycles after the damped ones")
        manual("Polish cycles", tip(self.polish, "pair_damp"))
        self.search = QtWidgets.QGroupBox("Search for the best fit")
        self.search.setCheckable(True)
        self.search.setChecked(True)
        grid = QtWidgets.QGridLayout(self.search)
        self.try_damp = QtWidgets.QLineEdit("1, 0.5")
        self.try_fix = QtWidgets.QComboBox()
        self.try_fix.addItems(["fixed and free", "fixed only", "free only"])
        self.try_factor = QtWidgets.QLineEdit("1.2, 1.35")
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
        manual(tip(self.search, "pair_search"))
        manual("NITER (each call)", tip(self.niter, "niter"))
        self.masking = QtWidgets.QCheckBox("Also mask a core around the companion (older method)")
        self.masking.setToolTip("CORES: by default nothing is masked around the companion")
        manual(self.masking)
        self.calls = note()
        manual(self.calls)
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
        self.corecap = number(0, 1e4, 50, decimals=1, tip="the largest core, pixels")
        self.timeout = number(1, 36000, 300, decimals=0, tip="seconds per elliprof call")
        self.bootstrap = QtWidgets.QCheckBox("Partial bootstrap start if a first fit fails")
        self.bootstrap.setChecked(True)
        self.views = QtWidgets.QCheckBox("Also image − sky − B and image − sky − A")
        self.views.setChecked(True)
        for label, w in (
            ("RLAW", self.rlaw),
            ("GAIN", self.gain),
            ("Core cap (px)", self.corecap),
            ("Timeout (s)", self.timeout),
            ("", self.bootstrap),
            ("", self.views),
        ):
            adv.addRow(label, w)
        self.advanced.toggled.connect(self._fold_advanced)
        self._fold_advanced(False)
        manual(self.advanced)
        # ---- the run
        form.addRow(heading("5. Run"))
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
        self.cancel = QtWidgets.QPushButton("Cancel")
        self.cancel.setEnabled(False)
        self.cancel.setToolTip(
            "Stop the fit as soon as the elliprof calls running now end, and keep nothing of "
            "it: no models, residuals, means or views; the buffers stay as they were"
        )
        self.cancel.clicked.connect(self.cancel_requested)
        buttons.addWidget(self.run, 2)
        buttons.addWidget(self.stop, 1)
        buttons.addWidget(self.cancel, 1)
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
        self.build_residual_sky(
            form,
            "6. Residual sky",
            "Fit pair",
            "image − sky − A − B on the sky pixels: its median in red, zero dashed. The "
            "models extend beyond R1, so their outer light counts.",
        )
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
        self.mode.currentIndexChanged.connect(self.set_mode)
        self.set_mode(AUTOMATIC)

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

    # ---- Joe's mode
    def _joe_settings(self) -> QtWidgets.QGroupBox:
        """Joe's mode's own settings (shown in that mode only), at its defaults."""
        box = QtWidgets.QGroupBox("Joe's mode")
        f = QtWidgets.QFormLayout(box)
        f.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)  # labels above in a narrow dock
        self.joe_first = QtWidgets.QComboBox()
        for label, key in (("the smaller (smaller R1)", "smaller"), ("A", "a"), ("B", "b")):
            self.joe_first.addItem(label, key)
        self.joe_first.setToolTip(
            "FIRST: the galaxy fitted first, not far out; if it cannot be started, the other "
            "starts instead"
        )
        f.addRow("Fit first", tip(self.joe_first, "pair_joe"))
        self.joe_grow = number(1.02, 3.0, JOE_GROW, decimals=2, step=0.05)
        self.joe_grow.setPrefix("× ")
        self.joe_grow.setToolTip("JOEGROW: each cycle, each galaxy tries its last R1 times this")
        f.addRow("R1 tried each cycle", self.joe_grow)
        self.joe_core = number(0.0, 1.0, JOE_CORE, decimals=2, step=0.05)
        self.joe_core_min = number(0.0, 200.0, JOE_CORE_MIN, decimals=1, step=1)
        self.joe_core_max = number(0.0, 500.0, JOE_CORE_MAX, decimals=1, step=5)
        for b, key in (
            (self.joe_core, "JOECORE: a share of the distance between the centres (0: none)"),
            (self.joe_core_min, "JOECOREMIN: at least this many pixels"),
            (self.joe_core_max, "JOECOREMAX: at most this many pixels"),
        ):
            b.setToolTip(key)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.joe_core)
        row.addWidget(QtWidgets.QLabel("of the separation"))
        row.addStretch(1)
        f.addRow("Centre left out", row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.joe_core_min)
        row.addWidget(QtWidgets.QLabel("to"))
        row.addWidget(self.joe_core_max)
        row.addWidget(QtWidgets.QLabel("px"))
        row.addStretch(1)
        f.addRow("… within", row)
        self.pull_px = number(0.05, 50.0, PULL_PX, decimals=2, step=0.1)
        self.pull_frac = number(0.0, 50.0, 100 * PULL_FRAC, decimals=1, step=0.5)
        self.pull_frac.setSuffix(" %")
        self.pull_run = whole(1, 10, PULL_RUN)
        for b, key in (
            (self.pull_px, "PULLPX: a move towards the other galaxy of more than this (px)"),
            (self.pull_frac, "PULLFRAC: ... or more than this share of the isophote's radius"),
            (self.pull_run, "PULLRUN: ... for this many isophotes in a row"),
        ):
            b.setToolTip(key)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.pull_px)
        row.addWidget(QtWidgets.QLabel("px or"))
        row.addWidget(self.pull_frac)
        row.addWidget(QtWidgets.QLabel("of radius"))
        row.addStretch(1)
        f.addRow("Pulled if moved", tip(row, "pair_pull"))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.pull_run)
        row.addWidget(QtWidgets.QLabel("isophotes in a row"))
        row.addStretch(1)
        f.addRow("… for", row)
        self.pull_r0 = number(0.0, 200.0, PULL_R0, decimals=1, step=1)
        self.pull_r0.setSuffix(" px")
        self.pull_r0.setToolTip(
            "PULLR0: the free-centre fit that judges the pull starts at this R0 at least (the "
            "centre of a small, undersampled galaxy cannot be fitted freely); 0: its own R0"
        )
        f.addRow("Free-centre fit from", self.pull_r0)
        self.joe_fixctr = QtWidgets.QComboBox()
        for label, v in (("fixed at X0, Y0", 1), ("free", 0), ("median of the fitted centres", 2)):
            self.joe_fixctr.addItem(label, v)
        self.joe_fixctr.setToolTip(
            "FIXCTR of the model kept (the pull is always judged with free centres); Joe's own "
            "fits use free centres"
        )
        f.addRow("Centres of the model", self.joe_fixctr)
        self.joe_cycles = whole(1, 50, 12, tip="CYCLES: at most this many A → B cycles")
        self.joe_converge = number(0.01, 50, 2.0, decimals=2, step=0.5)
        self.joe_converge.setSuffix(" %")
        self.joe_converge.setToolTip(
            "CONVERGE: it stops when neither galaxy grew for two cycles and A, B and A + B "
            "changed by less than this"
        )
        f.addRow("Cycles, at most", self.joe_cycles)
        f.addRow("Stable below", self.joe_converge)
        self.joe_tie = whole(-50, 10, -3, tip="TIE: -3 smooths over 3 isophotes; -1 none")
        self.joe_rmstar = QtWidgets.QCheckBox("RMSTAR")
        self.joe_rmstar.setChecked(True)
        self.joe_rmstar.setToolTip("RMSTAR: bright outliers along each isophote rejected")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("TIE"))
        row.addWidget(self.joe_tie)
        row.addWidget(self.joe_rmstar)
        row.addStretch(1)
        f.addRow("Isophotes", row)
        self.joe_tail = QtWidgets.QComboBox()
        for label, key in (
            ("measured in rings", "measured"),
            ("Sérsic law", "sersic"),
            ("elliprof's extrapolation", "elliprof"),
            ("exponential", "exp"),
            ("cut at R1", "cut"),
        ):
            self.joe_tail.addItem(label, key)
        self.joe_tail.setToolTip(
            "TAIL: the model beyond each galaxy's last isophote, subtracted from the other's "
            "image (elliprof's extrapolation is often much too bright)"
        )
        f.addRow("Beyond R1", self.joe_tail)
        self.shape_tol = number(0.0, 1.0, SHAPE_TOL, decimals=2, step=0.02)
        self.shape_tol.setSpecialValueText("off")
        self.shape_tol.setToolTip(
            "SHAPETOL: at the end, when one of a galaxy's 3 outermost ellipses is off the trend "
            "of those inside it by more than this (ellipticity and angle together), or its "
            "model beyond R1 is anomalous, coarser isophotes and a smaller R1 are tried, the "
            "largest clean one kept, and both galaxies fitted once more; 0: off"
        )
        f.addRow("Outer ellipses off by", tip(self.shape_tol, "pair_shape"))
        reset = QtWidgets.QPushButton("Joe's defaults")
        reset.clicked.connect(self.joe_defaults)
        f.addRow(reset)
        box.setVisible(False)
        return box

    def joe_defaults(self) -> None:
        self.joe_first.setCurrentIndex(0)
        self.joe_grow.setValue(JOE_GROW)
        self.joe_core.setValue(JOE_CORE)
        self.joe_core_min.setValue(JOE_CORE_MIN)
        self.joe_core_max.setValue(JOE_CORE_MAX)
        self.pull_px.setValue(PULL_PX)
        self.pull_frac.setValue(100 * PULL_FRAC)
        self.pull_run.setValue(PULL_RUN)
        self.pull_r0.setValue(PULL_R0)
        self.joe_fixctr.setCurrentIndex(0)
        self.joe_cycles.setValue(12)
        self.joe_converge.setValue(2.0)
        self.joe_tie.setValue(-3)
        self.joe_rmstar.setChecked(True)
        self.joe_tail.setCurrentIndex(0)
        self.shape_tol.setValue(SHAPE_TOL)

    def joe_keywords(self) -> List[str]:
        """Joe's mode's keywords: the method, its cycles and isophote settings, and those of
        its own settings that differ from the defaults."""
        out = [
            "METHOD=JOE",
            f"CYCLES={self.joe_cycles.value()}",
            f"CONVERGE={self.joe_converge.value() / 100:g}",
            f"TIE={self.joe_tie.value()}",
        ]
        if self.joe_rmstar.isChecked():
            out.append("RMSTAR")
        out.append(f"TAIL={self.joe_tail.currentData()}")
        if self.joe_first.currentData() != "smaller":
            out.append(f"FIRST={self.joe_first.currentData().upper()}")
        for key, box, default, scale in (
            ("JOEGROW", self.joe_grow, JOE_GROW, 1),
            ("JOECORE", self.joe_core, JOE_CORE, 1),
            ("JOECOREMIN", self.joe_core_min, JOE_CORE_MIN, 1),
            ("JOECOREMAX", self.joe_core_max, JOE_CORE_MAX, 1),
            ("PULLPX", self.pull_px, PULL_PX, 1),
            ("PULLFRAC", self.pull_frac, PULL_FRAC, 100),
            ("PULLR0", self.pull_r0, PULL_R0, 1),
            ("SHAPETOL", self.shape_tol, SHAPE_TOL, 1),
        ):
            v = box.value() / scale
            if abs(v - default) > 1e-9:
                out.append(f"{key}={v:g}")
        if self.pull_run.value() != PULL_RUN:
            out.append(f"PULLRUN={self.pull_run.value()}")
        return out

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
    def is_automatic(self) -> bool:
        """Automatic or Joe's mode: few settings shown, the rest fixed."""
        return self.mode.currentIndex() in (AUTOMATIC, JOE)

    def set_mode(self, index: int) -> None:
        """Automatic (few settings shown), automatic and adjustable (every setting, at the
        recommended values, radii automatic) or manual (every setting, radii typed)."""
        if self.mode.currentIndex() != index:
            self.mode.setCurrentIndex(index)  # comes back here
            return
        self.set_automatic(index in (AUTOMATIC, JOE))
        for card in self.cards.values():
            card.set_automatic(index in (AUTOMATIC, JOE), joe=index == JOE)
        self.auto_note.setVisible(index == AUTOMATIC)
        self.joe_note.setVisible(index == JOE)
        self.joe_box.setVisible(index == JOE)
        if index == ADJUSTABLE:
            self.recommended()
        elif index == MANUAL:
            for card in self.cards.values():
                for box in (card.r1_auto, card.start_auto, card.nr_auto):
                    box.setChecked(False)

    def recommended(self) -> None:
        """Every setting at the values of the automatic settings."""
        self.cycles.setValue(6)
        self.converge.setValue(2.0)
        self.consecutive.setValue(2)
        self.on_fail.setCurrentIndex(0)
        self.damp.setValue(0.5)
        self.polish.setValue(2)
        self.search.setChecked(True)
        self.try_damp.setText("1, 0.5")
        self.try_fix.setCurrentIndex(0)
        self.try_factor.setText("1.2, 1.35")
        self.tie.setValue(-3)
        self.rmstar.setChecked(True)
        self.tail.setCurrentIndex(self.tail.findData("measured"))
        self.sky_plane.setChecked(True)
        self.sky_iter.setValue(3)
        self.masking.setChecked(False)
        for card in self.cards.values():
            card.r0.setValue(3)
            card.fixctr.setCurrentIndex(1)
            card.growth.setCurrentIndex(0)
            card.growth_value.setValue(1.2)
            for box in (card.r1_auto, card.start_auto, card.nr_auto):
                box.setChecked(True)

    def set_automatic(self, on: bool) -> None:
        for row in self._manual_rows:
            self.form.setRowVisible(row, not on)
        self.auto_note.setVisible(on)
        for card in self.cards.values():
            card.set_automatic(on)
            if on:
                card.r1_auto.setChecked(True)
        self._summary_line()

    def run_keywords(self) -> List[str]:
        """PARALLEL (the fits averaged), JITTER, HARMONICS and WINGS: shown in every mode."""
        out = []
        if self.parallel.value() > 1:
            out.append(f"PARALLEL={self.parallel.value()}")
            if self.jitter.value() != 15:
                out.append(f"JITTER={self.jitter.value() / 100:g}")
        if self.all_harmonics.isChecked():
            out.append("HARMONICS=AUTO")
        if self.wings.isChecked():
            out.append("WINGS=1")
        return out

    def auto_command(self, n: int) -> str:
        """The pair command of the automatic settings."""
        parts = [f"pair {n}", f"SKY={self.sky.value():.10g}"]
        if self.factor.value() != 1.0:
            parts.append(f"SKYFACTOR={self.factor.value():.6g}")
        layers = self.ticked_layers()
        if layers:
            parts.append("MASK=" + ",".join(layers))
        # the sky iterations need a measured sky (its support); a typed sky goes as typed
        parts += [p for p in AUTO_RUN if self.sky_id or not p.startswith("SKYITER")]
        parts += self.run_keywords()
        if not self.pull_check.isChecked():
            parts.append("PULLCHECK=0")
        parts += residual_keywords(self.resid_rounds, self.resid_thresh)
        if self.views.isChecked():
            parts.append("VIEWS")
        parts.append("A: " + self.cards["A"].auto_keywords())
        parts.append("B: " + self.cards["B"].auto_keywords())
        return " ".join(parts)

    def joe_command(self, n: int) -> str:
        """The pair command of Joe's mode."""
        parts = [f"pair {n}", f"SKY={self.sky.value():.10g}"]
        if self.factor.value() != 1.0:
            parts.append(f"SKYFACTOR={self.factor.value():.6g}")
        layers = self.ticked_layers()
        if layers:
            parts.append("MASK=" + ",".join(layers))
        parts += self.joe_keywords()
        if not self.pull_check.isChecked():
            parts.append("PULLCHECK=0")
        parts += residual_keywords(self.resid_rounds, self.resid_thresh)
        if self.views.isChecked():
            parts.append("VIEWS")
        for cid in ("A", "B"):
            parts.append(f"{cid}: " + self.cards[cid].joe_keywords(self.joe_fixctr.currentData()))
        return " ".join(parts)

    def command(self, n: int) -> str:
        if self.mode.currentIndex() == JOE:
            return self.joe_command(n)
        if self.is_automatic():
            return self.auto_command(n)
        parts = [f"pair {n}", f"SKY={self.sky.value():.10g}"]
        if self.factor.value() != 1.0:
            parts.append(f"SKYFACTOR={self.factor.value():.6g}")
        if self.sky_iter.value() and self.sky_id:
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
        if self.tail.currentData() != "elliprof":
            parts.append(f"TAIL={self.tail.currentData()}")
        if self.sky_plane.isChecked():
            parts.append("SKYPLANE=1")
        if self.polish.value() != 2:
            parts.append(f"POLISH={self.polish.value()}")
        if self.corecap.value() != 50:
            parts.append(f"CORECAP={self.corecap.value():g}")
        if self.timeout.value() != 300:
            parts.append(f"TIMEOUT={self.timeout.value():g}")
        if not self.bootstrap.isChecked():
            parts.append("BOOTSTRAP=0")
        if self.views.isChecked():
            parts.append("VIEWS")
        parts += self.run_keywords()
        if not self.pull_check.isChecked():
            parts.append("PULLCHECK=0")
        parts += residual_keywords(self.resid_rounds, self.resid_thresh)
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
        cycle = str(info.get("cycle", ""))
        if info.get("variant_number"):  # parallel fits interleave: say whose call it is
            cycle = f"variant {info['variant_number']}: {cycle}"
        cells = [
            cycle,
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
        self.cancel.setEnabled(on)
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
                "; ".join(
                    x
                    for x in (
                        ("kept: " + self.record.result.search_choice) if r.get("chosen") else "",
                        "in the mean" if r.get("mean") else "",
                    )
                    if x
                ),
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
