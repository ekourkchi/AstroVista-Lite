"""The panels: ELLIPROF and the Source finder, layers, display, contours, the results and
the terminal."""

from __future__ import annotations

import html
import os
import re
import shlex
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import detection, guide, single
from .flow import FlowLayout

CONTROL = Qt.MetaModifier if sys.platform == "darwin" else Qt.ControlModifier
from .fitting import FitResult


def heading(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setStyleSheet("font-weight: 600; margin-top: 8px")
    return label


def title_row(title: str, topic: str, icon_name: str = "") -> QtWidgets.QHBoxLayout:
    """A panel's title (with the tool's icon before it) and a ? button for its guide topic."""
    row = QtWidgets.QHBoxLayout()
    if icon_name:
        from . import icons

        picture = QtWidgets.QLabel()
        picture.setPixmap(icons.icon(icon_name, 20).pixmap(20, 20))
        row.addWidget(picture)
    label = QtWidgets.QLabel(title)
    label.setStyleSheet("font-weight: 700; font-size: 14px")
    row.addWidget(label)
    row.addStretch(1)
    row.addWidget(guide.button(topic))
    return row


def tip(field, key: str) -> QtWidgets.QWidget:
    """A setting (a widget or a layout) with a small ? beside it that explains it."""
    box = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    if isinstance(field, QtWidgets.QLayout):
        row.addLayout(field, 1)
    else:
        row.addWidget(field, 1)
        if not field.toolTip() and key in guide.TIPS:
            field.setToolTip(re.sub("<[^>]+>", "", guide.TIPS[key]))
    row.addWidget(guide.tip_button(key), 0, Qt.AlignTop)
    return box


def note(text: str = "") -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: #8b919c")
    return label


def number(lo, hi, value, decimals=2, step=1.0, tip="") -> QtWidgets.QDoubleSpinBox:
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setToolTip(tip)
    return box


def whole(lo, hi, value, tip="") -> QtWidgets.QSpinBox:
    box = QtWidgets.QSpinBox()
    box.setRange(lo, hi)
    box.setValue(value)
    box.setToolTip(tip)
    return box


class BufferChoice(QtWidgets.QComboBox):
    """The buffer a result (a model, a residual, a segmentation image) goes into: Auto (a
    free one, or the one this result used before) or a buffer chosen.  Only the free
    buffers are offered, unless "Show the occupied buffers too" is chosen at the bottom;
    the buffer chosen stays in the list even once the result has filled it."""

    FREE = 8  # how many free buffers are offered
    SHOW_ALL = "show all"  # the data of the item at the bottom that lists the occupied ones

    def __init__(self, what: str, parent=None):
        super().__init__(parent)
        self.what = what
        self.occupied: dict = {}  # buffer number -> the name of the image in it
        self.show_all = False
        self._value: Optional[int] = None  # the buffer chosen (None: Auto)
        self.setToolTip(
            f"The buffer the {what} goes into.  Auto: a free buffer (or the one the {what} "
            "of this image went into before)"
        )
        self.activated.connect(self._chosen)
        self.set_buffers({})

    def value(self) -> Optional[int]:
        """The buffer chosen, or None for Auto."""
        return self._value

    def set_value(self, n: Optional[int]) -> None:
        self._value = n
        self._fill()

    def set_buffers(self, occupied: dict) -> None:
        """The buffers in use now (number -> image name): the list is rebuilt from them."""
        self.occupied = dict(occupied)
        self._fill()

    def _fill(self) -> None:
        keep = self._value
        self.blockSignals(True)
        self.clear()
        self.addItem("Auto", None)
        numbers, n = [], 1
        while len(numbers) < self.FREE:  # the first free buffers
            if n not in self.occupied:
                numbers.append(n)
            n += 1
        if self.show_all:
            numbers += list(self.occupied)
        if keep is not None:
            numbers.append(keep)
        for n in sorted(set(numbers)):
            if n in self.occupied:
                self.addItem(f"{n}  (replaces {self.occupied[n]})", n)
                self.setItemData(self.count() - 1, f"Replaces {self.occupied[n]}", Qt.ToolTipRole)
            else:
                self.addItem(f"{n}  (free)", n)
        if self.occupied and not self.show_all:
            self.insertSeparator(self.count())
            self.addItem("Show the occupied buffers too…", self.SHOW_ALL)
        self.setCurrentIndex(max(self.findData(keep), 0) if keep is not None else 0)
        self.blockSignals(False)

    def _chosen(self, index: int) -> None:
        data = self.itemData(index)
        if data == self.SHOW_ALL:  # list every buffer, then let them choose again
            self.show_all = True
            self._fill()
            QtCore.QTimer.singleShot(0, self.showPopup)
            return
        self._value = data if isinstance(data, int) else None


class InputChoice(QtWidgets.QComboBox):
    """The buffer a tool works on: the image shown (the default) or a buffer chosen from
    those loaded."""

    def __init__(self, what: str, parent=None):
        super().__init__(parent)
        self._value: Optional[int] = None
        self.setToolTip(
            f"The image the {what} works on: the one shown, or any buffer (its models, "
            "residuals and masks go with that image)"
        )
        self.activated.connect(lambda i: setattr(self, "_value", self.itemData(i)))
        self.set_buffers({})

    def value(self) -> Optional[int]:
        """The buffer chosen, or None for the image shown."""
        return self._value

    def set_value(self, n: Optional[int]) -> None:
        self._value = n
        self.setCurrentIndex(max(self.findData(n), 0) if n is not None else 0)

    def set_buffers(self, occupied: dict) -> None:
        """The buffers loaded now (number -> name); a chosen buffer that is gone goes back
        to the image shown."""
        if self._value is not None and self._value not in occupied:
            self._value = None
        self.blockSignals(True)
        self.clear()
        self.addItem("the image shown", None)
        for n in sorted(occupied):
            self.addItem(f"{n}  {occupied[n]}", n)
        self.setCurrentIndex(max(self.findData(self._value), 0) if self._value is not None else 0)
        self.blockSignals(False)


def residual_sources_row(rounds: int = 0, thresh: float = 8.0):
    """RESIDMASK: rounds of faint sources found in the residual, masked, fitted again; and
    their detection threshold.  (rounds box, threshold box, the row)"""
    r = whole(0, 5, rounds)
    r.setSpecialValueText("off")
    r.setToolTip(
        "RESIDMASK: after the fit, the faint sources its residual shows (globular clusters, "
        "stars and galaxies on the galaxy, which the source finder could not separate from "
        "its light) are found, masked and the fit made again, this many times; the layer "
        "residsources and a source run per round are made"
    )
    t = number(1.0, 100.0, thresh, decimals=1, step=0.5)
    t.setSuffix(" σ")
    t.setToolTip("RESIDTHRESH: their detection threshold, in units of the residual's local RMS")
    row = QtWidgets.QHBoxLayout()
    row.addWidget(r)
    row.addWidget(QtWidgets.QLabel("rounds, above"))
    row.addWidget(t)
    row.addStretch(1)
    return r, t, row


def residual_keywords(rounds, thresh) -> list:
    if not rounds.value():
        return []
    out = [f"RESIDMASK={rounds.value()}"]
    if abs(thresh.value() - 8.0) > 1e-9:
        out.append(f"RESIDTHRESH={thresh.value():g}")
    return out


def output_row(box: QtWidgets.QCheckBox, choice: BufferChoice) -> QtWidgets.QHBoxLayout:
    """A result's tick box with the buffer it goes into; the buffer is chosen only while
    the box is ticked."""
    row = QtWidgets.QHBoxLayout()
    row.addWidget(box, 1)
    row.addWidget(QtWidgets.QLabel("into"))
    row.addWidget(choice)
    choice.setEnabled(box.isChecked())
    box.toggled.connect(choice.setEnabled)
    return row


def scrolled(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.NoFrame)
    area.setWidget(widget)
    return area


from .skysection import SkySection  # noqa: E402  (it uses the helpers above)


class ElliprofPanel(SkySection, QtWidgets.QWidget):
    """The settings of an ELLIPROF fit; ``command`` turns them into a command line."""

    run_requested = QtCore.Signal()
    pick_requested = QtCore.Signal()
    source_centre_requested = QtCore.Signal()
    sky_requested = QtCore.Signal()
    mask_sources_requested = QtCore.Signal()
    radius_pick_requested = QtCore.Signal(str)  # "r0" or "r1"
    sky_segment_requested = QtCore.Signal()  # measure the sky by segmentation
    sky_mask_toggled = QtCore.Signal(bool)
    models_show_requested = QtCore.Signal(list)  # model numbers to put in buffers
    models_average_requested = QtCore.Signal(list)  # model numbers to average
    cancel_requested = QtCore.Signal()

    FIT_MODES = ["One elliprof call", "Automatic (as the pair fit)", "Iterative, my settings"]
    # the automatic settings: the pair fit's (see pairpanel.AUTO_RUN), for one galaxy
    AUTO = [
        "R1=AUTO",
        "NR=AUTO",
        "FIXCTR=1",
        "TIE=-3",
        "RMSTAR",
        "TAIL=measured",
        "SKYPLANE=1",
        "TRYFIX=1,0",
        "SHAPETOL=0.12",
        "HARMONICS=AUTO",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.addRow(title_row("ELLIPROF", "elliprof", "elliprof"))
        form.addRow(
            note(
                "Fits elliptical isophotes to a galaxy with the elliprof package, from the "
                "centre outwards."
            )
        )
        self.input = InputChoice("fit")
        form.addRow("Image", tip(self.input, "tool_input"))

        form.addRow(heading("Centre"))
        self.x0 = number(0, 1e6, 0, tip="X0: in pixels; the first pixel spans 0 to 1")
        self.y0 = number(0, 1e6, 0, tip="Y0: in pixels; the first pixel spans 0 to 1")
        form.addRow("X0", tip(self.x0, "x0"))
        form.addRow("Y0", tip(self.y0, "x0"))
        grid = QtWidgets.QGridLayout()
        pick = QtWidgets.QPushButton("Pick on the image")
        pick.setToolTip(
            "Click the galaxy's centre on the image. With 'Snap to sources', a click inside "
            "a found source takes that source's centre."
        )
        pick.clicked.connect(self.pick_requested)
        from_source = QtWidgets.QPushButton("Selected source")
        from_source.setToolTip(
            "The centre of the source selected in the Sources table (or double-click a row)"
        )
        from_source.clicked.connect(self.source_centre_requested)
        self.snap = QtWidgets.QCheckBox("Snap to sources")
        self.snap.setChecked(True)
        self.snap.setToolTip("Picking inside a found source takes the source's centre")
        grid.addWidget(pick, 0, 0)
        grid.addWidget(from_source, 1, 0)
        grid.addWidget(self.snap, 1, 1)
        form.addRow(grid)

        form.addRow(heading("Isophotes"))
        self.fit_mode = QtWidgets.QComboBox()
        self.fit_mode.addItems(self.FIT_MODES)
        self.fit_mode.setToolTip(
            "One elliprof call: the fit as elliprof makes it.\n"
            "Automatic: the pair fit's recipe for one galaxy: R1 where the light fades into "
            "the noise (fitted out to it at once: one galaxy has no growth), the search over centres, "
            "smoothing over 3 isophotes, RMSTAR, the light beyond R1 measured, the sky tilt, "
            "the outer-isophote check, every harmonic tried (the best kept, alternated with the "
            "search), the fits averaged and, with a measured sky, the sky refined 3 times.\nIterative, my settings: every one of those, to set"
        )
        form.addRow("Fit", tip(self.fit_mode, "elliprof_iterative"))
        self.r0 = number(0.1, 1e5, 5, tip="R0: semi-major axis of the first isophote, pixels")
        self.r1 = number(0.2, 1e5, 100, tip="R1: semi-major axis of the last isophote, pixels")
        self.nr = whole(2, 500, 20, tip="NR: number of isophotes from R0 to R1")
        self.niter = whole(1, 100, 10, tip="NITER: iterations of the fit")
        self.rmstar = QtWidgets.QCheckBox("Reject star-like bright samples (RMSTAR)")
        for key, box in (("r0", self.r0), ("r1", self.r1)):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(box, 1)
            pick = QtWidgets.QToolButton(text="Pick")
            pick.setToolTip(
                f"Click on the image at the distance {key.upper()} from the centre "
                "(along the galaxy's major axis)"
            )
            pick.clicked.connect(lambda _c=False, k=key: self.radius_pick_requested.emit(k))
            row.addWidget(pick)
            if key == "r1":
                self.r1_auto = QtWidgets.QCheckBox("auto")
                self.r1_auto.setToolTip(
                    "R1=AUTO (iterative fits): where the light fades into the noise"
                )
                self.r1_auto.toggled.connect(lambda on: self.r1.setEnabled(not on))
                row.addWidget(self.r1_auto)
            form.addRow(key.upper(), tip(row, key))
        self.nr_auto = QtWidgets.QCheckBox("auto")
        self.nr_auto.setToolTip("NR=AUTO (iterative fits): isophotes about 30 % apart")
        self.nr_auto.toggled.connect(lambda on: self.nr.setEnabled(not on))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.nr, 1)
        row.addWidget(self.nr_auto)
        form.addRow("NR", tip(row, "nr"))
        form.addRow("NITER", tip(self.niter, "niter"))
        form.addRow(tip(self.rmstar, "rmstar"))
        self.resid_rounds, self.resid_thresh, row = residual_sources_row()
        form.addRow("Residual sources", tip(row, "resid_sources"))
        self.iterative = self._iterative_box()
        form.addRow(self.iterative)

        # the sky: as the pair fit's (skysection.SkySection), and a clipped median
        self.build_sky(
            form,
            form.addRow,
            self.sky_segment_requested,
            self.sky_mask_toggled,
            "Sky",
            "Fit isophotes",
        )
        self.sky.setReadOnly(False)
        self.use_sky = QtWidgets.QCheckBox("Subtract the sky (base level × factor)")
        self.use_sky.setToolTip("Unticked: no sky is subtracted (elliprof's own way)")
        estimate = QtWidgets.QToolButton(text="Estimate")
        estimate.setToolTip(
            "A 3-sigma clipped median of the pixels the ticked mask layers leave (sky command)"
        )
        estimate.clicked.connect(self.sky_requested)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.use_sky, 1)
        row.addWidget(estimate)
        form.addRow(tip(row, "sky"))
        self.use_sky_button.clicked.connect(lambda: self.use_sky.setChecked(True))
        self.enter_sky_button.clicked.connect(lambda: self.use_sky.setChecked(True))

        form.addRow(heading("Masks"))
        self.mask_file = QtWidgets.QLineEdit()
        self.mask_file.setPlaceholderText("no mask file")
        self.mask_file.setToolTip("A mask elliprof reads (.dmask or FITS): 0 = masked, 1 = good")
        browse = QtWidgets.QToolButton(text="…")
        browse.clicked.connect(self._browse_mask)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.mask_file, 1)
        row.addWidget(browse)
        form.addRow("File", tip(row, "maskfile"))
        self.layers = QtWidgets.QListWidget()
        self._ticked = set()  # the layers ticked, kept while the list is rebuilt
        self.layers.itemChanged.connect(self._tick_changed)
        self.layers.setMinimumHeight(150)
        self.layers.setToolTip(
            "What masks the fit: the ticked mask layers, and the ticked buffers (where they "
            "are 0 or NaN), with the mask file"
        )
        form.addRow("Mask with", tip(self.layers, "maskwith"))
        sources = QtWidgets.QPushButton("Mask the found sources")
        sources.setToolTip(
            "A mask layer 'sources' with every source of the latest Source finder run, except "
            "the galaxy at the centre, each as its Kron ellipse times the size below"
        )
        sources.clicked.connect(self.mask_sources_requested)
        self.mask_scale = number(0.5, 10, 1.5, decimals=1, step=0.1)
        self.mask_scale.setToolTip("size of each source's mask, in Kron ellipses")
        form.addRow(sources)
        form.addRow("Size", tip(self.mask_scale, "masksize"))

        form.addRow(heading("Results"))
        self.make_model = QtWidgets.QCheckBox("Add the model image")
        self.make_residual = QtWidgets.QCheckBox("Add the residual image")
        self.make_model.setChecked(True)
        self.make_residual.setChecked(True)
        self.model_buffer = BufferChoice("model")
        self.residual_buffer = BufferChoice("residual")
        form.addRow(tip(output_row(self.make_model, self.model_buffer), "results"))
        form.addRow(output_row(self.make_residual, self.residual_buffer))

        self.run = QtWidgets.QPushButton("Fit isophotes")
        self.run.setDefault(True)
        self.run.clicked.connect(self.run_requested)
        self.cancel = QtWidgets.QPushButton("Cancel")
        self.cancel.setEnabled(False)
        self.cancel.setToolTip(
            "Stop an iterative fit after the elliprof calls running now, and keep nothing of it"
        )
        self.cancel.clicked.connect(self.cancel_requested)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.run, 3)
        row.addWidget(self.cancel, 1)
        form.addRow(row)
        self.status = note()
        form.addRow(self.status)
        self.progress = note()
        form.addRow(self.progress)
        self.calls = plain_table()
        self.calls.setColumnCount(5)
        self.calls.setHorizontalHeaderLabels(["call", "step", "R0–R1", "NR", "result"])
        self.calls.setMinimumHeight(170)
        self.calls.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.calls.verticalHeader().setDefaultSectionSize(22)
        self.calls.setToolTip(
            "The iterative fit as it runs: every elliprof call, passed or why it failed"
        )
        form.addRow(self.calls)
        self.models = plain_table()
        self.models.setColumnCount(len(self.MODEL_COLUMNS))
        self.models.setHorizontalHeaderLabels([c for c, _k in self.MODEL_COLUMNS])
        self.models.setMinimumHeight(240)
        self.models.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.models.verticalHeader().setVisible(False)
        self.models.verticalHeader().setDefaultSectionSize(22)
        self.models.setToolTip(
            "Every model made, with its parameters and its residual: between 1.15 × R0 and "
            "1.1 × R1 (as asked), in rings 20% apart, each ring's RMS over its own noise and "
            "every ring counted the same; core: the innermost rings. The lowest residual is "
            "kept (green, bold); red: a high-contrast core, never kept nor averaged; grey: "
            "listed only. Right-click a model for its parameters"
        )
        self.models.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.models.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.models.cellDoubleClicked.connect(
            lambda row, _c: self.models_show_requested.emit(self._model_numbers([row]))
        )
        self.models.itemSelectionChanged.connect(self._models_selection)
        self.models.setContextMenuPolicy(Qt.CustomContextMenu)
        self.models.customContextMenuRequested.connect(self._models_menu)
        # the models by step, one tab each (All first)
        self.model_tabs = QtWidgets.QTabBar()
        self.model_tabs.setExpanding(False)
        self.model_tabs.setDrawBase(False)
        self.model_tabs.setUsesScrollButtons(True)
        self.model_tabs.setElideMode(Qt.ElideNone)
        for key, label in [("", "All")] + [(k, lbl) for k, _p, lbl in single.CATEGORIES]:
            i = self.model_tabs.addTab(label)
            self.model_tabs.setTabData(i, key)
        self.model_tabs.setToolTip("The models of one step")
        self.model_tabs.currentChanged.connect(lambda _i: self._filter_models())
        self.models_show = QtWidgets.QPushButton("To buffers")
        self.models_show.setToolTip(
            "Each selected model and its residual (image − its sky − its model) into buffers "
            "(or double-click a row); right-click a buffer for its model's parameters"
        )
        self.models_show.clicked.connect(
            lambda: self.models_show_requested.emit(self._model_numbers())
        )
        self.models_average = QtWidgets.QPushButton("Average selected")
        self.models_average.setToolTip(
            "The mean of the selected models into buffers: the mean model and the residual "
            "image − mean sky − mean model, with its residual on the same rings"
        )
        self.models_average.clicked.connect(
            lambda: self.models_average_requested.emit(self._model_numbers())
        )
        self.models_bar = QtWidgets.QWidget()
        bar = QtWidgets.QHBoxLayout(self.models_bar)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addStretch(1)
        bar.addWidget(self.models_show)
        bar.addWidget(self.models_average)
        self.models_heading = heading("Models: the lowest residual is kept")
        form.addRow(self.models_heading)
        form.addRow(self.model_tabs)
        form.addRow(self.models)
        form.addRow(self.models_bar)
        self._models_selection()
        self._best_score = None
        for w in (self.calls, self.models_heading, self.model_tabs, self.models_bar, self.models):
            self.fit_mode.currentIndexChanged.connect(lambda i, w=w: w.setVisible(i > 0))
            w.setVisible(False)
        self.build_residual_sky(form, "Residual sky", "Fit isophotes")

    def _iterative_box(self) -> QtWidgets.QGroupBox:
        """The iterative fit's settings (Iterative, my settings), at the automatic values."""
        box = QtWidgets.QGroupBox("Iterative fit")
        f = QtWidgets.QFormLayout(box)
        f.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self.fixctr = QtWidgets.QComboBox()
        for label, v in (("fixed at X0, Y0", 1), ("free", 0), ("median of the fitted", 2)):
            self.fixctr.addItem(label, v)
        self.fixctr.setToolTip("FIXCTR: the centres of the isophotes")
        f.addRow("Centres", self.fixctr)
        self.tie = whole(-50, 10, -3, tip="TIE: -3 smooths over 3 isophotes; -1 none")
        f.addRow("Smoothing (TIE)", self.tie)
        self.cos3x = QtWidgets.QComboBox()
        self.cos4x = QtWidgets.QComboBox()
        for box_, values in (
            (
                self.cos3x,
                (
                    (2, "each isophote"),
                    (1, "median"),
                    (0, "none"),
                    (-2, "6th, each"),
                    (-1, "6th, median"),
                    (-3, "6th, none"),
                ),
            ),
            (self.cos4x, ((2, "each isophote"), (1, "median"), (0, "none"))),
        ):
            for v, label in values:
                box_.addItem(f"{v}: {label}", v)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("COS3X"))
        row.addWidget(self.cos3x)
        row.addWidget(QtWidgets.QLabel("COS4X"))
        row.addWidget(self.cos4x)
        f.addRow("Harmonics", row)
        self.all_harmonics = QtWidgets.QCheckBox("Search every harmonic")
        self.all_harmonics.setToolTip("HARMONICS=AUTO: every COS3X and COS4X tried on the fit kept")
        f.addRow(self.all_harmonics)
        self.search_fix = QtWidgets.QCheckBox("Search fixed and free centres")
        self.search_fix.setChecked(True)
        self.search_fix.setToolTip("TRYFIX=1,0: both fitted, the lower residual kept")
        f.addRow(self.search_fix)
        self.search_wide = QtWidgets.QCheckBox("Search R0, TIE, the centre and NR too")
        self.search_wide.setChecked(True)
        self.search_wide.setToolTip(
            "SEARCH=FULL: TIE off and a larger R0 (out of a masked core) tried as well, the "
            "fixed centre moved to 1/8 px and the number of isophotes changed, the lowest "
            "residual kept (unticked: SEARCH=BASIC)"
        )
        f.addRow(self.search_wide)
        self.sky_plane = QtWidgets.QCheckBox("Take out a sky tilt")
        self.sky_plane.setChecked(True)
        self.sky_plane.setToolTip("SKYPLANE=1: a plane fitted to the residual, kept if it helps")
        f.addRow(self.sky_plane)
        self.tail = QtWidgets.QComboBox()
        for label, key in (
            ("measured in rings", "measured"),
            ("Sérsic law", "sersic"),
            ("elliprof's extrapolation", "elliprof"),
            ("exponential", "exp"),
            ("cut at R1", "cut"),
        ):
            self.tail.addItem(label, key)
        self.tail.setToolTip("TAIL: the model beyond the last isophote")
        f.addRow("Beyond R1", self.tail)
        self.shape_tol = number(0.0, 1.0, 0.12, decimals=2, step=0.02)
        self.shape_tol.setSpecialValueText("off")
        self.shape_tol.setToolTip(
            "SHAPETOL: an outermost isophote off the trend inside it by more than this is "
            "refitted with coarser isophotes or a smaller R1"
        )
        f.addRow("Outer isophotes off by", self.shape_tol)
        self.parallel = whole(1, 16, max(1, min(5, (os.cpu_count() or 1) - 1)))
        self.parallel.setToolTip(
            "PARALLEL: N fits, all averaged (the fit kept and N - 1 with R0 and R1 varied), "
            "N at a time; buffers get the mean model and residual and each fit's own"
        )
        self.jitter = whole(1, 50, 15)
        self.jitter.setSuffix(" %")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.parallel)
        row.addWidget(QtWidgets.QLabel("varied by"))
        row.addWidget(self.jitter)
        row.addStretch(1)
        f.addRow("Fits to average", row)
        box.setVisible(False)
        self.fit_mode.currentIndexChanged.connect(lambda i: box.setVisible(i == 2))
        return box

    MODEL_COLUMNS = [
        ("#", "n"),
        ("step", "stage"),
        ("X0", "x0"),
        ("Y0", "y0"),
        ("FIXCTR", "fixctr"),
        ("COS3X", "cos3x"),
        ("COS4X", "cos4x"),
        ("TIE", "tie"),
        ("R0", "r0"),
        ("R1", "r1"),
        ("reached", "r1_reached"),
        ("NR", "nr"),
        ("sky", "sky"),
        ("tilt", "tilt"),
        ("core", "core"),
        ("residual", "score"),
    ]

    def clear_calls(self) -> None:
        self.calls.setRowCount(0)
        self.models.setRowCount(0)
        self.progress.setText("")
        self._best_score = None

    def add_call(self, info: dict) -> None:
        """A row of the calls table: an elliprof call, passed or why it failed."""
        row = self.calls.rowCount()
        self.calls.insertRow(row)
        ok = info.get("passed")
        stage = info.get("stage", "")
        if info.get("sky_iteration"):
            stage += f" (sky {info['sky_iteration']}: ×{info['sky_factor']:.5f})"
        cells = [
            str(info.get("call", "")),
            stage,
            f"{info.get('r0', 0):g}–{info.get('r1', 0):.1f}",
            str(info.get("nr", "")),
            "passed" if ok else f"FAILED: {info.get('reason', '')}",
        ]
        for col, text in enumerate(cells):
            item = QtWidgets.QTableWidgetItem(text)
            if col == 4 and not ok:
                item.setForeground(QtGui.QColor("#f85149"))
            self.calls.setItem(row, col, item)
        self.calls.scrollToBottom()

    def add_model(self, info: dict) -> None:
        """A row of the models table: a finished model's parameters and its residual
        (green: the lowest so far of those compared)."""
        for row in range(self.models.rowCount()):  # a model already listed
            item = self.models.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == info.get("n"):
                return
        row = self.models.rowCount()
        self.models.insertRow(row)
        sc = info.get("score")
        compared = info.get("chooses", True)
        best = False
        if compared and info.get("status") == "fitted" and sc is not None:
            best = self._best_score is None or sc < self._best_score
            if best:
                self._best_score = sc
        for col, (_label, key) in enumerate(self.MODEL_COLUMNS):
            v = info.get(key)
            if key == "score":
                text = (
                    f"{v:.3f}"
                    if info.get("status") == "fitted" and v is not None
                    else str(info.get("status", ""))
                )
            elif key == "core":
                text = "-" if v is None else f"{v:.2f}"
            elif key == "tilt":
                text = "yes" if v else "no"
            elif key == "tie":
                text = "off" if v is None else str(v)
            elif key in ("x0", "y0"):
                text = "-" if v is None else f"{v:.3f}"
            elif key in ("r0", "r1", "r1_reached"):
                text = f"{v:.1f}"
            elif key == "sky":
                text = f"{v:.6g}"
            else:
                text = str(v)
            item = QtWidgets.QTableWidgetItem(text)
            if key == "n":
                item.setData(Qt.UserRole + 2, dict(info))
            if key in ("n", "score"):
                item.setData(Qt.UserRole, info.get("n"))
            if key == "score":
                item.setData(Qt.UserRole + 1, info.get("status"))
            if key == "stage":
                item.setData(Qt.UserRole + 1, single.category(str(v)))
            if not compared:
                item.setForeground(QtGui.QColor("#8b949e"))
            elif key == "score" and best:
                item.setForeground(QtGui.QColor("#3fb950"))
            self.models.setItem(row, col, item)
        self._style_flag(row, info)
        want = self._tab_key()
        if want:
            self.models.setRowHidden(row, single.category(str(info.get("stage", ""))) != want)
        self._count_tabs()
        self.models.scrollToBottom()

    def _style_flag(self, row: int, info: dict) -> None:
        """A high-contrast core in red, a fit left out of the mean in italics."""
        flag = info.get("flag")
        for col, (_label, key) in enumerate(self.MODEL_COLUMNS):
            cell = self.models.item(row, col)
            if cell is None:
                continue
            if flag == "core" and key in ("core", "score"):
                cell.setForeground(QtGui.QColor("#f85149"))
            f = cell.font()
            f.setItalic(flag == "poor")
            cell.setFont(f)
            tip = {
                "core": "a high-contrast core: never kept nor averaged",
                "poor": "left out of the mean (its residual or R1 reached)",
            }.get(flag)
            if info.get("averaged"):
                tip = "in the mean"
            if tip:
                cell.setToolTip(tip)

    def _tab_key(self) -> str:
        return self.model_tabs.tabData(self.model_tabs.currentIndex()) or ""

    def _count_tabs(self) -> None:
        """Each tab with how many models it has."""
        counts = {}
        for row in range(self.models.rowCount()):
            item = self.models.item(row, 1)
            cat = item.data(Qt.UserRole + 1) if item is not None else ""
            counts[cat] = counts.get(cat, 0) + 1
        labels = dict([("", "All")] + [(k, lbl) for k, _p, lbl in single.CATEGORIES])
        for i in range(self.model_tabs.count()):
            key = self.model_tabs.tabData(i) or ""
            k = self.models.rowCount() if not key else counts.get(key, 0)
            self.model_tabs.setTabText(i, f"{labels[key]} ({k})" if k else labels[key])
            self.model_tabs.setTabEnabled(i, bool(k) or not key)

    def fill_models(self, rows) -> None:
        """The models table of a finished fit (every model, the kept one marked)."""
        self.models.setRowCount(0)
        self._best_score = None
        kept = None
        for r in rows:
            self.add_model(r)
            if r.get("kept"):
                kept = r["n"]
        if kept is not None:
            self.mark_kept_model(kept)
        self._filter_models()

    def model_row(self, row: int) -> Optional[dict]:
        """The parameters of the model in table row ``row``."""
        item = self.models.item(row, 0)
        return item.data(Qt.UserRole + 2) if item is not None else None

    def model_text(self, row: int) -> str:
        info = self.model_row(row)
        if not info:
            return ""
        return single.describe(info, self.input.value() or 1)

    def _models_menu(self, pos) -> None:
        """Right-click on a model: its parameters, to buffers, the mean of the selected."""
        item = self.models.itemAt(pos)
        if item is None:
            return
        row = item.row()
        info = self.model_row(row) or {}
        menu = QtWidgets.QMenu(self)
        menu.addAction(
            f"Parameters of model #{info.get('n', '')}…",
            lambda: show_text(self, f"Model #{info.get('n', '')}", self.model_text(row)),
        )
        menu.addAction(
            "Copy its elliprof command",
            lambda: QtWidgets.QApplication.clipboard().setText(
                single.command_for(info, self.input.value() or 1)
            ),
        )
        menu.addSeparator()
        numbers = self._model_numbers()
        here = self._model_numbers([row])
        menu.addAction("To buffers", lambda: self.models_show_requested.emit(here)).setEnabled(
            bool(here)
        )
        if len(numbers) >= 2:
            menu.addAction(
                f"Average {len(numbers)} selected",
                lambda: self.models_average_requested.emit(numbers),
            )
        menu.exec(self.models.viewport().mapToGlobal(pos))

    def _model_numbers(self, rows=None) -> list:
        """The model numbers of ``rows`` (default: the selected rows), complete ones only."""
        if rows is None:
            rows = sorted({i.row() for i in self.models.selectedIndexes()})
        out = []
        for row in rows:
            item = self.models.item(row, 0)
            done = self.models.item(row, len(self.MODEL_COLUMNS) - 1)
            if item is None or self.models.isRowHidden(row):
                continue
            if done is not None and done.data(Qt.UserRole + 1) == "fitted":
                out.append(int(item.data(Qt.UserRole)))
        return out

    def _models_selection(self) -> None:
        k = len(self._model_numbers())
        self.models_show.setEnabled(k >= 1)
        self.models_average.setEnabled(k >= 2)
        self.models_average.setText(f"Average {k} selected" if k >= 2 else "Average selected")

    def _filter_models(self) -> None:
        want = self._tab_key()
        for row in range(self.models.rowCount()):
            item = self.models.item(row, 1)
            cat = item.data(Qt.UserRole + 1) if item is not None else ""
            self.models.setRowHidden(row, bool(want) and cat != want)
        self._models_selection()

    def mark_kept_model(self, number: int) -> None:
        """The model kept: its row green and bold, and shown."""
        for row in range(self.models.rowCount()):
            item = self.models.item(row, 0)
            kept = item is not None and item.data(Qt.UserRole) == number
            for col in range(self.models.columnCount()):
                cell = self.models.item(row, col)
                if cell is None:
                    continue
                f = cell.font()
                f.setBold(kept)
                cell.setFont(f)
                if kept:
                    cell.setForeground(QtGui.QColor("#3fb950"))
                elif (
                    col == self.models.columnCount() - 1
                    and cell.foreground().color() == QtGui.QColor("#3fb950")
                ):
                    cell.setData(Qt.ForegroundRole, None)  # no longer the best: plain
            if kept:
                self.models.scrollToItem(item)

    def is_iterative(self) -> bool:
        return self.fit_mode.currentIndex() > 0

    def iterative_keywords(self, measured_sky: bool) -> list:
        """The keywords of the iterative fit (Automatic, or the box's settings)."""
        if self.fit_mode.currentIndex() == 1:
            out = list(self.AUTO)
            if measured_sky:
                out.append("SKYITER=3")
            if self.parallel.value() > 1:
                out.append(f"PARALLEL={self.parallel.value()}")
            return out
        out = [f"FIXCTR={self.fixctr.currentData()}", f"TIE={self.tie.value()}"]
        for key, b in (("COS3X", self.cos3x), ("COS4X", self.cos4x)):
            if b.currentData() != 2:
                out.append(f"{key}={b.currentData()}")
        if self.all_harmonics.isChecked():
            out.append("HARMONICS=AUTO")
        if self.search_fix.isChecked():
            out.append("TRYFIX=1,0")
        if not self.search_wide.isChecked():
            out.append("SEARCH=BASIC")
        if self.sky_iter.value() and measured_sky:
            out.append(f"SKYITER={self.sky_iter.value()}")
        if self.sky_plane.isChecked():
            out.append("SKYPLANE=1")
        out.append(f"TAIL={self.tail.currentData()}")
        if self.shape_tol.value():
            out.append(f"SHAPETOL={self.shape_tol.value():g}")
        if self.parallel.value() > 1:
            out.append(f"PARALLEL={self.parallel.value()}")
            if self.jitter.value() != 15:
                out.append(f"JITTER={self.jitter.value() / 100:g}")
        return out

    def _browse_mask(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "A mask for ELLIPROF", "", "Masks (*.dmask *.fits *.fit *.fts);;All files (*)"
        )
        if path:
            self.mask_file.setText(path)

    def set_centre(self, x: float, y: float) -> None:
        self.x0.setValue(round(x, 2))
        self.y0.setValue(round(y, 2))

    def set_mask_layers(self, names, buffers=()) -> None:
        """List what can mask the fit: the mask layers that fit the image, then the image
        buffers of its size (where they are 0 or NaN), with the ticks set before."""
        self.layers.blockSignals(True)
        self.layers.clear()

        def group(text):
            item = QtWidgets.QListWidgetItem(text)
            item.setFlags(Qt.NoItemFlags)  # a heading, not a choice
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            self.layers.addItem(item)

        group("Mask layers" if names else "Mask layers: none yet")
        for name in names:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in self._ticked else Qt.Unchecked)
            item.setToolTip(f"the mask layer {name}")
            self.layers.addItem(item)
        group("Buffers (0 or NaN = masked)" if buffers else "Buffers: none of this size")
        for n, label in buffers:
            key = f"buffer {n}"
            item = QtWidgets.QListWidgetItem(f"{key}: {label}")
            item.setData(Qt.UserRole, n)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if key in self._ticked else Qt.Unchecked)
            item.setToolTip(f"buffer {n} used as a mask: its pixels that are 0 or NaN are masked")
            self.layers.addItem(item)
        self.layers.blockSignals(False)

    @staticmethod
    def _key(item) -> str:
        n = item.data(Qt.UserRole)
        return item.text() if n is None else f"buffer {n}"

    def tick(self, *names) -> None:
        """Use these mask layers in the fit (now, or as soon as they exist)."""
        self._ticked |= set(names)
        for k in range(self.layers.count()):
            item = self.layers.item(k)
            if item.text() in names:
                item.setCheckState(Qt.Checked)

    def _tick_changed(self, item) -> None:
        if item.checkState() == Qt.Checked:
            self._ticked.add(self._key(item))
        else:
            self._ticked.discard(self._key(item))

    def _ticked_items(self):
        return [
            self.layers.item(k)
            for k in range(self.layers.count())
            if self.layers.item(k).checkState() == Qt.Checked
        ]

    def ticked_layers(self):
        """The mask layers ticked."""
        return [
            i.text()
            for i in self._ticked_items()
            if i.data(Qt.UserRole) is None and i.flags() & Qt.ItemIsUserCheckable
        ]

    def ticked_buffers(self):
        """The buffers ticked to be used as masks."""
        return [
            i.data(Qt.UserRole) for i in self._ticked_items() if i.data(Qt.UserRole) is not None
        ]

    def set_values(self, values: dict) -> None:
        """Fill the panel (keys of FitSettings)."""
        for key in ("x0", "y0", "r0", "r1"):
            if key in values:
                getattr(self, key).setValue(float(values[key]))
        for key in ("nr", "niter"):
            if key in values:
                getattr(self, key).setValue(int(values[key]))
        if "rmstar" in values:
            self.rmstar.setChecked(bool(values["rmstar"]))
        if "sky" in values:
            self.use_sky.setChecked(values["sky"] is not None)
            if values["sky"] is not None:
                self.set_sky_value(float(values["sky"]), "from the fit's settings")
        if "mask_file" in values:
            self.mask_file.setText(str(values["mask_file"] or ""))

    def command(
        self, n: int, tomodel: Optional[int], toresid: Optional[int], measured_sky: bool = False
    ) -> str:
        """The elliprof command line for buffer ``n`` (``measured_sky``: the image has a
        current sky measurement, which the sky refinement needs)."""
        parts = [f"elliprof {n}", f"X0={self.x0.value():g}", f"Y0={self.y0.value():g}"]
        iterative = self.is_iterative()
        auto = self.fit_mode.currentIndex() == 1
        r1 = "AUTO" if iterative and (auto or self.r1_auto.isChecked()) else f"{self.r1.value():g}"
        nr = "AUTO" if iterative and (auto or self.nr_auto.isChecked()) else str(self.nr.value())
        parts += [f"R0={self.r0.value():g}", f"R1={r1}"]
        parts += [f"NR={nr}", f"NITER={self.niter.value()}"]
        if self.use_sky.isChecked():
            parts.append(f"SKY={self.sky.value():.10g}")
            if abs(self.factor.value() - 1.0) > 1e-9:
                parts.append(f"SKYFACTOR={self.factor.value():.6g}")
        if iterative:
            parts += [
                w for w in self.iterative_keywords(measured_sky) if w not in ("R1=AUTO", "NR=AUTO")
            ]
        if self.rmstar.isChecked() and "RMSTAR" not in parts:
            parts.append("RMSTAR")
        parts += residual_keywords(self.resid_rounds, self.resid_thresh)
        mask = self.mask_file.text().strip()
        if mask:
            parts.append("MASKFILE=" + shlex.quote(mask))
        layers = self.ticked_layers()
        if layers:
            parts.append("MASK=" + ",".join(layers))
        # a buffer this fit writes cannot also mask it
        buffers = [b for b in self.ticked_buffers() if b not in (n, tomodel, toresid)]
        if buffers:
            parts.append("MASKBUF=" + ",".join(str(b) for b in buffers))
        if tomodel:
            parts.append(f"TOMODEL={tomodel}")
        if toresid:
            parts.append(f"TORESID={toresid}")
        return " ".join(parts)


class SourceFinderPanel(QtWidgets.QWidget):
    """The Source finder: the settings of a run (in SExtractor's keyword names)."""

    run_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.addRow(title_row("Source finder", "sources", "sources"))
        form.addRow(
            note(
                "Finds the stars and galaxies in the image and measures them: position, "
                "shape, flux, magnitude and size."
            )
        )
        self.input = InputChoice("source finder")
        form.addRow("Image", tip(self.input, "tool_input"))
        self.fields = {}
        group = None
        for key, label, default, kind, explanation, section in detection.SETTINGS:
            if section != group:
                group = section
                form.addRow(heading(section))
            if kind is int:
                box = whole(0, 100000, default)
            elif kind is float:
                decimals = 4 if key == "DEBLEND_MINCONT" else 2
                box = number(-1e9, 1e9, default, decimals=decimals, step=0.5)
            elif isinstance(kind, list):
                box = QtWidgets.QComboBox()
                box.addItems(kind)
                box.setCurrentText(str(default))
            else:
                box = QtWidgets.QLineEdit(str(default))
            box.setToolTip(f"{key}: {explanation}")
            self.fields[key] = box
            form.addRow(label, tip(box, key) if key in guide.TIPS else box)
        self.masks = QtWidgets.QLineEdit()
        self.masks.setPlaceholderText("none")
        self.masks.setToolTip(
            "Mask layers whose pixels are left out of the detection (names, comma separated)"
        )
        form.addRow(heading("Masks"))
        form.addRow("Leave out", tip(self.masks, "MASK"))
        reset = QtWidgets.QPushButton("Default settings")
        reset.clicked.connect(self.reset)
        form.addRow(reset)
        self.show_sources = QtWidgets.QCheckBox("Show the sources on the image")
        self.show_sources.setChecked(True)
        self.add_segmentation = QtWidgets.QCheckBox("Add the segmentation image")
        self.segmentation_buffer = BufferChoice("segmentation image")
        form.addRow(self.show_sources)
        form.addRow(output_row(self.add_segmentation, self.segmentation_buffer))
        self.run = QtWidgets.QPushButton("Find sources")
        self.run.clicked.connect(self.run_requested)
        form.addRow(self.run)
        self.status = note()
        form.addRow(self.status)

    def refresh(self) -> None:
        """Nothing to check: the Source finder is part of this package."""
        self.run.setEnabled(True)

    @staticmethod
    def _read(box):
        if isinstance(box, QtWidgets.QComboBox):
            return box.currentText()
        if isinstance(box, QtWidgets.QLineEdit):
            return box.text().strip()
        return box.value()

    @staticmethod
    def _write(box, value) -> None:
        if isinstance(box, QtWidgets.QComboBox):
            box.setCurrentText(str(value))
        elif isinstance(box, QtWidgets.QLineEdit):
            box.setText(str(value))
        else:
            box.setValue(value)

    def reset(self) -> None:
        for key, box in self.fields.items():
            self._write(box, detection.DEFAULTS[key])
        self.masks.clear()

    def settings(self) -> dict:
        return {key: self._read(box) for key, box in self.fields.items()}

    def command(self, n: int, toseg: Optional[int] = None) -> str:
        """The sources command line for buffer ``n`` (only the settings that differ from
        the defaults are written)."""
        parts = [f"sources {n}"]
        for key, value in self.settings().items():
            if value != detection.DEFAULTS[key]:
                text = f"{value:g}" if isinstance(value, float) else str(value)
                parts.append(f"{key}=" + (shlex.quote(text) if " " in text else text))
        masks = ",".join(m for m in re.split(r"[,\s]+", self.masks.text()) if m)
        if masks:
            parts.append(f"MASK={masks}")
        if toseg:
            parts.append(f"TOSEG={toseg}")
        return " ".join(parts)

    def set_values(self, values: dict) -> None:
        for key, value in values.items():
            if key in self.fields:
                self._write(self.fields[key], value)


class NumberItem(QtWidgets.QTableWidgetItem):
    """A table cell that sorts by its number, not its text."""

    def __init__(self, value: float, text: str):
        super().__init__(text)
        self.value = value
        self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setFlags(self.flags() & ~Qt.ItemIsEditable)

    def __lt__(self, other):
        if isinstance(other, NumberItem):
            return self.value < other.value
        return super().__lt__(other)


def fill_table(table: QtWidgets.QTableWidget, columns, rows) -> None:
    """``columns``: [(header, formatter)]; ``rows``: list of lists of numbers."""
    table.setSortingEnabled(False)
    table.clear()
    table.setColumnCount(len(columns))
    table.setRowCount(len(rows))
    table.setHorizontalHeaderLabels([c[0] for c in columns])
    for r, values in enumerate(rows):
        for c, value in enumerate(values):
            v = float(value)
            text = columns[c][1].format(v) if np.isfinite(v) else ""
            table.setItem(r, c, NumberItem(v, text))
    table.resizeColumnsToContents()


def show_text(parent, title: str, text: str) -> None:
    """A text to read and copy (a model's parameters)."""
    box = QtWidgets.QDialog(parent)
    box.setWindowTitle(title)
    lay = QtWidgets.QVBoxLayout(box)
    view = QtWidgets.QPlainTextEdit(text)
    view.setReadOnly(True)
    view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
    view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
    view.setMinimumSize(640, 300)
    lay.addWidget(view)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
    copy = buttons.addButton("Copy", QtWidgets.QDialogButtonBox.ActionRole)
    copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(text))
    buttons.rejected.connect(box.reject)
    lay.addWidget(buttons)
    box.setAttribute(Qt.WA_DeleteOnClose)
    box.show()


def plain_table() -> QtWidgets.QTableWidget:
    table = QtWidgets.QTableWidget()
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    return table


# the profile's columns: header, number format, explanation
PROFILE_COLUMNS = [
    ("Rmaj", "{:.2f}", "semi-major axis, pixels"),
    ("I0", "{:.5g}", "intensity of the isophote"),
    ("ellip", "{:.4f}", "ellipticity 1 - b/a"),
    ("alpha", "{:.2f}", "position angle as ELLIPROF stores it, degrees"),
    ("x0", "{:.2f}", "centre x"),
    ("y0", "{:.2f}", "centre y"),
    ("I3", "{:.4f}", "amplitude of the cos 3θ term"),
    ("A3", "{:.2f}", "phase of the cos 3θ term"),
    ("I4", "{:.4f}", "amplitude of the cos 4θ term (boxy or discy)"),
    ("A4", "{:.2f}", "phase of the cos 4θ term"),
    ("slope", "{:.4f}", "logarithmic slope of the profile"),
]
# the profile written to a file: more decimals than the table shows
PROFILE_FORMATS = {
    "Rmaj": "{:.3f}",
    "I0": "{:.6e}",
    "ellip": "{:.5f}",
    "alpha": "{:.3f}",
    "x0": "{:.3f}",
    "y0": "{:.3f}",
    "I3": "{:.5f}",
    "A3": "{:.3f}",
    "I4": "{:.5f}",
    "A4": "{:.3f}",
    "slope": "{:.5f}",
}
PLOTS = {
    "Intensity I0": ("I0", True),
    "Ellipticity": ("ellip", False),
    "Position angle": ("alpha", False),
    "Centre x0": ("x0", False),
    "Centre y0": ("y0", False),
    "cos 3θ amplitude I3": ("I3", False),
    "cos 3θ angle A3": ("A3", False),
    "cos 4θ amplitude I4": ("I4", False),
    "cos 4θ angle A4": ("A4", False),
    "Slope": ("slope", False),
}


class ProfilePanel(QtWidgets.QWidget):
    """The fitted profile: a plot against the semi-major axis (or its table of numbers)."""

    row_selected = QtCore.Signal(int)
    save_png = QtCore.Signal()  # "Save PNG…": the plot as a figure
    save_csv = QtCore.Signal()  # "Save CSV…": the table, columns aligned

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: Optional[FitResult] = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.quantity = QtWidgets.QComboBox()
        self.quantity.addItems(list(PLOTS))
        self.quarter = QtWidgets.QCheckBox("R^¼ axis")
        self.quarter.setToolTip("Plot against the fourth root of the semi-major axis")
        self.as_table = QtWidgets.QCheckBox("Table")
        self.as_table.setToolTip("The profile's numbers instead of the plot")
        self.devauc = QtWidgets.QCheckBox("R¼ law")
        self.devauc.setToolTip(
            "Over-plot the de Vaucouleurs (R^¼) law fitted to the intensity profile, with "
            "its effective radius Re, the intensity there Ie, and the scatter about it"
        )
        # the controls wrap onto more rows when the panel is narrow
        holder = QtWidgets.QWidget()
        flow = FlowLayout(holder, spacing=10)
        # what is shown is what is saved: the plot as PNG, the table as CSV
        self.save = save = QtWidgets.QToolButton()
        save.clicked.connect(
            lambda: (self.save_csv if self.as_table.isChecked() else self.save_png).emit()
        )
        for w in (self.quantity, self.quarter, self.devauc, self.as_table, save):
            flow.addWidget(w)
        flow.addWidget(guide.button("profile"))
        lay.addWidget(holder)
        # one area: the plot, or (ticked) the table of numbers
        self.stack = QtWidgets.QStackedWidget()
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.table = plain_table()
        self.table.itemSelectionChanged.connect(self._selected)
        self.stack.addWidget(self.plot)
        self.stack.addWidget(self.table)
        lay.addWidget(self.stack, 1)
        self.as_table.toggled.connect(lambda on: self.stack.setCurrentIndex(1 if on else 0))
        self.as_table.toggled.connect(lambda _on: self._save_label())
        self._save_label()
        self.quantity.currentTextChanged.connect(lambda _t: self.draw())
        self.quarter.toggled.connect(lambda _t: self.draw())
        self.devauc.toggled.connect(lambda _t: self.draw())
        self.fit_note = QtWidgets.QLabel()
        self.fit_note.setWordWrap(True)
        self.fit_note.setStyleSheet("color: #8b9099")
        lay.addWidget(self.fit_note)

    def set_result(self, result: Optional[FitResult]) -> None:
        self.result = result
        if result is None:
            self.table.clear()
            self.table.setRowCount(0)
            self.plot.clear()
            return
        p = result.profile
        columns = [(name, fmt) for name, fmt, _ in PROFILE_COLUMNS if name in p.columns]
        rows = [[row[name] for name, _ in columns] for _, row in p.iterrows()]
        fill_table(self.table, columns, rows)
        for c, (name, _fmt, explanation) in enumerate(PROFILE_COLUMNS):
            item = self.table.horizontalHeaderItem(c)
            if item is not None:
                item.setToolTip(explanation)
        self.draw()

    def draw(self) -> None:
        self.plot.clear()
        if self.result is None:
            return
        p = self.result.profile
        column, logarithmic = PLOTS[self.quantity.currentText()]
        r = p["Rmaj"].to_numpy(float)
        y = p[column].to_numpy(float)
        x = r**0.25 if self.quarter.isChecked() else r
        if logarithmic:
            keep = y > 0
            x, y = x[keep], y[keep]
        self.plot.setLogMode(x=False, y=logarithmic)
        self.plot.plot(x, y, pen=pg.mkPen("#4fa3ff", width=1.5), symbol="o", symbolSize=6)
        self.fit_note.setText("")
        if self.devauc.isChecked():
            self._devauc(p, r)
        self.plot.setLabel(
            "bottom", "Rmaj^¼ (pixels^¼)" if self.quarter.isChecked() else "Rmaj (pixels)"
        )
        self.plot.setLabel("left", self.quantity.currentText())

    def _devauc(self, profile, r) -> None:
        """Fit and draw the de Vaucouleurs law (exports.devauc_fit)."""
        from .exports import devauc_fit

        law = devauc_fit(r, profile["I0"].to_numpy(float))
        if "error" in law:
            self.fit_note.setText(law["error"])
            return
        column, _logarithmic = PLOTS[self.quantity.currentText()]
        if column == "I0":  # drawn over the intensity profile
            rr = np.linspace(law["rmin"], law["rmax"], 200)
            curve = np.exp(law["intercept"] + law["slope"] * rr**0.25)
            x = rr**0.25 if self.quarter.isChecked() else rr
            self.plot.plot(x, curve, pen=pg.mkPen("#ffb347", width=2, style=Qt.DashLine))
        self.fit_note.setText(
            f"R¼ law: Re = {law['re']:.1f} px, Ie = {law['ie']:.4g}, scatter "
            f"{law['scatter']:.3f} mag ({law['n']} ellipses)"
            + ("" if column == "I0" else "   (choose Intensity I0 to see the curve)")
        )

    def _save_label(self) -> None:
        if self.as_table.isChecked():
            self.save.setText("Save CSV…")
            self.save.setToolTip("Save the table as CSV, every column aligned")
        else:
            self.save.setText("Save PNG…")
            self.save.setToolTip("Save the plot as a PNG figure (made with matplotlib)")

    def write_csv(self, path, profile=None) -> None:
        """The profile table (or ``profile``) as CSV with its columns aligned: the table's
        columns first, in its order, then any others elliprof gave."""
        from .exports import aligned_csv

        p = self.result.profile if profile is None else profile
        names = [n for n, _f, _e in PROFILE_COLUMNS if n in p.columns]
        names += [n for n in p.columns if n not in names]
        aligned_csv(path, {n: p[n].to_numpy() for n in names}, PROFILE_FORMATS)

    def write_png(self, path, title: str = "") -> None:
        """The plot as shown (quantity, axis, R¼ law) as a PNG figure."""
        from .exports import profile_png

        label = self.quantity.currentText()
        column, logarithmic = PLOTS[label]
        profile_png(
            path,
            self.result.profile,
            label,
            column,
            logarithmic,
            self.quarter.isChecked(),
            self.devauc.isChecked(),
            title,
        )

    def _selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows and self.result is not None:
            # the row's Rmaj identifies the isophote (the table may be sorted)
            rmaj = self.table.item(rows[0].row(), 0).value
            index = int(np.argmin(np.abs(self.result.profile["Rmaj"].to_numpy(float) - rmaj)))
            self.row_selected.emit(index)


# the source table's columns: catalogue column, header, number format
SOURCE_COLUMNS = [
    ("NUMBER", "#", "{:.0f}"),
    ("X_IMAGE", "x", "{:.2f}"),
    ("Y_IMAGE", "y", "{:.2f}"),
    ("MAG_AUTO", "mag", "{:.3f}"),
    ("MAGERR_AUTO", "± mag", "{:.3f}"),
    ("FLUX_AUTO", "flux", "{:.5g}"),
    ("FLUX_RADIUS", "r½", "{:.2f}"),
    ("FWHM_IMAGE", "FWHM", "{:.2f}"),
    ("ELLIPTICITY", "ellip", "{:.3f}"),
    ("THETA_IMAGE", "θ", "{:.1f}"),
    ("CLASS_STAR", "star", "{:.2f}"),
    ("FLAGS", "flags", "{:.0f}"),
    ("ALPHA_J2000", "RA", "{:.6f}"),
    ("DELTA_J2000", "Dec", "{:.6f}"),
]


class SourcesPanel(QtWidgets.QWidget):
    """The sources of the Source finder's runs: one run at a time."""

    source_selected = QtCore.Signal(int, int)  # run number, row in the run's catalogue
    source_activated = QtCore.Signal(int, int)  # a double-click: use it as ELLIPROF's centre
    run_changed = QtCore.Signal(int)
    save_csv = QtCore.Signal()  # "Save CSV…": this run's sources

    def __init__(self, parent=None):
        super().__init__(parent)
        self.runs: List[detection.SourceRun] = []
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Run"))
        self.choice = QtWidgets.QComboBox()
        self.choice.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.choice.currentIndexChanged.connect(self._run_chosen)
        bar.addWidget(self.choice)
        self.details = note()
        bar.addWidget(self.details, 1)
        save = QtWidgets.QToolButton(text="Save CSV…")
        save.setToolTip("Save this run's sources as CSV, every column aligned")
        save.clicked.connect(self.save_csv)
        bar.addWidget(save)
        bar.addWidget(guide.button("sources"))
        lay.addLayout(bar)
        self.table = plain_table()
        self.table.itemSelectionChanged.connect(self._selected)
        self.table.itemDoubleClicked.connect(self._activated)
        self.table.setToolTip("Click a source to show it; double-click to use it as the centre")
        lay.addWidget(self.table, 1)

    def current(self) -> Optional[detection.SourceRun]:
        number = self.choice.currentData()
        return next((r for r in self.runs if r.number == number), None)

    def _run_chosen(self, _index: int) -> None:
        run = self.current()
        if run is None:
            return
        columns = [(c, h, f) for c, h, f in SOURCE_COLUMNS if c in run.columns]
        rows = [[run.columns[c][i] for c, _h, _f in columns] for i in range(len(run))]
        fill_table(self.table, [(h, f) for _c, h, f in columns], rows)
        for k, (_c, h, _f) in enumerate(columns):  # what each column means
            item = self.table.horizontalHeaderItem(k)
            if item is not None and h in guide.COLUMNS:
                item.setToolTip(guide.COLUMNS[h])
        self.table.setSortingEnabled(True)
        changed = [f"{k}={v:g}" for k, v in run.settings.items() if v != detection.DEFAULTS.get(k)]
        self.details.setText(
            f"{run.seconds:.1f} s · " + (" ".join(changed) if changed else "default settings")
        )
        self.run_changed.emit(run.number)

    def _selected(self) -> None:
        index = self.selected_index()
        if index is not None:
            self.source_selected.emit(self.current().number, index)

    def selected_index(self) -> Optional[int]:
        run = self.current()
        rows = self.table.selectionModel().selectedRows()
        if run is None or not rows:
            return None
        source = int(self.table.item(rows[0].row(), 0).value)  # NUMBER
        return int(np.flatnonzero(run.columns["NUMBER"] == source)[0])

    def _activated(self, _item) -> None:
        index = self.selected_index()
        if index is not None:
            self.source_activated.emit(self.current().number, index)

    def set_runs(self, runs) -> None:
        """Show the runs of the session (choosing the newest one when one was added)."""
        known = [r.number for r in self.runs]
        added = [r for r in runs if r.number not in known]
        self.runs = list(runs)
        current = self.choice.currentData()
        self.choice.blockSignals(True)
        self.choice.clear()
        for run in self.runs:
            self.choice.addItem(f"{run.number}: {run.image_name}, {len(run)} sources", run.number)
        number = added[-1].number if added else current
        index = self.choice.findData(number)
        self.choice.setCurrentIndex(index if index >= 0 else self.choice.count() - 1)
        self.choice.blockSignals(False)
        if added or current != self.choice.currentData():
            self._run_chosen(self.choice.currentIndex())

    def select(self, index: int) -> None:
        """Select the source in row ``index`` of the current run's catalogue."""
        run = self.current()
        if run is None:
            return
        source = float(run.columns["NUMBER"][index])
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).value == source:
                self.table.blockSignals(True)
                self.table.selectRow(r)
                self.table.blockSignals(False)
                self.table.scrollToItem(self.table.item(r, 0))
                return


class CommandPanel(QtWidgets.QWidget):
    """The terminal: what was run (the window's actions too) and a line to type more.
    Tab completes; Up and Down go through the history; Ctrl+C stops a program running (or
    clears the line)."""

    submitted = QtCore.Signal(str)
    interrupt = QtCore.Signal()
    link = QtCore.Signal(str)  # a click on a folder (cd:path) or a FITS file (open:path)

    def __init__(self, parent=None):
        super().__init__(parent)
        # one dark surface, like a shell: the output, then the prompt right after its last
        # line (moving down as output comes, then staying at the bottom)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.surface = QtWidgets.QFrame()
        self.surface.setObjectName("terminal")
        self.surface.setStyleSheet(
            "#terminal { background: #15171c; border: 1px solid #2a2e36; }"
            "#terminal QTextBrowser { background: #15171c; color: #d5d9e0; border: none; }"
        )
        outer.addWidget(self.surface)
        lay = QtWidgets.QVBoxLayout(self.surface)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(0)
        fixed = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.fixed = fixed
        self.output = QtWidgets.QTextBrowser()  # colours, and links in listings
        self.output.setOpenLinks(False)
        self.output.anchorClicked.connect(lambda url: self.link.emit(url.toString()))
        self.output.setFont(fixed)
        self.output.document().setMaximumBlockCount(5000)
        self.output.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.output.document().setDocumentMargin(0)
        lay.addWidget(self.output)
        self._filler = QtWidgets.QWidget()  # takes the room below the prompt
        lay.addWidget(self._filler, 1)
        self.layout_ = lay  # the command line joins it when the terminal is open
        self.output.document().contentsChanged.connect(self._fit)
        self.line = QtWidgets.QLineEdit()
        self.line.setFont(fixed)
        self.line.setPlaceholderText("a command (help lists them; Tab completes)")
        self.line.returnPressed.connect(self._submit)
        self.line.installEventFilter(self)
        # the command line is its own widget, placed below the result tabs by the window,
        # so that it stays visible when the panel is folded
        self.input_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.input_row)
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(4)
        row.addWidget(QtWidgets.QLabel("›"))
        row.addWidget(self.line, 1)
        self.history: List[str] = []
        self._position = 0
        self.complete = None  # callable(line) -> (start, candidates), set by the window
        self.completer = QtWidgets.QCompleter(self)
        self.completer.setWidget(self.line)
        self.completer.setCompletionMode(QtWidgets.QCompleter.UnfilteredPopupCompletion)
        self.completer.setCaseSensitivity(Qt.CaseSensitive)
        self.completer.activated.connect(self._insert)
        self._start = 0
        # a click in the terminal types at the prompt (set up once everything exists)
        self.output.viewport().installEventFilter(self)
        self._filler.installEventFilter(self)

    def _submit(self) -> None:
        text = self.line.text().strip()
        self.line.clear()
        if text:
            if not self.history or self.history[-1] != text:
                self.history.append(text)
            self._position = len(self.history)
            self.submitted.emit(text)

    def eventFilter(self, obj, event):
        if obj in (self.output.viewport(), self._filler):
            # a click in the terminal (not a drag that selects text) types at the prompt
            if (
                event.type() == QtCore.QEvent.MouseButtonRelease
                and not self.output.textCursor().hasSelection()
            ):
                self.line.setFocus()
            return False
        if obj is self.line and event.type() == QtCore.QEvent.KeyPress:
            # Ctrl+C (Qt reports the Control key as Meta on macOS, where ⌘C copies)
            if event.key() == Qt.Key_C and event.modifiers() & CONTROL:
                if not self.line.hasSelectedText():
                    self.line.clear()
                    self.interrupt.emit()
                    return True
            if event.key() == Qt.Key_Tab:
                self._tab()
                return True
            if event.key() in (Qt.Key_Up, Qt.Key_Down) and self.history:
                step = -1 if event.key() == Qt.Key_Up else 1
                self._position = min(max(self._position + step, 0), len(self.history))
                text = self.history[self._position] if self._position < len(self.history) else ""
                self.line.setText(text)
                return True
        return False

    def take_line(self, inside: bool) -> None:
        """The command line inside the terminal (right after the output, as in a shell) or
        out of it (the window puts it in the bar at the bottom, as a plain field)."""
        if inside:
            self.layout_.insertWidget(1, self.input_row)
            self.input_row.setStyleSheet(
                "QLineEdit { background: #15171c; color: #f2f4f8; border: none; padding: 0; }"
                "QLabel { color: #7fb2ff; }"
            )
            self.line.setFrame(False)
            self.input_row.show()
        else:
            self.layout_.removeWidget(self.input_row)
            self.input_row.setStyleSheet("")
            self.line.setFrame(True)
        self._fit()

    DEFAULT_FONT = 0.0  # 0: the system's fixed-width size

    def font_size(self) -> float:
        return self.output.font().pointSizeF()

    def set_font_size(self, points: float) -> None:
        """The terminal's text size, in points (0: the system's size), for the output and
        the command line alike."""
        font = QtGui.QFont(self.fixed)
        if points > 0:
            font.setPointSizeF(max(6.0, min(float(points), 40.0)))
        self.output.setFont(font)
        self.output.document().setDefaultFont(font)
        self.line.setFont(font)
        QtCore.QTimer.singleShot(0, self._fit)

    def _fit(self) -> None:
        """The output as tall as its text (so the prompt follows its last line), up to the
        room there is, where it scrolls and the prompt stays at the bottom."""
        if self.input_row.parent() is not self.surface:
            self.output.setMaximumHeight(16777215)
            return
        doc = self.output.document()
        doc.setTextWidth(self.output.viewport().width())
        text = int(doc.size().height()) + 4  # the real height of the output, in pixels
        room = self.surface.height() - self.input_row.sizeHint().height() - 10
        self.output.setMaximumHeight(max(min(text, room), 0))
        if text >= room:
            bar = self.output.verticalScrollBar()
            bar.setValue(bar.maximum())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _tab(self) -> None:
        """Complete the word being typed: at once when one candidate fits (or as far as
        they all agree), otherwise in a list."""
        if self.complete is None:
            return
        text = self.line.text()[: self.line.cursorPosition()]
        start, candidates = self.complete(text)
        if not candidates:
            return
        self._start = start
        word = text[start:]
        common = os.path.commonprefix(candidates)
        if len(candidates) == 1:
            end = "" if candidates[0].endswith(("/", "=")) else " "
            self._insert(candidates[0] + end)
        elif len(common) > len(word):
            self._insert(common)
        else:
            self.completer.setModel(QtCore.QStringListModel(candidates, self.completer))
            rect = self.line.cursorRect()
            rect.setWidth(max(260, self.completer.popup().sizeHintForColumn(0) + 30))
            self.completer.complete(rect)

    def _insert(self, completion: str) -> None:
        text = self.line.text()
        cursor = self.line.cursorPosition()
        self.line.setText(text[: self._start] + completion + text[cursor:])
        self.line.setCursorPosition(self._start + len(completion))
        self.line.setFocus()

    def _add(self, body: str) -> None:
        """One block of output (HTML), in the terminal's fixed-width font."""
        self.output.append(f"<div style='white-space: pre-wrap'>{body}</div>")
        self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().maximum())

    def echo(self, line: str) -> None:
        self._add(
            "<span style='color:#7fb2ff'>› </span>"
            f"<span style='color:#f2f4f8; font-weight:600'>{html.escape(line)}</span>"
        )

    def say(self, text: str, error: bool = False, rich: str = "") -> None:
        """Output: plain text, an error (red), or a rich version (colours and links)."""
        if rich:
            self._add(rich)
        elif text:
            body = html.escape(text).replace("\n", "<br>")
            self._add(f"<span style='color:#ff6b6b'>{body}</span>" if error else body)


OVERLAYS = (
    # key, label, colour, shown at the start
    ("isophotes", "Isophotes (ELLIPROF)", "#7CFC00", True),
    ("sources", "Sources (Source finder)", "#00d5ff", True),
    ("centre", "ELLIPROF centre, R0 and R1", "#ff4d4d", True),
    ("pair", "Pair fit (A and B: guides and isophotes)", "#ff4dff", True),
    ("contours", "Contours", "#ffd84a", False),
    ("crosshair", "Crosshair", "#ff9f40", False),
)


class LayersPanel(QtWidgets.QWidget):
    """The layers over the image: the overlays, then the mask layers.  Tick to show,
    click the colour box to change a colour, right-click for what can be done (new and
    loaded masks, undo and redo, and what can be done to the mask layer under the mouse).
    The mask tools draw on the mask layer selected here."""

    toggled = QtCore.Signal(str, bool)  # an overlay key, or "mask:<name>"
    mask_command = QtCore.Signal(str)  # a command for the mask layers
    colour_changed = QtCore.Signal(str, str)  # overlay key or "mask:<name>", colour
    opacity_changed = QtCore.Signal(str, float)  # mask name ("" = every mask), opacity
    active_changed = QtCore.Signal(str)  # the mask the mask tools draw on
    show_only = QtCore.Signal(str)  # show this mask layer alone
    from_buffer = QtCore.Signal()  # make a mask layer from a buffer
    tip_folded = QtCore.Signal(bool)  # the tip folded to its bulb (or unfolded)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        top = title_row("Layers", "layers")
        top.addWidget(guide.button("masks", label="? Masks"))
        lay.addLayout(top)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)  # the colour box, then the tick and the name
        self.tree.header().setStretchLastSection(True)
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.tree.setIndentation(10)
        self.tree.itemChanged.connect(self._changed)
        self.tree.currentItemChanged.connect(self._current)
        self.tree.itemClicked.connect(self._clicked)
        self.tree.setAcceptDrops(True)  # mask files dropped here become mask layers
        self.tree.viewport().setAcceptDrops(True)
        self.tree.viewport().installEventFilter(self)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.tree, 1)
        self.overlays = {}
        self.overlay_root = QtWidgets.QTreeWidgetItem(["Overlays"])
        self.overlay_root.setFlags(Qt.ItemIsEnabled)
        self.tree.addTopLevelItem(self.overlay_root)
        self.overlay_root.setFirstColumnSpanned(True)
        for key, label, colour, on in OVERLAYS:
            item = self._row(key, label, colour, on, "Tick to show")
            self.overlay_root.addChild(item)
            self.overlays[key] = item
        self.mask_root = QtWidgets.QTreeWidgetItem(["Masks"])
        self.mask_root.setFlags(Qt.ItemIsEnabled)
        self.tree.addTopLevelItem(self.mask_root)
        self.mask_root.setFirstColumnSpanned(True)
        self.overlay_root.setExpanded(True)
        self.mask_root.setExpanded(True)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Opacity"))
        self.opacity = QtWidgets.QSlider(Qt.Horizontal)
        self.opacity.setRange(5, 100)
        self.opacity.setValue(45)
        self.opacity.setToolTip("the opacity of the selected mask layer")
        self.opacity.valueChanged.connect(self._opacity)
        row.addWidget(self.opacity, 1)
        lay.addLayout(row)
        self.active = note()
        lay.addWidget(self.active)
        # how to unmask, as a tip with a yellow bulb
        from . import icons

        self.tip_box = QtWidgets.QFrame()
        self.tip_box.setObjectName("tip")
        self.tip_box.setStyleSheet(
            "#tip { background: rgba(255, 210, 63, 0.12); border: 1px solid "
            "rgba(255, 210, 63, 0.45); border-radius: 6px; }"
        )
        tip_row = QtWidgets.QHBoxLayout(self.tip_box)
        tip_row.setContentsMargins(4, 3, 6, 3)
        tip_row.setSpacing(6)
        # a click on the bulb folds the tip to the bulb alone, and unfolds it
        self.bulb = QtWidgets.QToolButton()
        self.bulb.setIcon(icons.icon("bulb", 14))
        self.bulb.setIconSize(QtCore.QSize(14, 14))
        self.bulb.setAutoRaise(True)
        self.bulb.setStyleSheet("QToolButton { border: none; padding: 0; }")
        self.bulb.setCursor(Qt.PointingHandCursor)
        self.bulb.clicked.connect(lambda: self.set_tip_folded(not self.tip_text.isHidden()))
        tip_row.addWidget(self.bulb, 0, Qt.AlignTop)
        self.tip_text = QtWidgets.QLabel(
            "<b>Alt</b> or <b>Shift</b> while drawing unmasks (or turn on <i>Unmask</i> "
            "in the tool bar). <b>Ctrl-click</b> (⌘-click on a Mac) removes a masked patch."
        )
        self.tip_text.setWordWrap(True)
        tip_row.addWidget(self.tip_text, 1)
        lay.addWidget(self.tip_box)
        self._active: Optional[str] = None
        self._opacities = {}

    def set_tip_folded(self, folded: bool) -> None:
        """Fold the tip to its bulb (a click on the bulb), or unfold it."""
        self.tip_text.setHidden(folded)
        self.bulb.setToolTip("Show the tip" if folded else "Hide the tip")
        # folded: only as wide as the bulb, at the left; unfolded: the panel's width
        self.layout().setAlignment(self.tip_box, Qt.AlignLeft if folded else Qt.Alignment())
        self.tip_folded.emit(folded)

    def set_masks(self, masks, active: Optional[str]) -> None:
        self.tree.blockSignals(True)
        self.mask_root.takeChildren()
        self._active = active
        for m in masks:
            item = self._row(
                f"mask:{m.name}",
                f"{m.name}   ({m.count():,} px)",
                m.colour,
                m.visible,
                f"{m.source or 'drawn'}; right-click for more",
            )
            self._opacities[m.name] = m.opacity
            if m.name == active:
                font = item.font(1)
                font.setBold(True)
                item.setFont(1, font)
            self.mask_root.addChild(item)
            if m.name == active:
                self.tree.setCurrentItem(item)
        self.tree.blockSignals(False)
        self.tree.resizeColumnToContents(0)
        if active:
            self.opacity.blockSignals(True)
            self.opacity.setValue(int(round(self._opacities.get(active, 0.45) * 100)))
            self.opacity.blockSignals(False)
        self.active.setText(
            f"Mask tools draw on <b>{active}</b>."
            if active
            else "Right-click here to make or load a mask (or drop its file here), or draw "
            "one: the mask tools make a layer when there is none."
        )

    @staticmethod
    def _row(key, label, colour, on, tip) -> QtWidgets.QTreeWidgetItem:
        """A layer: its colour box (click to change the colour), its tick and its name."""
        item = QtWidgets.QTreeWidgetItem(["", label])
        item.setData(0, Qt.UserRole, key)
        item.setIcon(0, swatch(colour, 14))
        item.setToolTip(0, "Click to change the colour")
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(1, Qt.Checked if on else Qt.Unchecked)
        item.setToolTip(1, tip)
        return item

    def checked(self, key: str) -> bool:
        return self.overlays[key].checkState(1) == Qt.Checked

    def set_checked(self, key: str, on: bool) -> None:
        self.overlays[key].setCheckState(1, Qt.Checked if on else Qt.Unchecked)

    def _changed(self, item, column) -> None:
        key = item.data(0, Qt.UserRole)
        if key and column == 1:
            self.toggled.emit(key, item.checkState(1) == Qt.Checked)

    def _current(self, item, _previous) -> None:
        key = item.data(0, Qt.UserRole) if item else None
        if key and key.startswith("mask:"):
            self.active_changed.emit(key[5:])

    def _clicked(self, item, column) -> None:
        key = item.data(0, Qt.UserRole)
        if key and column == 0:  # the colour box
            self._colour(key)

    def _opacity(self, value: int) -> None:
        if self._active:
            self.opacity_changed.emit(self._active, value / 100)

    def _on_current(self, action: str) -> None:
        if not self._active:
            return
        if action == "copy":
            name, ok = QtWidgets.QInputDialog.getText(
                self, "Copy the mask", "Name of the copy:", text=f"{self._active}_copy"
            )
            if ok and name.strip():
                self.mask_command.emit(f"mask copy {name.strip().replace(' ', '_')} {self._active}")
        else:
            self.mask_command.emit(f"mask {action} {self._active}")

    def _new(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "New mask", "Name of the mask layer:")
        name = name.strip().replace(" ", "_")
        if ok and name:
            self.mask_command.emit(f"mask new {name}")

    def eventFilter(self, obj, event):
        """Files dropped on the list are loaded as mask layers."""
        kind = event.type()
        if kind in (QtCore.QEvent.DragEnter, QtCore.QEvent.DragMove):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return True
        elif kind == QtCore.QEvent.Drop and event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    self.load_file(url.toLocalFile())
            event.acceptProposedAction()
            return True
        return super().eventFilter(obj, event)

    def load_file(self, path: str) -> None:
        name = re.sub(r"\W", "_", Path(path).name.split(".")[0]) or "mask"
        self.mask_command.emit(f"mask load {name} {shlex.quote(path)}")

    def _load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load a mask", "", "All files (*);;Masks (*.dmask *.fits *.fit *.fts *.gz)"
        )
        if path:
            self.load_file(path)

    def _menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        key = item.data(0, Qt.UserRole) if item else None
        if not key:  # the Masks heading, or the empty part of the list
            if item is None or item is self.mask_root:
                menu = QtWidgets.QMenu(self)
                self._general_actions(menu)
                menu.exec(self.tree.viewport().mapToGlobal(pos))
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Colour…", lambda: self._colour(key))
        if key.startswith("mask:"):
            name = key[5:]
            opacity = menu.addMenu("Opacity")
            for percent in (15, 30, 45, 60, 80, 100):
                opacity.addAction(
                    f"{percent}%", lambda p=percent: self.opacity_changed.emit(name, p / 100)
                )
            menu.addAction("Show only this one", lambda: self.show_only.emit(name))
            menu.addSeparator()
            menu.addAction("Grow…", lambda: self._radius(name, "grow"))
            menu.addAction("Shrink…", lambda: self._radius(name, "shrink"))
            for action in ("invert", "clear", "fill"):
                menu.addAction(
                    action.capitalize(), lambda a=action: self.mask_command.emit(f"mask {a} {name}")
                )
            menu.addSeparator()
            menu.addAction("Rename…", lambda: self._rename(name))
            menu.addAction("Copy…", lambda: (self._set_active(name), self._on_current("copy")))
            menu.addAction("Save as…", lambda: self._save(name))
            menu.addAction("To a buffer", lambda: self.mask_command.emit(f"mask tobuf {name} new"))
            menu.addSeparator()
            menu.addAction("Delete", lambda: self.mask_command.emit(f"mask delete {name}"))
            menu.addSeparator()
            self._general_actions(menu)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _general_actions(self, menu: QtWidgets.QMenu) -> None:
        """What a right-click offers anywhere on the mask layers: new and loaded masks, and
        undo and redo of the mask edits."""
        menu.addAction("New mask…", self._new)
        load = menu.addAction("Load mask…", self._load)
        load.setToolTip("A FITS or .dmask file (any name); files can also be dropped here")
        menu.addAction("Mask from a buffer…", self.from_buffer.emit)
        menu.addSeparator()
        menu.addAction("Undo the last mask edit", lambda: self.mask_command.emit("undo"))
        menu.addAction("Redo", lambda: self.mask_command.emit("redo"))

    def _set_active(self, name: str) -> None:
        self._active = name

    def _radius(self, name: str, action: str) -> None:
        r, ok = QtWidgets.QInputDialog.getDouble(
            self, f"{action.capitalize()} {name}", "By how many pixels:", 3.0, 0.5, 500.0, 1
        )
        if ok:
            self.mask_command.emit(f"mask {action} {name} {r:g}")

    def _rename(self, name: str) -> None:
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename", "New name:", text=name)
        new = new.strip().replace(" ", "_")
        if ok and new and new != name:
            self.mask_command.emit(f"mask rename {name} {new}")

    def _colour(self, key: str) -> None:
        colour = QtWidgets.QColorDialog.getColor(parent=self, title="Colour")
        if colour.isValid():
            self.colour_changed.emit(key, colour.name())
            item = self.overlays.get(key)
            if item is not None:
                item.setIcon(0, swatch(colour.name(), 14))

    def _save(self, name: str) -> None:
        path, kind = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the mask", f"{name}_mask.fits", "FITS (*.fits);;Legacy bitmap (*.dmask)"
        )
        if path:
            self.mask_command.emit(f"mask save {name} {shlex.quote(mask_file_name(path, kind))}")


def mask_file_name(path: str, kind: str) -> str:
    """The file name for a mask saved with the type chosen in the dialog: the type decides
    the ending (the ending decides the format), so choosing "Legacy bitmap" writes a
    .dmask even when the name typed ends in .fits, and the other way round."""
    stem, dot, ending = path.rpartition(".")
    if not dot or "/" in ending:
        stem, ending = path, ""
    if "dmask" in kind and ending.lower() != "dmask":
        return (stem if ending.lower() in ("fits", "fit", "fts") else path) + ".dmask"
    if "FITS" in kind and ending.lower() == "dmask":
        return stem + ".fits"
    return path


class DisplayPanel(QtWidgets.QWidget):
    """Display settings beyond the viewer's bar: orientation, contrast and bias."""

    orientation = QtCore.Signal(float, bool)  # rotation, mirror
    north_up = QtCore.Signal()
    contrast_bias = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.addRow(title_row("Display", "display", "display"))
        form.addRow(note("Display only: none of this changes pixel values."))

        form.addRow(heading("Orientation"))
        grid = QtWidgets.QGridLayout()
        self.rotation, self.mirror = 0.0, False
        for k, (text, slot) in enumerate(
            (
                ("North up", self.north_up.emit),
                ("Rotate ⟲ 90°", lambda: self._turn(90)),
                ("Rotate ⟳ 90°", lambda: self._turn(-90)),
                ("Mirror", self._flip),
                ("Reset", lambda: self._set(0.0, False)),
            )
        ):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            grid.addWidget(b, k // 3, k % 3)
        form.addRow(tip(grid, "orientation"))
        self.angle = number(-360, 360, 0, decimals=1, step=5)
        self.angle.setToolTip("rotation of the display, degrees counter-clockwise")
        self.angle.editingFinished.connect(lambda: self._set(self.angle.value(), self.mirror))
        form.addRow("Angle", self.angle)

        form.addRow(heading("Contrast and bias"))
        self.contrast = QtWidgets.QSlider(Qt.Horizontal)
        self.contrast.setRange(5, 500)  # contrast x 100 on a log-like scale
        self.contrast.setValue(100)
        self.bias = QtWidgets.QSlider(Qt.Horizontal)
        self.bias.setRange(0, 100)
        self.bias.setValue(50)
        for slider in (self.contrast, self.bias):
            slider.valueChanged.connect(self._contrast_bias)
        form.addRow("Contrast", tip(self.contrast, "contrast"))
        form.addRow("Bias", self.bias)
        form.addRow(note("Or drag on the image with the right mouse button."))

    def _turn(self, degrees: float) -> None:
        self._set(self.rotation + degrees, self.mirror)

    def _flip(self) -> None:
        self._set(self.rotation, not self.mirror)

    def _set(self, rotation: float, mirror: bool) -> None:
        self.rotation, self.mirror = (rotation + 180.0) % 360.0 - 180.0, mirror
        self.angle.blockSignals(True)
        self.angle.setValue(self.rotation)
        self.angle.blockSignals(False)
        self.orientation.emit(self.rotation, self.mirror)

    def show_orientation(self, rotation: float, mirror: bool) -> None:
        self.rotation, self.mirror = (rotation + 180.0) % 360.0 - 180.0, mirror
        self.angle.blockSignals(True)
        self.angle.setValue(self.rotation)
        self.angle.blockSignals(False)

    def _contrast_bias(self) -> None:
        self.contrast_bias.emit(self.contrast.value() / 100.0, self.bias.value() / 100.0)

    def show_contrast_bias(self, contrast: float, bias: float) -> None:
        for slider, value in ((self.contrast, contrast * 100), (self.bias, bias * 100)):
            slider.blockSignals(True)
            slider.setValue(int(round(value)))
            slider.blockSignals(False)


class ContourPanel(QtWidgets.QWidget):
    """Contours, with the parameters of DS9's contour dialog: the number of levels and the
    smoothness, the smoothing method, the scale and limits the levels are spread over, the
    levels themselves (generated, then editable), and the line style."""

    apply = QtCore.Signal()
    clear = QtCore.Signal()
    style_changed = QtCore.Signal(str, float, bool)  # colour, width, dashed
    from_display = QtCore.Signal()  # copy the display's scale and limits

    def __init__(self, parent=None):
        super().__init__(parent)
        from . import contours as ct
        from . import scales

        form = QtWidgets.QFormLayout(self)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.addRow(title_row("Contours", "contours", "contours"))
        form.addRow(note("Lines of equal brightness over the image; display only."))

        form.addRow(heading("Levels"))
        self.count = whole(1, 100, 6, tip="the number of contour levels")
        form.addRow("Levels", tip(self.count, "c_levels"))
        self.scale = QtWidgets.QComboBox()
        self.scale.addItems(list(scales.STRETCHES))
        self.scale.setToolTip(
            "how the levels are spread between the limits (Log: more at the faint end)"
        )
        form.addRow("Scale", tip(self.scale, "c_scale"))
        self.exponent = whole(2, 100000, 1000, tip="the constant of the Log and Power scales")
        form.addRow("Exponent", tip(self.exponent, "c_scale"))
        self.limit_mode = QtWidgets.QComboBox()
        self.limit_mode.addItems([m for m in scales.LIMITS if m != "Manual"] + ["User"])
        self.limit_mode.setToolTip("the faintest and brightest levels")
        form.addRow("Limits", tip(self.limit_mode, "c_limits"))
        row = QtWidgets.QHBoxLayout()
        self.low = QtWidgets.QLineEdit()
        self.high = QtWidgets.QLineEdit()
        for box, words in ((self.low, "the lowest level"), (self.high, "the highest level")):
            box.setToolTip(words)
            box.textEdited.connect(lambda _t: self.limit_mode.setCurrentText("User"))
            row.addWidget(box)
        form.addRow("Low, high", tip(row, "c_limits"))
        same = QtWidgets.QPushButton("Same as the display")
        same.setToolTip("Take the scale and the limits of the image's display")
        same.clicked.connect(self.from_display)
        generate = QtWidgets.QPushButton("Generate")
        generate.setToolTip("Fill the list of levels from the settings above")
        generate.clicked.connect(self.generate_requested)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(same)
        row.addWidget(generate)
        form.addRow(row)
        self.levels = QtWidgets.QPlainTextEdit()
        self.levels.setPlaceholderText("one level per line (Generate fills it; edit freely)")
        self.levels.setMaximumHeight(120)
        self.levels.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        form.addRow("Values", tip(self.levels, "c_values"))

        form.addRow(heading("Smoothing"))
        self.method = QtWidgets.QComboBox()
        self.method.addItems(list(ct.METHODS))
        self.method.setCurrentText("Smooth")
        self.method.setToolTip("Block: average blocks of pixels; Smooth: a Gaussian")
        form.addRow("Method", tip(self.method, "c_smoothing"))
        self.smoothness = whole(1, 100, 3, tip="the block size, or the Gaussian's sigma, in pixels")
        form.addRow("Smoothness", tip(self.smoothness, "c_smoothing"))

        form.addRow(heading("Lines"))
        self.colour = "#ffd84a"
        self.colour_button = QtWidgets.QPushButton()
        self.colour_button.clicked.connect(self._choose_colour)
        self._show_colour()
        form.addRow("Colour", tip(self.colour_button, "c_lines"))
        self.width = number(0.5, 6, 1.0, decimals=1, step=0.5, tip="line width, screen pixels")
        self.dashed = QtWidgets.QCheckBox("Dashed")
        for w in (self.width, self.dashed):
            (w.valueChanged if w is self.width else w.toggled).connect(lambda *_: self._style())
        form.addRow("Width", tip(self.width, "c_lines"))
        form.addRow(self.dashed)

        row = QtWidgets.QHBoxLayout()
        go = QtWidgets.QPushButton("Apply")
        go.setDefault(True)
        go.clicked.connect(self.apply)
        off = QtWidgets.QPushButton("Clear")
        off.clicked.connect(self.clear)
        row.addWidget(go)
        row.addWidget(off)
        form.addRow(row)
        self.status = note()
        form.addRow(self.status)
        self.image = None  # the data levels are generated from (set by the window)

    def _choose_colour(self) -> None:
        colour = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.colour), self, "Contour colour")
        if colour.isValid():
            self.colour = colour.name()
            self._show_colour()
            self._style()

    def _show_colour(self) -> None:
        self.colour_button.setIcon(swatch(self.colour, 14))
        self.colour_button.setText(self.colour)

    def _style(self) -> None:
        self.style_changed.emit(self.colour, self.width.value(), self.dashed.isChecked())

    def set_limits(self, lo: float, hi: float) -> None:
        self.low.setText(f"{lo:.6g}")
        self.high.setText(f"{hi:.6g}")

    def generate_requested(self) -> None:
        """The levels from the settings, for the image shown."""
        from . import contours as ct
        from . import scales

        if self.image is None:
            return
        mode = self.limit_mode.currentText()
        if mode == "User":
            try:
                lo, hi = float(self.low.text()), float(self.high.text())
            except ValueError:
                self.status.setText("Type the low and high limits (numbers).")
                return
        else:
            lo, hi = scales.limits(self.image.data, mode)
            self.set_limits(lo, hi)
        if self.scale.currentText() == "Histogram":
            levels = ct.histogram_levels(self.image.data, lo, hi, self.count.value())
        else:
            levels = ct.generate_levels(
                lo, hi, self.count.value(), self.scale.currentText(), self.exponent.value()
            )
        self.levels.setPlainText("\n".join(f"{v:.6g}" for v in levels))

    def values(self):
        """The levels in the list (raises ValueError for what is not a number)."""
        words = self.levels.toPlainText().replace(",", " ").split()
        return sorted(float(w) for w in words)


class QuickInspector(QtWidgets.QWidget):
    """The pixel under the cursor (or the crosshair), in brief: where it is, its value,
    its sky position, a magnified view of its surroundings and their statistics."""

    BOX = 15  # pixels on a side of the magnified view
    STATS = 5  # pixels on a side of the statistics box

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.addRow(title_row("Inspector", "inspect"))
        form.addRow(note("Move the mouse over the image (or use the crosshair)."))
        fixed = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.fields = {}
        for key, label in (
            ("buffer", "Buffer"),
            ("xy", "x, y"),
            ("pixel", "Pixel"),
            ("value", "Value"),
            ("ra", "RA"),
            ("dec", "Dec"),
            ("degrees", "RA, Dec (°)"),
            ("mean", "Mean"),
            ("median", "Median"),
            ("sigma", "σ"),
        ):
            field = QtWidgets.QLabel("–")
            field.setFont(fixed)
            field.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.fields[key] = field
            form.addRow(label, field)
            if key == "degrees":
                self.magnifier = QtWidgets.QLabel()
                self.magnifier.setFixedSize(165, 165)
                self.magnifier.setStyleSheet("background: #111; border: 1px solid #444")
                self.magnifier.setToolTip(f"{self.BOX} × {self.BOX} pixels around the cursor")
                form.addRow("Around", self.magnifier)
                form.addRow(note(f"Statistics of the {self.STATS} × {self.STATS} pixels around:"))

    def show_at(self, viewer, number, x: float, y: float) -> None:
        image = viewer.image
        if image is None:
            return
        f = self.fields
        ny, nx = image.shape
        i, j = int(np.floor(x)), int(np.floor(y))
        f["buffer"].setText(f"{number}  {image.name}")
        f["xy"].setText(f"{x:.2f}, {y:.2f}")
        inside = 0 <= i < nx and 0 <= j < ny
        f["pixel"].setText(f"{i + 1}, {j + 1}" if inside else "outside the image")
        f["value"].setText(f"{image.data[j, i]:.6g}" if inside else "–")
        w = image.wcs()
        if w is not None:
            sky = w.pixel_to_world(x - 0.5, y - 0.5)
            f["ra"].setText(sky.ra.to_string(unit="hourangle", sep=":", precision=2, pad=True))
            f["dec"].setText(sky.dec.to_string(sep=":", precision=1, alwayssign=True, pad=True))
            f["degrees"].setText(f"{sky.ra.deg:.6f}, {sky.dec.deg:+.6f}")
        else:
            for key in ("ra", "dec", "degrees"):
                f[key].setText("no WCS")
        h = self.STATS // 2
        box = image.data[max(j - h, 0) : j + h + 1, max(i - h, 0) : i + h + 1]
        values = box[np.isfinite(box)] if inside else np.array([])
        if values.size:
            f["mean"].setText(f"{values.mean():.6g}")
            f["median"].setText(f"{np.median(values):.6g}")
            f["sigma"].setText(f"{values.std():.4g}")
        else:
            for key in ("mean", "median", "sigma"):
                f[key].setText("–")
        self._magnify(viewer, i, j)

    def _magnify(self, viewer, i: int, j: int) -> None:
        """The pixels around (i, j) in the display's colours, the centre one outlined."""
        scaled = viewer.item.image
        table = viewer.item.lut
        if scaled is None or table is None:
            return
        ny, nx = scaled.shape[:2]
        h = self.BOX // 2
        patch = np.zeros((self.BOX, self.BOX), np.float32)
        j0, j1, i0, i1 = max(j - h, 0), min(j + h + 1, ny), max(i - h, 0), min(i + h + 1, nx)
        if j0 < j1 and i0 < i1:
            patch[j0 - (j - h) : j1 - (j - h), i0 - (i - h) : i1 - (i - h)] = scaled[j0:j1, i0:i1]
        index = np.clip((patch * 255).astype(int), 0, 255)
        rgb = np.ascontiguousarray(np.asarray(table)[index][::-1, :, :3], dtype=np.uint8)
        qimage = QtGui.QImage(
            rgb.data, self.BOX, self.BOX, 3 * self.BOX, QtGui.QImage.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(qimage.copy()).scaled(
            165, 165, Qt.IgnoreAspectRatio, Qt.FastTransformation
        )
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffd200"), 2))
        cell = 165 / self.BOX
        painter.drawRect(QtCore.QRectF(h * cell, h * cell, cell, cell))
        painter.end()
        self.magnifier.setPixmap(pixmap)


class HeaderPanel(QtWidgets.QPlainTextEdit):
    """The FITS header of the buffer shown."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))

    def show_header(self, image) -> None:
        self.setPlainText(image.header.tostring(sep="\n").rstrip() if image is not None else "")


class PlotsPanel(QtWidgets.QWidget):
    """A scatter plot of two columns of the current source-finder run; clicking a point
    selects that source."""

    source_selected = QtCore.Signal(int)  # row of the run's catalogue
    save_png = QtCore.Signal()  # "Save PNG…": the plot as a figure

    def __init__(self, parent=None):
        super().__init__(parent)
        self.run = None
        self.marked_row: Optional[int] = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QtWidgets.QHBoxLayout()
        self.x = QtWidgets.QComboBox()
        self.y = QtWidgets.QComboBox()
        bar.addWidget(QtWidgets.QLabel("x"))
        bar.addWidget(self.x)
        bar.addWidget(QtWidgets.QLabel("y"))
        bar.addWidget(self.y)
        bar.addStretch(1)
        save = QtWidgets.QToolButton(text="Save PNG…")
        save.setToolTip("Save the plot as a PNG figure (made with matplotlib)")
        save.clicked.connect(self.save_png)
        bar.addWidget(save)
        lay.addLayout(bar)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.points = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(0, 213, 255, 150))
        self.points.sigClicked.connect(self._clicked)
        self.marked = pg.ScatterPlotItem(size=12, pen=pg.mkPen("#ffd200", width=2), brush=None)
        self.plot.addItem(self.points)
        self.plot.addItem(self.marked)
        lay.addWidget(self.plot, 1)
        self.x.currentTextChanged.connect(lambda _t: self.draw())
        self.y.currentTextChanged.connect(lambda _t: self.draw())

    def set_run(self, run) -> None:
        self.run = run
        names = [c for c in (run.columns if run else {}) if c != "NUMBER"]
        for box, default in ((self.x, "MAG_AUTO"), (self.y, "FLUX_RADIUS")):
            current = box.currentText() or default
            box.blockSignals(True)
            box.clear()
            box.addItems(names)
            box.setCurrentText(current if current in names else default)
            box.blockSignals(False)
        self.draw()

    def draw(self) -> None:
        self.marked.setData([], [])
        self.marked_row = None
        if self.run is None or not self.x.currentText() or not self.y.currentText():
            self.points.setData([], [])
            return
        x = np.asarray(self.run.columns[self.x.currentText()], float)
        y = np.asarray(self.run.columns[self.y.currentText()], float)
        ok = np.isfinite(x) & np.isfinite(y)
        self.points.setData(x[ok], y[ok], data=np.flatnonzero(ok))
        self.plot.setLabel("bottom", self.x.currentText())
        self.plot.setLabel("left", self.y.currentText())

    def mark(self, row: int) -> None:
        if self.run is None:
            return
        x = float(self.run.columns[self.x.currentText()][row])
        y = float(self.run.columns[self.y.currentText()][row])
        self.marked.setData([x], [y])
        self.marked_row = row

    def write_png(self, path, title: str = "") -> None:
        """The plot as shown (the two columns, the selected source circled) as a PNG."""
        from .exports import scatter_png

        xname, yname = self.x.currentText(), self.y.currentText()
        scatter_png(
            path,
            self.run.columns[xname],
            self.run.columns[yname],
            xname,
            yname,
            self.marked_row,
            title,
        )

    def _clicked(self, _item, points, *_args) -> None:
        if len(points):
            row = int(points[0].data())
            self.mark(row)
            self.source_selected.emit(row)


def swatch(colour: str, size: int = 12) -> QtGui.QIcon:
    """A colour box with a thin outline (visible on light and dark backgrounds)."""
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtGui.QColor(colour))
    painter = QtGui.QPainter(pixmap)
    painter.setPen(QtGui.QColor(128, 128, 128))
    painter.drawRect(0, 0, size - 1, size - 1)
    painter.end()
    return QtGui.QIcon(pixmap)
