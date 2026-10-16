"""The sky section of a fitting panel (the Pair fit's, and ELLIPROF's): the sky measured by
segmentation (its settings, the verdict, the histogram of the sky pixels, the map of the
cells and the gradient), the base level used or typed, the sky factor and its refinement,
and after a fit the residual on the sky pixels with the next factor.  A panel mixes in
SkySection, calls build_sky and build_residual_sky in its form, and has ticked_layers()."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from .panels import heading, note, number, tip, whole

STATE_COLOURS = {
    "usable_support": "#3fb950",
    "candidate_inspect_support": "#d29922",
    "no_estimate": "#f85149",
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


class SkySection:
    def build_sky(
        self, form, manual, measure_signal, mask_signal, title: str = "Sky", fit_word: str = "fit"
    ) -> None:
        form.addRow(heading(title))
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
        measure.clicked.connect(measure_signal)
        self.show_mask = QtWidgets.QCheckBox("Show the sky mask")
        self.show_mask.setToolTip("Draw the pixels left out of the sky (the layer skymask)")
        self.show_mask.toggled.connect(mask_signal)
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
        self.use_sky_button = QtWidgets.QPushButton("Use this sky")
        self.use_sky_button.setEnabled(False)
        self.use_sky_button.setToolTip(
            "The median of the sky pixels as the base level (a new measurement is used at "
            "once; this takes it back after a typed value); the verdict is advice and is "
            "kept with the fit"
        )
        self.use_sky_button.clicked.connect(self._use_measured)
        enter = self.enter_sky_button = QtWidgets.QPushButton("Enter a value…")
        enter.setToolTip("Type a sky level, with a note on where it comes from")
        enter.clicked.connect(self._enter_sky)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.use_sky_button)
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
        manual("Sky factor", tip(self.factor, "pair_skyfactor"))
        self.sky_used = note()
        manual(self.sky_used)
        self.sky_iter = whole(
            0, 50, 3, tip="SKYITER: refit, each time with the factor from the residual sky"
        )
        self.sky_iter.setSpecialValueText("off")
        manual("Refine the factor", tip(self.sky_iter, "pair_skyfactor"))
        manual(
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
        self._taken = set()  # the measurements already taken as the sky

    def build_residual_sky(
        self, form, title: str = "Residual sky", fit_word: str = "fit again", what: str = ""
    ) -> None:
        what = what or (
            "image − sky − model on the sky pixels: its median in red, zero dashed. The model "
            "extends beyond R1, so its outer light counts."
        )
        form.addRow(heading(title))
        self.resid_card = note("after a fit: the residual on the sky pixels")
        self.resid_card.setTextFormat(Qt.RichText)
        form.addRow(self.resid_card)
        self.resid_hist = SkyHistogram()
        self.resid_hist.setToolTip(what)
        form.addRow(self.resid_hist)
        self.next_factor = QtWidgets.QPushButton("Use the next factor")
        self.next_factor.setEnabled(False)
        self.next_factor.setToolTip(f"Set the factor to factor + median / base; then {fit_word}")
        self.next_factor.clicked.connect(self._use_next_factor)
        form.addRow(self.next_factor)
        self._next = None

    def _fold_sky(self, on: bool) -> None:
        for w in self.sky_more.findChildren(QtWidgets.QWidget):
            w.setVisible(on)

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
            self.use_sky_button.setEnabled(False)
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
        self.use_sky_button.setEnabled(bool(np.isfinite(result.level)) and not stale)
        # a new measurement is the sky used, until another is typed or measured
        if result.id not in self._taken and not stale and np.isfinite(result.level):
            self._taken.add(result.id)
            self._use_measured()

    def _use_measured(self) -> None:
        r = self._measurement
        if r is None:
            return
        self.sky.setValue(r.level)
        self.factor.setValue(1.0)
        self.sky_id = r.id
        if hasattr(self, "use_sky"):  # ELLIPROF: subtract it
            self.use_sky.setChecked(True)
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
