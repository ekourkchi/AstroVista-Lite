"""The Sky plane tool: the sky fitted as a plane (a level and a gradient), and the image with
the plane's tilt taken out, ready for ELLIPROF and the pair fit.

The panel only builds the command line (``skyplane N ...``); the window runs it.
"""

from __future__ import annotations

import re
from typing import Optional

from PySide6 import QtCore, QtWidgets

from .pairpanel import STATE_COLOURS, SkyMap
from .panels import BufferChoice, heading, note, number, output_row, tip, title_row, whole
from .sky import DEFAULTS, PLANE_DEFAULTS


class SkyPlanePanel(QtWidgets.QWidget):
    run_requested = QtCore.Signal()
    mask_toggled = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.addRow(title_row("Sky plane", "skyplane", "skyplane"))
        form.addRow(
            note(
                "Fits the sky as a plane: its level and how it changes across the frame. The "
                "sources are found on the image minus the plane, and the plane is fitted again "
                "to what is left, until it settles. Use it when one side of the frame is "
                "brighter than the other."
            )
        )
        form.addRow(heading("Masks"))
        self.masks = QtWidgets.QLineEdit()
        self.masks.setPlaceholderText("none")
        self.masks.setToolTip(
            "Mask layers whose pixels are left out of the sky as well (names, comma "
            "separated): bright objects the detection misses, bad regions"
        )
        form.addRow("Leave out", tip(self.masks, "MASK"))
        self.advanced = QtWidgets.QGroupBox("Detection")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        self.advanced.setToolTip("The defaults suit most images: tick to change them")
        inner = QtWidgets.QFormLayout(self.advanced)
        self.thresh = number(0.1, 50, DEFAULTS["THRESH"], step=0.1)
        self.thresh.setToolTip("Sources are found down to this many sigma above the sky")
        self.grow = number(0, 50, DEFAULTS["GROW"], step=1)
        self.grow.setToolTip("Pixels added around every source")
        self.box = whole(4, 512, PLANE_DEFAULTS["BOX"])
        self.box.setToolTip(
            "The plane is fitted to the medians of boxes this many pixels wide; smaller "
            "boxes reach closer to the galaxies, larger ones are steadier"
        )
        inner.addRow("Threshold (σ)", self.thresh)
        inner.addRow("Grow (px)", self.grow)
        inner.addRow("Box (px)", self.box)
        for w in (self.thresh, self.grow, self.box):
            w.setEnabled(False)
            self.advanced.toggled.connect(w.setEnabled)
        form.addRow(self.advanced)
        form.addRow(heading("Results"))
        self.add_flat = QtWidgets.QCheckBox("Add the flat image (the tilt taken out)")
        self.add_flat.setChecked(True)
        self.add_flat.setToolTip(
            "The image minus the plane's tilt, its level kept: fit this one. Its sky is "
            "measured too, so the Pair fit and ELLIPROF can use it as a constant"
        )
        self.flat_buffer = BufferChoice("flat image")
        form.addRow(output_row(self.add_flat, self.flat_buffer))
        self.add_plane = QtWidgets.QCheckBox("Add the plane")
        self.plane_buffer = BufferChoice("plane")
        form.addRow(output_row(self.add_plane, self.plane_buffer))
        self.add_sub = QtWidgets.QCheckBox("Add the image minus the plane")
        self.sub_buffer = BufferChoice("image minus the plane")
        form.addRow(output_row(self.add_sub, self.sub_buffer))
        self.run = QtWidgets.QPushButton("Fit the sky plane")
        self.run.setDefault(True)
        self.run.clicked.connect(self.run_requested)
        form.addRow(self.run)
        self.status = note()
        form.addRow(self.status)
        self.card = QtWidgets.QLabel()
        self.card.setWordWrap(True)
        self.card.setTextFormat(QtCore.Qt.RichText)
        self.card.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        form.addRow(self.card)
        self.map_title = note("What is left after the plane (the flat image's sky, by cells):")
        form.addRow(self.map_title)
        self.map = SkyMap()
        form.addRow(self.map)
        self.show_mask = QtWidgets.QCheckBox("Show the pixels left out of the sky")
        self.show_mask.toggled.connect(self.mask_toggled)
        form.addRow(self.show_mask)
        self.show_fit(None)

    def choices(self):
        return (self.flat_buffer, self.plane_buffer, self.sub_buffer)

    def command(self, n: int, outputs: dict) -> str:
        """The skyplane line for buffer ``n``; ``outputs`` maps TOFLAT, TOPLANE and TOSUB to
        buffers."""
        parts = [f"skyplane {n}"]
        if self.advanced.isChecked():
            for key, box, default in (
                ("THRESH", self.thresh, DEFAULTS["THRESH"]),
                ("GROW", self.grow, DEFAULTS["GROW"]),
                ("BOX", self.box, PLANE_DEFAULTS["BOX"]),
            ):
                if box.value() != default:
                    parts.append(f"{key}={box.value():g}")
        masks = ",".join(m for m in re.split(r"[,\s]+", self.masks.text()) if m)
        if masks:
            parts.append(f"MASK={masks}")
        for key, value in outputs.items():
            parts.append(f"{key}={value}")
        return " ".join(parts)

    def wanted(self) -> dict:
        """The results asked for: keyword -> the buffer chosen (None: Auto)."""
        out = {}
        for key, box, choice in (
            ("TOFLAT", self.add_flat, self.flat_buffer),
            ("TOPLANE", self.add_plane, self.plane_buffer),
            ("TOSUB", self.add_sub, self.sub_buffer),
        ):
            if box.isChecked():
                out[key] = choice.value()
        return out

    def show_fit(self, fit, stale: Optional[str] = None) -> None:
        """Show a plane fit (None: none yet)."""
        self.map.set_result(fit.sky if fit is not None and fit.plane is not None else None)
        self.map_title.setVisible(fit is not None and fit.plane is not None)
        if fit is None:
            self.card.setText("")
            return
        if fit.plane is None:
            self.card.setText(
                "<b style='color:#f85149'>no plane</b><br>" + "<br>".join(fit.reasons)
            )
            return
        p, e, g, s = fit.plane, fit.errors, fit.gradient(), fit.sky
        if fit.significant and fit.scatter <= 3:
            verdict = ("#3fb950", "a significant gradient: fit the flat image")
        elif fit.significant:
            verdict = ("#d29922", "a gradient by its errors, but the sky is not a plane here")
        else:
            verdict = ("#8b919c", "no significant gradient: a constant sky is enough")
        lines = [f"<b style='color:{verdict[0]}'>{verdict[1]}</b>"]
        if stale:
            lines.append(f"<span style='color:#f85149'>stale: {stale}</span>")
        lines += [
            f"<b>level {p['s0']:.6g} ± {e['s0']:.2g}</b> at the centre "
            f"({p['xc']:.0f}, {p['yc']:.0f})",
            f"gradient {g['size']:.3g} per 1000 px toward {g['angle']:.0f}° "
            f"(image-axis angle), {fit.significance:.1f} × its error",
            f"x: {g['gx']:+.4g} ± {g['gx_error']:.2g}, y: {g['gy']:+.4g} ± {g['gy_error']:.2g} "
            "per 1000 px",
            f"{fit.change:.3g} across the frame"
            + (f" = {fit.change / s.sigma:.2f} σ" if s.sigma > 0 else ""),
            f"{fit.boxes} boxes used, {fit.rejected} left out; they scatter "
            f"{fit.scatter:.2f} × what the noise gives",
            f"sky of the flat image: {s.level:.6g}, σ {s.sigma:.4g} "
            f"<span style='color:{STATE_COLOURS.get(s.state, '#8b919c')}'>[{s.state}]</span>",
        ]
        for reason in fit.reasons:
            lines.append(f"• {reason}")
        self.card.setText("<br>".join(lines))
