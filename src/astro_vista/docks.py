"""Panels: resized by dragging their edges, moved to another side by dragging their title,
folded to a narrow strip (the arrow button, or a double-click on the title), and popped out
as windows of their own (for another monitor) and back into their place (the ⧉ button).
A panel can keep one part visible when folded: the Results panel keeps its command line.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

STRIP = 26  # the width (or height) of a folded panel, pixels
MAX = 16777215  # Qt's largest widget size


class TitleBar(QtWidgets.QWidget):
    """A panel's title bar: its name and a fold button; drawn upright along the side when
    the panel is folded against the left or right edge."""

    def __init__(self, dock: "FoldableDock"):
        super().__init__(dock)
        self.dock = dock
        self.show_title = True  # False: the bar holds only its buttons
        self.button = QtWidgets.QToolButton(self)
        self.button.setAutoRaise(True)
        self.button.setFixedSize(20, 20)
        self.button.clicked.connect(dock.toggle_fold)
        self.pop = QtWidgets.QToolButton(self)
        self.pop.setAutoRaise(True)
        self.pop.setFixedSize(20, 20)
        self.pop.clicked.connect(dock.toggle_float)
        self.update_button()

    def vertical(self) -> bool:
        return bool(self.dock.features() & QtWidgets.QDockWidget.DockWidgetVerticalTitleBar)

    def update_button(self) -> None:
        floating = self.dock.isFloating()
        self.pop.setText("⤓" if floating else "⧉")
        self.pop.setToolTip(
            f"Put {self.dock.windowTitle()} back in its place"
            if floating
            else f"Pop {self.dock.windowTitle()} out as a window of its own (for another monitor)"
        )
        self.button.setVisible(not floating)  # a window of its own is not folded
        side = self.dock.side()
        if self.dock.folded:
            arrow = {"left": "▸", "right": "◂", "bottom": "▴"}.get(side, "▾")
            self.button.setToolTip(f"Unfold {self.dock.windowTitle()}")
        else:
            arrow = {"left": "◂", "right": "▸", "bottom": "▾"}.get(side, "▴")
            self.button.setToolTip(f"Fold {self.dock.windowTitle()} (or double-click the title)")
        self.button.setText(arrow)
        self.update()

    def _text_length(self) -> int:
        font = self.font()
        font.setBold(True)
        return QtGui.QFontMetrics(font).horizontalAdvance(self.dock.windowTitle())

    def sizeHint(self):
        if self.vertical():  # room for the button and the whole title
            return QtCore.QSize(STRIP, self._text_length() + 44)
        return QtCore.QSize(120, STRIP - 2)

    def minimumSizeHint(self):
        return self.sizeHint() if self.vertical() else QtCore.QSize(40, STRIP - 2)

    def resizeEvent(self, _event):
        if self.vertical():
            self.button.move((self.width() - 20) // 2, 3)
            self.pop.hide()  # folded: only the unfold button
        else:
            y = (self.height() - 20) // 2
            self.button.move(self.width() - 23, y)
            self.pop.move(self.width() - (23 if self.dock.isFloating() else 46), y)
            self.pop.show()

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        palette = self.palette()
        p.fillRect(self.rect(), palette.color(QtGui.QPalette.Window).darker(115))
        p.setPen(palette.color(QtGui.QPalette.WindowText))
        font = p.font()
        font.setBold(True)
        p.setFont(font)
        if not self.show_title:
            return
        if self.vertical():  # the title reads downwards, below the button
            metrics = QtGui.QFontMetrics(font)
            p.translate(self.width() / 2 - metrics.ascent() / 2 + 1, 30)
            p.rotate(90)
            p.drawText(0, 0, self.dock.windowTitle())
        else:
            p.drawText(self.rect().adjusted(8, 0, -52, 0), Qt.AlignVCenter, self.dock.windowTitle())

    def mouseDoubleClickEvent(self, event):
        if self.dock.isFloating():
            self.dock.toggle_float()  # a double-click puts a popped-out panel back
        else:
            self.dock.toggle_fold()
        event.accept()


class FoldableDock(QtWidgets.QDockWidget):
    """A dock panel that can be folded and unfolded, and keeps its size meanwhile."""

    folded_changed = QtCore.Signal(bool)

    def __init__(
        self, title: str, widget: QtWidgets.QWidget, parent: QtWidgets.QMainWindow, keep=None
    ):
        super().__init__(title, parent)
        self.setObjectName(title)
        self.setWidget(widget)
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self.keep = keep  # what stays visible when the panel is folded (or None)
        self.fold_hook = None  # a function(folded) that folds the panel its own way
        self.folded = False
        self._size = 300  # the width (side) or height (bottom) to unfold to
        self.bar = TitleBar(self)
        self.setTitleBarWidget(self.bar)
        self.dockLocationChanged.connect(self._moved)
        self.topLevelChanged.connect(lambda _floating: self.bar.update_button())

    def toggle_float(self) -> None:
        """Pop the panel out as a window of its own, or put it back where it was."""
        if self.isFloating():
            self.setFloating(False)
        else:
            if self.folded:
                self.set_folded(False)
            size = self.size()
            self.setFloating(True)
            self.resize(max(size.width(), 360), max(size.height(), 300))
            corner = self.parent().geometry().topLeft() if self.parent() else QtCore.QPoint()
            self.move(corner + QtCore.QPoint(60, 60))
        self.bar.update_button()

    def side(self) -> str:
        area = self.parent().dockWidgetArea(self) if self.parent() else Qt.LeftDockWidgetArea
        return {
            Qt.LeftDockWidgetArea: "left",
            Qt.RightDockWidgetArea: "right",
            Qt.TopDockWidgetArea: "top",
            Qt.BottomDockWidgetArea: "bottom",
        }.get(area, "left")

    def _horizontal_side(self) -> bool:
        return self.side() in ("left", "right")

    def toggle_fold(self) -> None:
        self.set_folded(not self.folded)

    def set_folded(self, folded: bool) -> None:
        if folded == self.folded:
            return
        window = self.parent()
        sideways = self._horizontal_side()
        if folded and self.isFloating():
            return  # a window of its own is not folded
        if self.fold_hook is not None:  # the panel folds itself
            self.folded = folded
            self.fold_hook(folded)
            self.folded_changed.emit(folded)
            return
        if folded and self.keep is not None and not sideways:
            # fold to the title bar and the part kept (the command line)
            self._size = self.height()
            for child in self.widget().findChildren(
                QtWidgets.QWidget, options=Qt.FindDirectChildrenOnly
            ):
                child.setVisible(child is self.keep)
            self.setFixedHeight(STRIP + self.keep.sizeHint().height() + 8)
            self.folded = True
            self.bar.update_button()
            self.folded_changed.emit(True)
            return
        if folded:
            self._size = self.width() if sideways else self.height()
            self.widget().hide()
            if sideways:
                self.setFeatures(self.features() | QtWidgets.QDockWidget.DockWidgetVerticalTitleBar)
                self.setFixedWidth(STRIP)
            else:
                self.setFixedHeight(STRIP)
        else:
            self.setFeatures(self.features() & ~QtWidgets.QDockWidget.DockWidgetVerticalTitleBar)
            self.setMinimumSize(0, 0)
            self.setMaximumSize(MAX, MAX)
            self.widget().show()
            for child in self.widget().findChildren(
                QtWidgets.QWidget, options=Qt.FindDirectChildrenOnly
            ):
                child.show()
            orientation = Qt.Horizontal if sideways else Qt.Vertical
            QtCore.QTimer.singleShot(
                0, lambda: window.resizeDocks([self], [self._size], orientation)
            )
        self.folded = folded
        self.bar.update_button()
        self.bar.updateGeometry()
        self.folded_changed.emit(folded)

    def _moved(self, _area) -> None:
        """Moved to another side while folded: fold again in the new direction."""
        if self.folded:
            self.folded = False
            self.setMinimumSize(0, 0)
            self.setMaximumSize(MAX, MAX)
            self.setFeatures(self.features() & ~QtWidgets.QDockWidget.DockWidgetVerticalTitleBar)
            self.widget().show()
            self.set_folded(True)
        self.bar.update_button()
