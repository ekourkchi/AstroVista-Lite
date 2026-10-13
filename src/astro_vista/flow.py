"""A layout that places its widgets in a row and wraps them onto further rows when the
room runs out, like words in a paragraph."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class FlowLayout(QtWidgets.QLayout):
    def __init__(self, parent=None, spacing: int = 8):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QtCore.QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, move=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:  # as narrow as the widest single item
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _arrange(self, rect, move: bool) -> int:
        """Place the items row by row; each row's items are centred on its middle line, so
        a check box lines up with the menus and fields beside it."""
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        rows, row, x = [], [], area.x()
        for item in self._items:  # first, which items go on which row
            hint = item.sizeHint()
            if row and x + hint.width() > area.right() + 1:
                rows.append(row)
                row, x = [], area.x()
            row.append(item)
            x += hint.width() + self._spacing
        if row:
            rows.append(row)
        y = area.y()
        for row in rows:
            height = max(item.sizeHint().height() for item in row)
            x = area.x()
            for item in row:
                hint = item.sizeHint()
                if move:
                    top = y + (height - hint.height()) // 2
                    item.setGeometry(QtCore.QRect(QtCore.QPoint(x, top), hint))
                x += hint.width() + self._spacing
            y += height + self._spacing // 2
        return (y - self._spacing // 2 if rows else y) - rect.y() + m.bottom()
