"""Line-art icons, drawn as in the full AstroVista: ELLIPROF (nested, tilted ellipses), the
Source finder (detected sources, circled), the display (a screen), contours, unmasking and
the tip bulb."""

from __future__ import annotations

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt


def _pen(colour, width=1.6) -> QtGui.QPen:
    pen = QtGui.QPen(QtGui.QColor(colour))
    pen.setWidthF(width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def icon(name: str, size: int = 22) -> QtGui.QIcon:
    """An icon on a 24 x 24 grid, in the palette's text colour (sharp on Retina too)."""
    colour = QtWidgets.QApplication.palette().color(QtGui.QPalette.WindowText)
    result = QtGui.QIcon()
    for scale in (1, 2):
        pixmap = QtGui.QPixmap(size * scale, size * scale)
        pixmap.setDevicePixelRatio(scale)
        pixmap.fill(Qt.transparent)
        p = QtGui.QPainter(pixmap)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.scale(size / 24.0, size / 24.0)
        p.setPen(_pen(colour))
        if name == "elliprof":  # two nested ellipses, tilted
            p.translate(12, 12)
            p.rotate(-30)
            p.drawEllipse(QRectF(-9, -5, 18, 10))
            p.drawEllipse(QRectF(-5, -2.8, 10, 5.6))
        elif name == "display":  # a screen showing a half-dark, half-bright disc
            p.drawRoundedRect(QRectF(3, 4, 18, 13), 1.5, 1.5)
            p.drawLine(QPointF(9, 20.5), QPointF(15, 20.5))
            p.drawLine(QPointF(12, 17), QPointF(12, 20.5))
            p.drawEllipse(QPointF(12, 10.5), 4, 4)
            p.setBrush(colour)
            p.drawPie(QRectF(8, 6.5, 8, 8), 90 * 16, 180 * 16)
            p.setBrush(Qt.NoBrush)
        elif name == "contours":  # three nested, uneven closed lines
            for k, (rx, ry) in enumerate(((9.5, 7.5), (6.5, 5), (3.2, 2.4))):
                path = QtGui.QPainterPath()
                path.moveTo(12 + rx, 12)
                path.cubicTo(
                    12 + rx, 12 - ry * 1.2, 12 - rx * 0.4, 12 - ry * 1.1, 12 - rx, 12 - ry * 0.2
                )
                path.cubicTo(
                    12 - rx * 1.1, 12 + ry * 0.9, 12 + rx * 0.3, 12 + ry * 1.2, 12 + rx, 12
                )
                p.drawPath(path)
        elif name == "sources":  # detected sources, two of them circled
            p.setBrush(colour)
            for x, y, r in ((7, 8, 1.6), (16, 6, 1.2), (14, 16, 2.0), (6, 17, 1.0)):
                p.drawEllipse(QPointF(x, y), r, r)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(14, 16), 4.5, 4.5)
            p.drawEllipse(QPointF(7, 8), 3.6, 3.6)
        elif name == "unmask":  # a circle with a minus inside: take pixels out of the mask
            p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
            p.setPen(_pen(colour, 2.4))
            p.drawLine(QPointF(7.5, 12), QPointF(16.5, 12))
        elif name == "bulb":  # a tip: a yellow light bulb (in its own colours)
            p.setPen(_pen("#a8790a", 1.3))
            p.setBrush(QtGui.QColor("#ffd23f"))
            glass = QtGui.QPainterPath()
            glass.moveTo(9, 15.5)
            glass.cubicTo(4, 12, 5.5, 3, 12, 3)
            glass.cubicTo(18.5, 3, 20, 12, 15, 15.5)
            glass.closeSubpath()
            p.drawPath(glass)
            p.setBrush(QtGui.QColor("#9aa3ad"))
            p.setPen(_pen("#6d757e", 1.1))
            p.drawRoundedRect(QRectF(9, 16, 6, 4.5), 1, 1)
            p.drawLine(QPointF(10.5, 22), QPointF(13.5, 22))
            p.setBrush(Qt.NoBrush)
        p.end()
        result.addPixmap(pixmap)
    return result
