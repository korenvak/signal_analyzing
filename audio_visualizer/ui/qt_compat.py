"""
Qt compatibility layer for PySide6/PyQt5.

This module provides a unified interface for Qt imports, allowing the application
to work with either PySide6 or PyQt5 depending on which is available.

Usage:
    from .qt_compat import QtWidgets, QtCore, QtGui, Signal, Slot
"""

import sys

# Try PyQt5 first (for compatibility with older systems), then fall back to PySide6
QT_BINDING = None

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import (
        Qt, QTimer, QThread, QObject, QEvent, QSize, QPoint, QRect,
        QPointF, QRectF, QMimeData, QUrl, QByteArray
    )
    from PyQt5.QtGui import (
        QKeySequence, QColor, QPen, QBrush, QPainter, QImage, QPixmap,
        QFont, QFontMetrics, QCursor, QPalette, QIcon, QDoubleValidator,
        QIntValidator, QLinearGradient, QRadialGradient, QTransform,
        QPolygonF, QPainterPath, QDragEnterEvent, QDropEvent, QKeyEvent,
        QMouseEvent, QWheelEvent, QResizeEvent, QCloseEvent, QPaintEvent
    )
    from PyQt5.QtWidgets import QShortcut
    # In PyQt5, QAction is in QtWidgets
    from PyQt5.QtWidgets import QAction
    QT_BINDING = 'PyQt5'
except ImportError:
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Signal, Slot
        from PySide6.QtWidgets import *
        from PySide6.QtCore import (
            Qt, QTimer, QThread, QObject, QEvent, QSize, QPoint, QRect,
            QPointF, QRectF, QMimeData, QUrl, QByteArray
        )
        from PySide6.QtGui import (
            QKeySequence, QColor, QPen, QBrush, QPainter, QImage, QPixmap,
            QFont, QFontMetrics, QCursor, QPalette, QIcon, QDoubleValidator,
            QIntValidator, QLinearGradient, QRadialGradient, QTransform,
            QPolygonF, QPainterPath, QDragEnterEvent, QDropEvent, QKeyEvent,
            QMouseEvent, QWheelEvent, QResizeEvent, QCloseEvent, QPaintEvent,
            QShortcut
        )
        # In PySide6, QAction is in QtGui
        from PySide6.QtGui import QAction
        QT_BINDING = 'PySide6'
    except ImportError:
        raise ImportError("Neither PyQt5 nor PySide6 is available. Please install one of them.")

# Print which binding is being used (for debugging)
# print(f"Using Qt binding: {QT_BINDING}")

# Handle differences between PyQt5 and PySide6

# exec() vs exec_() difference
if QT_BINDING == 'PyQt5':
    # PyQt5 uses exec_() for compatibility with Python 2
    # But modern PyQt5 also supports exec()
    # We'll patch QDialog and QApplication to ensure exec() works
    _original_dialog_exec = QDialog.exec if hasattr(QDialog, 'exec') else None
    if not _original_dialog_exec:
        QDialog.exec = QDialog.exec_

    _original_app_exec = QApplication.exec if hasattr(QApplication, 'exec') else None
    if not _original_app_exec:
        QApplication.exec = QApplication.exec_

    # QMessageBox.exec() might also need patching
    _original_msgbox_exec = QMessageBox.exec if hasattr(QMessageBox, 'exec') else None
    if not _original_msgbox_exec:
        QMessageBox.exec = QMessageBox.exec_

    # PyQt5 enum compatibility - PySide6 uses Qt.Orientation.Horizontal, PyQt5 uses Qt.Horizontal
    # Create compatibility namespace classes
    # Create a metaclass that dynamically forwards attribute access to Qt
    class _QtEnumProxyMeta(type):
        """Metaclass for dynamic Qt enum attribute access."""
        def __getattr__(cls, name):
            return getattr(Qt, name)

    # Create proxy classes for all enum-style accesses
    class _QtOrientation(metaclass=_QtEnumProxyMeta):
        Horizontal = Qt.Horizontal
        Vertical = Qt.Vertical

    class _QtKey(metaclass=_QtEnumProxyMeta):
        """Dynamic proxy for Qt.Key enum values."""
        pass

    class _QtWindowType(metaclass=_QtEnumProxyMeta):
        Window = Qt.Window
        WindowStaysOnTopHint = Qt.WindowStaysOnTopHint
        Dialog = Qt.Dialog
        FramelessWindowHint = Qt.FramelessWindowHint

    class _QtAlignmentFlag(metaclass=_QtEnumProxyMeta):
        AlignLeft = Qt.AlignLeft
        AlignRight = Qt.AlignRight
        AlignCenter = Qt.AlignCenter
        AlignTop = Qt.AlignTop
        AlignBottom = Qt.AlignBottom
        AlignVCenter = Qt.AlignVCenter
        AlignHCenter = Qt.AlignHCenter

    class _QtItemFlag(metaclass=_QtEnumProxyMeta):
        ItemIsEnabled = Qt.ItemIsEnabled
        ItemIsSelectable = Qt.ItemIsSelectable
        ItemIsEditable = Qt.ItemIsEditable
        ItemIsUserCheckable = Qt.ItemIsUserCheckable

    class _QtCheckState(metaclass=_QtEnumProxyMeta):
        Unchecked = Qt.Unchecked
        Checked = Qt.Checked
        PartiallyChecked = Qt.PartiallyChecked

    class _QtMouseButton(metaclass=_QtEnumProxyMeta):
        LeftButton = Qt.LeftButton
        RightButton = Qt.RightButton
        MiddleButton = Qt.MiddleButton
        NoButton = Qt.NoButton

    class _QtKeyboardModifier(metaclass=_QtEnumProxyMeta):
        NoModifier = Qt.NoModifier
        ShiftModifier = Qt.ShiftModifier
        ControlModifier = Qt.ControlModifier
        AltModifier = Qt.AltModifier
        MetaModifier = Qt.MetaModifier

    class _QtTextFormat(metaclass=_QtEnumProxyMeta):
        PlainText = Qt.PlainText
        RichText = Qt.RichText
        AutoText = Qt.AutoText

    class _QtCursorShape(metaclass=_QtEnumProxyMeta):
        ArrowCursor = Qt.ArrowCursor
        CrossCursor = Qt.CrossCursor
        WaitCursor = Qt.WaitCursor
        PointingHandCursor = Qt.PointingHandCursor

    class _QtFocusPolicy(metaclass=_QtEnumProxyMeta):
        NoFocus = Qt.NoFocus
        TabFocus = Qt.TabFocus
        ClickFocus = Qt.ClickFocus
        StrongFocus = Qt.StrongFocus
        WheelFocus = Qt.WheelFocus

    class _QtScrollBarPolicy(metaclass=_QtEnumProxyMeta):
        ScrollBarAsNeeded = Qt.ScrollBarAsNeeded
        ScrollBarAlwaysOff = Qt.ScrollBarAlwaysOff
        ScrollBarAlwaysOn = Qt.ScrollBarAlwaysOn

    class _QtSizePolicy(metaclass=_QtEnumProxyMeta):
        pass  # QSizePolicy is a class, not enum on Qt

    class _QtContextMenuPolicy(metaclass=_QtEnumProxyMeta):
        CustomContextMenu = Qt.CustomContextMenu
        NoContextMenu = Qt.NoContextMenu
        DefaultContextMenu = Qt.DefaultContextMenu
        ActionsContextMenu = Qt.ActionsContextMenu
        PreventContextMenu = Qt.PreventContextMenu

    class _QtItemDataRole(metaclass=_QtEnumProxyMeta):
        UserRole = Qt.UserRole
        DisplayRole = Qt.DisplayRole
        EditRole = Qt.EditRole
        DecorationRole = Qt.DecorationRole
        ToolTipRole = Qt.ToolTipRole

    # Patch Qt to add PySide6-style enum access
    Qt.Orientation = _QtOrientation
    Qt.Key = _QtKey
    Qt.WindowType = _QtWindowType
    Qt.AlignmentFlag = _QtAlignmentFlag
    Qt.ItemFlag = _QtItemFlag
    Qt.CheckState = _QtCheckState
    Qt.MouseButton = _QtMouseButton
    Qt.KeyboardModifier = _QtKeyboardModifier
    Qt.TextFormat = _QtTextFormat
    Qt.CursorShape = _QtCursorShape
    Qt.FocusPolicy = _QtFocusPolicy
    Qt.ScrollBarPolicy = _QtScrollBarPolicy
    Qt.ContextMenuPolicy = _QtContextMenuPolicy
    Qt.ItemDataRole = _QtItemDataRole

    # Create enum proxies for widget classes that use PySide6-style enum namespacing
    # QHeaderView.ResizeMode
    class _QHeaderViewResizeMode:
        Interactive = QHeaderView.Interactive
        Fixed = QHeaderView.Fixed
        Stretch = QHeaderView.Stretch
        ResizeToContents = QHeaderView.ResizeToContents
    QHeaderView.ResizeMode = _QHeaderViewResizeMode

    # QAbstractItemView enums (also used by QTableWidget, QListWidget, QTreeWidget)
    class _QAbstractItemViewSelectionBehavior:
        SelectItems = QAbstractItemView.SelectItems
        SelectRows = QAbstractItemView.SelectRows
        SelectColumns = QAbstractItemView.SelectColumns
    QAbstractItemView.SelectionBehavior = _QAbstractItemViewSelectionBehavior

    class _QAbstractItemViewSelectionMode:
        SingleSelection = QAbstractItemView.SingleSelection
        MultiSelection = QAbstractItemView.MultiSelection
        ExtendedSelection = QAbstractItemView.ExtendedSelection
        ContiguousSelection = QAbstractItemView.ContiguousSelection
        NoSelection = QAbstractItemView.NoSelection
    QAbstractItemView.SelectionMode = _QAbstractItemViewSelectionMode

    class _QAbstractItemViewDragDropMode:
        NoDragDrop = QAbstractItemView.NoDragDrop
        DragOnly = QAbstractItemView.DragOnly
        DropOnly = QAbstractItemView.DropOnly
        DragDrop = QAbstractItemView.DragDrop
        InternalMove = QAbstractItemView.InternalMove
    QAbstractItemView.DragDropMode = _QAbstractItemViewDragDropMode

    class _QAbstractItemViewEditTrigger:
        NoEditTriggers = QAbstractItemView.NoEditTriggers
        CurrentChanged = QAbstractItemView.CurrentChanged
        DoubleClicked = QAbstractItemView.DoubleClicked
        SelectedClicked = QAbstractItemView.SelectedClicked
        EditKeyPressed = QAbstractItemView.EditKeyPressed
        AnyKeyPressed = QAbstractItemView.AnyKeyPressed
        AllEditTriggers = QAbstractItemView.AllEditTriggers
    QAbstractItemView.EditTrigger = _QAbstractItemViewEditTrigger

    # Also add to subclasses that might use these
    QTableWidget.SelectionBehavior = _QAbstractItemViewSelectionBehavior
    QTableWidget.SelectionMode = _QAbstractItemViewSelectionMode
    QTableWidget.DragDropMode = _QAbstractItemViewDragDropMode
    QTableWidget.EditTrigger = _QAbstractItemViewEditTrigger

    QListWidget.SelectionBehavior = _QAbstractItemViewSelectionBehavior
    QListWidget.SelectionMode = _QAbstractItemViewSelectionMode
    QListWidget.DragDropMode = _QAbstractItemViewDragDropMode

    QTreeWidget.SelectionBehavior = _QAbstractItemViewSelectionBehavior
    QTreeWidget.SelectionMode = _QAbstractItemViewSelectionMode
    QTreeWidget.DragDropMode = _QAbstractItemViewDragDropMode

    # QTabWidget enums
    class _QTabWidgetTabPosition:
        North = QTabWidget.North
        South = QTabWidget.South
        West = QTabWidget.West
        East = QTabWidget.East
    QTabWidget.TabPosition = _QTabWidgetTabPosition

    class _QTabWidgetTabShape:
        Rounded = QTabWidget.Rounded
        Triangular = QTabWidget.Triangular
    QTabWidget.TabShape = _QTabWidgetTabShape

    # QSizePolicy enum
    class _QSizePolicyPolicy:
        Fixed = QSizePolicy.Fixed
        Minimum = QSizePolicy.Minimum
        Maximum = QSizePolicy.Maximum
        Preferred = QSizePolicy.Preferred
        Expanding = QSizePolicy.Expanding
        MinimumExpanding = QSizePolicy.MinimumExpanding
        Ignored = QSizePolicy.Ignored
    QSizePolicy.Policy = _QSizePolicyPolicy


def get_qt_binding():
    """Return the name of the Qt binding being used."""
    return QT_BINDING


def is_pyqt5():
    """Return True if using PyQt5."""
    return QT_BINDING == 'PyQt5'


def is_pyside6():
    """Return True if using PySide6."""
    return QT_BINDING == 'PySide6'
