#
# This file is part of the PyMeasure package.
#
# Copyright (c) 2013-2026 PyMeasure Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

import logging
import re
import textwrap

from ...experiment.icechunk_store import (
    IcechunkResults,
    IcechunkSaveResult,
    default_store,
    parse_metadata_text,
    save_results_to_icechunk,
)
from ..Qt import QtCore, QtGui, QtWidgets
from .directory_widget import DirectoryLineEdit
from .filename_widget import FilenameValidator, PlaceholderCompleter

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# Characters that may not appear outside placeholders (placeholders are checked separately)
_TEXT_PATTERN = r"^[^{}]*$"
_GROUP_PATTERN = r'^[^<>"\\|?*{}]*$'


class _PlaceholderValidator(FilenameValidator):
    """:class:`FilenameValidator` for text with a different set of allowed characters."""

    def __init__(self, placeholders, parent, valid_text):
        super().__init__(placeholders, parent)
        self.valid_filename = re.compile(valid_text)


class PlaceholderLineEdit(QtWidgets.QLineEdit):
    """
    Line edit for text with ``{placeholders}``, like the filename field of the
    :class:`~pymeasure.display.widgets.fileinput_widget.FileInputWidget`.

    The parameters and metadata of the procedure, ``date`` and ``time`` can be used as
    placeholders (in standard python format, i.e.: ``{variable name:formatspec}``). A completer
    suggests them after typing ``{``, and an unknown placeholder is marked with a warning icon.

    :param procedure_class: Procedure class that provides the placeholders.
    :param text: Initial text.
    :param summary: Description of the field for its tooltip.
    :param valid_text: Regular expression for the text without its placeholders.
    """

    def __init__(self, procedure_class, text="", summary="", valid_text=_TEXT_PATTERN,
                 parent=None):
        super().__init__(text, parent=parent)
        self.placeholders = procedure_class.placeholder_names()
        self.placeholders.extend(["date", "time"])

        # keep references: Qt does not own a validator or completer without a Qt parent
        self._completer = PlaceholderCompleter(self.placeholders)
        self._validator = _PlaceholderValidator(self.placeholders, self, valid_text)
        self.setCompleter(self._completer)
        self.setValidator(self._validator)

        placeholders = "'" + "';\n- '".join(self.placeholders) + "'"
        self.setToolTip("\n".join(textwrap.wrap(
            summary + " Placeholders (in standard python format, i.e.: "
            "'{variable name:formatspec}') are replaced by the respective value.", width=100)
        ) + f"\nValid placeholders are:\n- {placeholders}.")


class GroupPathLineEdit(PlaceholderLineEdit):
    """
    :class:`PlaceholderLineEdit` for the path of a zarr group, groups separated by ``/``.
    Characters that are not valid in a group path are refused.
    """

    def __init__(self, procedure_class, text="DATA", parent=None):
        super().__init__(
            procedure_class, text, parent=parent, valid_text=_GROUP_PATTERN,
            summary="The path of the group in the store to which the data is written; use "
                    "'/' for nested groups. An index number ('_#') is added to the last part "
                    "to ensure that the group is new.")


class IcechunkInputWidget(QtWidgets.QWidget):
    """
    Widget for saving the data of an experiment to an
    `icechunk <https://icechunk.io>`_ store, next to (or instead of) the file that the
    :class:`~pymeasure.display.widgets.fileinput_widget.FileInputWidget` controls.

    It consists of a field for the directory of the store (with browse button), the branch,
    the group in the store (:class:`GroupPathLineEdit`), the commit message and extra commit
    metadata (:class:`PlaceholderLineEdit`; entries ``key=value`` separated by ``;``), a
    checkbox to control whether the measurement is saved, and a line showing the outcome of
    the last save. The group, the commit message and the metadata values accept placeholders
    for the parameters of the procedure.

    Connect :meth:`experiment_finished` to the ``finished`` signal of the manager of a window
    to save every completed experiment (connect ``abort_returned`` or ``failed`` as well to
    keep partial data). The save is done in the GUI thread and errors are not raised but logged,
    shown in the widget and emitted as :attr:`failed`.

    .. code-block:: python

        class MainWindow(ManagedWindow):
            def __init__(self):
                super().__init__(MyProcedure, x_axis="Voltage (V)", y_axis="Current (A)")
                self.icechunk_input = IcechunkInputWidget(MyProcedure, parent=self)
                dock = QtWidgets.QDockWidget("Icechunk store")
                dock.setWidget(self.icechunk_input)
                dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
                self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
                self.manager.finished.connect(self.icechunk_input.experiment_finished)

    The store field starts with the default store, if one is configured: the environment
    variable ``PYMEASURE_ICECHUNK_STORE``, or the entry ``PYMEASURE_ICECHUNK_STORE=...`` of a
    ``.env`` file in the working directory or one of its parents (see
    :func:`~pymeasure.experiment.icechunk_store.default_store`). An empty field uses it too.

    This needs the optional dependencies ``icechunk``, ``zarr`` and ``xarray``
    (``pip install pymeasure[icechunk]``, Python 3.11 or newer).

    :param procedure_class: Procedure class of the experiments (provides the placeholders).
    :param parent: Parent widget.
    :param storage_factory: Callable that turns the text of the store field into an
        ``icechunk.Storage``, for stores that are not a local directory (for example S3). By
        default the text is the path of a local directory.
    """

    #: Emitted with the :class:`~pymeasure.experiment.icechunk_store.IcechunkSaveResult` after
    #: a successful save.
    saved = QtCore.Signal(object)
    #: Emitted with the error message if a save failed.
    failed = QtCore.Signal(str)

    # results report from the thread of the worker; these signals bring that to the GUI thread
    _commit_done = QtCore.Signal(object)
    _commit_failed = QtCore.Signal(str)

    def __init__(self, procedure_class, parent=None, storage_factory=None):
        super().__init__(parent)
        self.procedure_class = procedure_class
        self.storage_factory = storage_factory
        self._commit_done.connect(self._show_saved)
        self._commit_failed.connect(self._show_failed)

        self._setup_ui()
        self._layout()

        default = default_store()  # environment variable or .env file
        if default:
            self.store = default

    def _setup_ui(self):
        self.save_toggle = QtWidgets.QCheckBox("Save data", self)
        self.save_toggle.setLayoutDirection(QtCore.Qt.LayoutDirection.RightToLeft)
        self.save_toggle.setChecked(True)
        self.save_toggle.stateChanged.connect(self.set_input_fields_enabled)
        self.save_toggle.setToolTip(
            "Control whether the measurement is saved to the icechunk store with the settings\n"
            "below (checked) or not (unchecked)."
        )

        self.store_input = DirectoryLineEdit(parent=self)
        self.store_input.setPlaceholderText("Directory of the icechunk store")
        self.store_input.setToolTip(
            "The directory of the icechunk store. It is created and initialised if it does not "
            "exist yet."
        )

        self.branch_input = QtWidgets.QLineEdit("main", self)
        self._branch_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"[^\s/]*"), self.branch_input)
        self.branch_input.setValidator(self._branch_validator)
        self.branch_input.setToolTip(
            "The branch that the data is committed to (no spaces or '/'). A branch that does\n"
            "not exist yet is created from 'main'."
        )

        self.group_input = GroupPathLineEdit(self.procedure_class, parent=self)

        self.message_input = PlaceholderLineEdit(
            self.procedure_class, "Add measurement {date} {time}", parent=self,
            summary="The message of the commit that saves the measurement.")

        self.metadata_input = PlaceholderLineEdit(
            self.procedure_class, parent=self,
            summary="Extra metadata of the commit as 'key=value' entries separated by ';', "
                    "for example 'sample={Sample}; operator=me'. The group, the procedure and "
                    "the pymeasure version are always added.")
        self.metadata_input.setPlaceholderText("key=value; key2=value2")

        self.status_label = QtWidgets.QLabel("", self)
        self.status_label.setWordWrap(True)

    def _layout(self):
        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)

        store_label = QtWidgets.QLabel("Icechunk store", self)
        store_label.setToolTip(self.store_input.toolTip())
        store_box = QtWidgets.QHBoxLayout()
        store_box.addWidget(store_label)
        store_box.addWidget(self.save_toggle)
        vbox.addLayout(store_box)
        vbox.addWidget(self.store_input)

        for text, field in (("Branch", self.branch_input), ("Group", self.group_input),
                            ("Commit message", self.message_input),
                            ("Commit metadata", self.metadata_input)):
            label = QtWidgets.QLabel(text, self)
            label.setToolTip(field.toolTip())
            vbox.addWidget(label)
            vbox.addWidget(field)
        vbox.addWidget(self.status_label)

        self.setLayout(vbox)

    @property
    def store(self):
        """String controlling the store: a directory (or whatever the ``storage_factory``
        understands)."""
        return self.store_input.text()

    @store.setter
    def store(self, value):
        self.store_input.setText(str(value))

    @property
    def branch(self):
        """String controlling the branch that is committed to."""
        return self.branch_input.text()

    @branch.setter
    def branch(self, value):
        self.branch_input.setText(str(value))

    @property
    def group(self):
        """String controlling the template of the group path."""
        return self.group_input.text()

    @group.setter
    def group(self, value):
        self.group_input.setText(str(value))

    @property
    def commit_message(self):
        """String controlling the template of the commit message."""
        return self.message_input.text()

    @commit_message.setter
    def commit_message(self, value):
        self.message_input.setText(str(value))

    @property
    def metadata_text(self):
        """String controlling the extra commit metadata, as ``key=value; key2=value2``."""
        return self.metadata_input.text()

    @metadata_text.setter
    def metadata_text(self, value):
        self.metadata_input.setText(str(value))

    @property
    def store_measurement(self):
        """Boolean controlling whether the measurement will be saved."""
        return self.save_toggle.isChecked()

    @store_measurement.setter
    def store_measurement(self, value):
        self.save_toggle.setChecked(bool(value))

    def set_input_fields_enabled(self, state):
        for field in (self.store_input, self.branch_input, self.group_input,
                      self.message_input, self.metadata_input):
            field.setEnabled(bool(state))

    def save(self, results):
        """Save ``results`` with the current settings of the fields.

        Errors do not propagate: they are logged, shown below the fields and emitted as
        :attr:`failed`; on success :attr:`saved` is emitted.

        :param results: :class:`~pymeasure.experiment.results.Results` to save.
        :returns: The :class:`~pymeasure.experiment.icechunk_store.IcechunkSaveResult`, or
            ``None`` if the save failed.
        """
        try:
            store = self._store_text()
            storage = self.storage_factory(store) if self.storage_factory else store
            result = save_results_to_icechunk(
                results,
                storage,
                group=self.group,
                message=self.commit_message,
                branch=self.branch,
                metadata=parse_metadata_text(self.metadata_text),
            )
        except Exception as exc:
            log.exception("Failed to save the data to the icechunk store")
            self._show_failed(str(exc))
            return None
        self._show_saved(result)
        return result

    def new_results(self, procedure):
        """Create :class:`~pymeasure.experiment.icechunk_store.IcechunkResults` for ``procedure``
        with the current settings of the fields, to use instead of a CSV file.

        The data of the experiment is then written to the store while it runs and the plots read
        it from there; the widget shows the outcome of the final commit. If saving is switched
        off, a store in memory is used (the data can be plotted, but is not kept).
        Meant for the ``queue`` method of a window, see the tutorial. Do not connect
        :meth:`experiment_finished` as well, that would save the data a second time.

        Errors (for example an invalid placeholder) do not propagate: they are logged, shown
        below the fields and emitted as :attr:`failed`.

        :param procedure: Procedure of the experiment.
        :returns: The results, or ``None`` if they could not be created.
        """
        try:
            if self.store_measurement:
                store = self._store_text()
                store = self.storage_factory(store) if self.storage_factory else store
            else:
                store = None
            results = IcechunkResults(
                procedure,
                store,
                group=self.group,
                message=self.commit_message,
                branch=self.branch,
                metadata=parse_metadata_text(self.metadata_text),
                on_commit=self._commit_done.emit,
                on_error=lambda exc: self._commit_failed.emit(str(exc)),
            )
        except Exception as exc:
            log.exception("Failed to prepare the icechunk store")
            self._show_failed(str(exc))
            return None
        self.status_label.setStyleSheet("")
        self.status_label.setText(f"Recording in '{results.group}' on branch '{results.branch}'")
        return results

    def _store_text(self) -> str:
        """The store field, or the default store (see
        :func:`~pymeasure.experiment.icechunk_store.default_store`) if the field is empty."""
        store = self.store.strip() or default_store()
        if not store:
            raise ValueError(
                "No icechunk store is selected (and no default store is set in the environment "
                "variable PYMEASURE_ICECHUNK_STORE or in a .env file).")
        return store

    def _show_failed(self, message: str):
        first_line = (message.strip().splitlines() or ["unknown error"])[0]
        self.status_label.setStyleSheet("color: red")
        self.status_label.setText(f"Not saved: {first_line}")
        self.failed.emit(message)

    def _show_saved(self, result: IcechunkSaveResult):
        self.status_label.setStyleSheet("")
        self.status_label.setText(
            f"Saved as '{result.group}' on branch '{result.branch}' ({result.snapshot_id[:8]})")
        self.saved.emit(result)

    def experiment_finished(self, experiment):
        """Save the results of ``experiment`` if saving is switched on.

        Meant to be connected to the ``finished`` signal of a
        :class:`~pymeasure.display.manager.Manager`.
        """
        if self.store_measurement:
            self.save(experiment.results)
