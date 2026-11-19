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

"""Widget to record the data of an experiment in a zarr store (see
:mod:`pymeasure.experiment.zarr_store`)."""

import logging

from ...experiment.zarr_store import (
    ZARR_STORE_VARIABLE,
    ZarrResults,
    default_store,
    parse_metadata_text,
    save_results_to_zarr,
)
from .icechunk_widget import IcechunkInputWidget

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class ZarrInputWidget(IcechunkInputWidget):
    """
    Widget for recording the data of an experiment in a `zarr <https://zarr.dev>`_ store, a
    directory to which every experiment is appended as a new group. It works on every operating
    system (the local-file storage of icechunk does not work on Windows).

    It has the fields of the :class:`~pymeasure.display.widgets.icechunk_widget.IcechunkInputWidget`
    that make sense for zarr: the directory of the store, the group (a name that is valid on
    every operating system, with the placeholders of the filename), extra metadata as
    ``key=value`` entries separated by ``;`` (stored in the attribute ``extra_metadata`` of the
    group), the checkbox to control whether the measurement is saved, and a line showing the
    outcome. A zarr store has no branches and commits, so those fields are hidden.

    Use it as described for the icechunk widget: create the results of an experiment with
    :meth:`new_results` to record in the store while the experiment runs and plot from it, or
    connect :meth:`experiment_finished` to save a CSV-based experiment at its end.

    The default store is the environment variable ``PYMEASURE_ZARR_STORE`` or the entry of that
    name in a ``.env`` file (see :func:`~pymeasure.experiment.zarr_store.default_store`).

    :param procedure_class: Procedure class of the experiments (provides the placeholders).
    :param parent: Parent widget.
    :param storage_factory: Callable that turns the text of the store field into a zarr store
        object, for stores that are not a local directory (for example S3).
    """

    _TITLE = "Zarr store"
    _STORE_PLACEHOLDER = "Directory of the zarr store"
    _STORE_TOOLTIP = ("The directory of the zarr store. It is created if it does not exist yet. "
                      "Every experiment is added to it as a new group.")
    _SAVE_TOOLTIP = ("Control whether the measurement is saved to the zarr store with the "
                     "settings\nbelow (checked) or not (unchecked).")
    _METADATA_LABEL = "Extra metadata"
    _METADATA_SUMMARY = (
        "Extra metadata as 'key=value' entries separated by ';', for example "
        "'sample={Sample}; operator=me', stored in the attribute 'extra_metadata' of the group.")

    def __init__(self, procedure_class, parent=None, storage_factory=None):
        super().__init__(procedure_class, parent=parent, storage_factory=storage_factory)
        for name in ("Branch", "Commit message"):  # a zarr store has neither
            for item in self._rows[name]:
                item.hide()

    def _default_store(self):
        return default_store()

    def _no_store_message(self) -> str:
        return (f"No zarr store is selected (and no default store is set in the environment "
                f"variable {ZARR_STORE_VARIABLE} or in a .env file).")

    def _save_results(self, results, storage):
        return save_results_to_zarr(
            results,
            storage,
            group=self.group,
            metadata=parse_metadata_text(self.metadata_text),
        )

    def _new_results(self, procedure, store):
        return ZarrResults(
            procedure,
            store,
            group=self.group,
            metadata=parse_metadata_text(self.metadata_text),
            on_commit=self._commit_done.emit,
            on_error=lambda exc: self._commit_failed.emit(str(exc)),
        )

    def _recording_text(self, results) -> str:
        return f"Recording in '{results.group}'"

    def _saved_text(self, result) -> str:
        return f"Saved as '{result.group}'"
