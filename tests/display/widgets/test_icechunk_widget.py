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

import os
import subprocess
import sys

import logging

import pytest

from pymeasure.display.Qt import QtCore, QtGui, QtWidgets
from pymeasure.display.widgets import IcechunkInputWidget
from pymeasure.display.widgets.icechunk_widget import GroupPathLineEdit, PlaceholderLineEdit
from pymeasure.display.windows.managed_window import ManagedWindow
from pymeasure.experiment import FloatParameter, Parameter, Procedure, Results

State = QtGui.QValidator.State


class _Procedure(Procedure):
    voltage = FloatParameter("Voltage", units="V", default=2.5)
    sample = Parameter("Sample", default="S1")

    DATA_COLUMNS = ["Voltage (V)", "Current (A)"]

    def execute(self):
        pass


@pytest.fixture(autouse=True)
def _clean_root_logging_handlers():
    """Remove the LogWidget handler that a ManagedWindow adds to the root logger.

    Left in place after the window is destroyed, it raises RuntimeError on the deleted Qt
    object for every later log record, which breaks unrelated tests.
    """
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)


def make_results(tmp_path, rows=3):
    results = Results(_Procedure(), str(tmp_path / "data.csv"))
    with open(results.data_filename, "a", encoding="utf-8") as file:
        file.writelines(
            results.format({"Voltage (V)": 0.5 * i, "Current (A)": 1e-6 * i}) + "\n"
            for i in range(rows)
        )
    return results


@pytest.fixture
def widget(qtbot):
    wdg = IcechunkInputWidget(_Procedure)
    qtbot.addWidget(wdg)
    wdg.show()
    return wdg


@pytest.fixture
def store(tmp_path):
    """Skip the test without the optional dependencies; return a store directory."""
    pytest.importorskip("icechunk", reason="needs Python >= 3.11 and pymeasure[icechunk]")
    pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    return tmp_path / "store"


def history(directory):
    import icechunk

    storage = icechunk.local_filesystem_storage(str(directory))
    return list(icechunk.Repository.open(storage).ancestry(branch="main"))


# ---- fields ---------------------------------------------------------------------------------

def test_defaults(widget):
    assert widget.store == "" and widget.branch == "main" and widget.group == "DATA"
    assert widget.commit_message == "Add measurement {date} {time}"
    assert widget.metadata_text == "" and widget.store_measurement is True
    assert widget.status_label.text() == ""


def test_get_set(widget):
    widget.store = "/tmp/store"
    widget.branch = "dev"
    widget.group = "a/{Sample}"
    widget.commit_message = "Run {Voltage}"
    widget.metadata_text = "k=v"
    widget.store_measurement = False
    assert (widget.store, widget.branch, widget.group) == ("/tmp/store", "dev", "a/{Sample}")
    assert (widget.commit_message, widget.metadata_text) == ("Run {Voltage}", "k=v")
    assert widget.store_measurement is False


def test_toggle_enables_and_disables_all_fields(widget):
    fields = (widget.store_input, widget.branch_input, widget.group_input,
              widget.message_input, widget.metadata_input)
    assert all(field.isEnabled() for field in fields)
    widget.save_toggle.setChecked(False)
    assert not any(field.isEnabled() for field in fields)
    widget.save_toggle.setChecked(True)
    assert all(field.isEnabled() for field in fields)


def test_placeholders_come_from_the_procedure(widget):
    assert sorted(widget.group_input.placeholders) == ["Sample", "Voltage", "date", "time"]
    tooltip = widget.message_input.toolTip()
    assert "'Voltage'" in tooltip and "'date'" in tooltip


def test_store_field_has_a_browse_button(widget):
    assert len(widget.store_input.actions()) == 1


@pytest.mark.parametrize("text, state", (
    ("DATA", State.Acceptable),
    ("run/{Voltage}_{Sample}", State.Acceptable),
    ("sample 1/run (2)", State.Acceptable),
    ("run_{Vol", State.Intermediate),
    ("a<b", State.Invalid),
    ("a|b", State.Invalid),
    ('a"b', State.Invalid),
    ("a\\b", State.Invalid),
))
def test_group_validator(widget, text, state):
    assert widget.group_input.validator().validate(text, len(text))[0] == state


def test_group_field_marks_unknown_placeholders(widget):
    validator = widget.group_input.validator()
    assert not widget.group_input.actions()
    validator.validate("{Nope}", 6)
    assert len(widget.group_input.actions()) == 1
    validator.validate("{Voltage}", 9)
    assert not widget.group_input.actions()


@pytest.mark.parametrize("text, state", (
    ("Add measurement {date} {time}", State.Acceptable),
    ("Cooldown at 4 K: sample S1 (rerun)", State.Acceptable),
    ("{Sample", State.Intermediate),
    ("stray } brace", State.Invalid),
))
def test_message_validator(widget, text, state):
    assert widget.message_input.validator().validate(text, len(text))[0] == state


@pytest.mark.parametrize("text, state", (
    ("main", State.Acceptable), ("dev-2", State.Acceptable),
    ("a b", State.Invalid), ("feature/x", State.Invalid),
))
def test_branch_validator(widget, text, state):
    assert widget.branch_input.validator().validate(text, len(text))[0] == state


def test_field_classes_are_reusable(qtbot):
    group, text = GroupPathLineEdit(_Procedure), PlaceholderLineEdit(_Procedure, "x")
    qtbot.addWidget(group)
    qtbot.addWidget(text)
    assert group.text() == "DATA" and text.text() == "x"


def test_importing_the_widgets_does_not_import_icechunk():
    code = ("import sys, pymeasure.display.widgets; "
            "assert not {'icechunk', 'zarr', 'xarray'} & set(sys.modules), sorted(sys.modules)")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


# ---- saving ----------------------------------------------------------------------------------

def test_save_writes_data_and_reports(widget, store, tmp_path, qtbot):
    from pymeasure.experiment.icechunk_store import load_group

    widget.store = str(store)
    with qtbot.waitSignal(widget.saved, timeout=5000) as blocker:
        result = widget.save(make_results(tmp_path))
    assert blocker.args == [result]
    assert result.group == "DATA_1" and result.branch == "main"
    assert widget.status_label.text() == (
        f"Saved as 'DATA_1' on branch 'main' ({result.snapshot_id[:8]})")
    assert widget.status_label.styleSheet() == ""
    assert load_group(store, "DATA_1")["Voltage (V)"].values.tolist() == [0.0, 0.5, 1.0]


def test_save_uses_all_fields(widget, store, tmp_path):
    widget.store = str(store)
    widget.group = "{Sample}/V{Voltage}"
    widget.commit_message = "Run of {Sample} at {Voltage} V"
    widget.metadata_text = "operator=me; sample={Sample}"
    result = widget.save(make_results(tmp_path))
    assert result.group == "S1/V2.5_1"
    commit = history(store)[0]
    assert commit.message == "Run of S1 at 2.5 V"
    assert commit.metadata["operator"] == "me" and commit.metadata["sample"] == "S1"
    assert commit.metadata["group"] == "S1/V2.5_1"


def test_save_to_another_branch(widget, store, tmp_path):
    widget.store = str(store)
    widget.branch = "dev"
    result = widget.save(make_results(tmp_path))
    assert result is not None and result.branch == "dev"


def test_saving_twice_does_not_overwrite(widget, store, tmp_path):
    widget.store = str(store)
    results = [widget.save(make_results(tmp_path)) for _ in range(2)]
    assert [r.group for r in results if r is not None] == ["DATA_1", "DATA_2"]


def test_storage_factory_turns_the_text_into_a_storage(qtbot, store, tmp_path):
    import icechunk

    storage = icechunk.in_memory_storage()
    seen = []

    def factory(text):
        seen.append(text)
        return storage

    widget = IcechunkInputWidget(_Procedure, storage_factory=factory)
    qtbot.addWidget(widget)
    widget.store = "  memory://runs "
    result = widget.save(make_results(tmp_path))
    assert result is not None and result.group == "DATA_1"
    assert seen == ["memory://runs"]                           # the text is stripped
    assert icechunk.Repository.exists(storage)


@pytest.mark.parametrize("setup, message", (
    (lambda w, store: None, "No icechunk store"),                              # no store set
    (lambda w, store: (setattr(w, "store", str(store)), setattr(w, "group", "{Nope}")),
     "Invalid placeholder"),
    (lambda w, store: (setattr(w, "store", str(store)), setattr(w, "metadata_text", "oops")),
     "key=value"),
    (lambda w, store: (setattr(w, "store", str(store)), setattr(w, "branch", "a b")),
     "Invalid branch name"),
    (lambda w, store: (setattr(w, "store", str(store)), setattr(w, "group", "a//b")),
     "Invalid group path"),
))
def test_failed_save_is_reported_not_raised(widget, store, tmp_path, qtbot, setup, message):
    setup(widget, store)
    with qtbot.waitSignal(widget.failed, timeout=5000) as blocker:
        assert widget.save(make_results(tmp_path)) is None
    assert message in blocker.args[0]
    assert widget.status_label.text().startswith("Not saved: ")
    assert "red" in widget.status_label.styleSheet()
    assert not store.exists()                                                   # nothing created


def test_a_successful_save_clears_the_error(widget, store, tmp_path):
    assert widget.save(make_results(tmp_path)) is None                          # no store yet
    widget.store = str(store)
    assert widget.save(make_results(tmp_path)) is not None
    assert widget.status_label.text().startswith("Saved as")
    assert not widget.status_label.styleSheet()


def test_a_save_without_data_fails_cleanly(widget, store, tmp_path, qtbot):
    widget.store = str(store)
    with qtbot.waitSignal(widget.failed, timeout=5000) as blocker:
        widget.save(make_results(tmp_path, rows=0))
    assert "no data" in blocker.args[0]


def test_error_status_shows_only_the_first_line(widget, tmp_path, monkeypatch):
    from pymeasure.display.widgets import icechunk_widget

    def explode(*args, **kwargs):
        raise OSError("first line\nsecond line\nthird line")

    monkeypatch.setattr(icechunk_widget, "save_results_to_icechunk", explode)
    widget.store = str(tmp_path / "store")
    widget.save(make_results(tmp_path))
    assert widget.status_label.text() == "Not saved: first line"


def test_experiment_finished_saves_when_enabled(widget, store, tmp_path):
    widget.store = str(store)

    class Experiment:
        results = make_results(tmp_path)

    widget.experiment_finished(Experiment)
    assert widget.status_label.text().startswith("Saved as 'DATA_1'")


def test_experiment_finished_does_nothing_when_disabled(widget, store, tmp_path):
    widget.store = str(store)
    widget.store_measurement = False

    class Experiment:
        results = make_results(tmp_path)

    widget.experiment_finished(Experiment)
    assert widget.status_label.text() == "" and not store.exists()


# ---- in a window --------------------------------------------------------------------------------

def test_saves_every_finished_experiment_of_a_managed_window(qtbot, store, tmp_path):
    from pymeasure.experiment.icechunk_store import load_group

    window = ManagedWindow(_Procedure, x_axis="Voltage (V)", y_axis="Current (A)")
    qtbot.addWidget(window)
    widget = IcechunkInputWidget(_Procedure, parent=window)
    dock = QtWidgets.QDockWidget("Icechunk store")
    dock.setWidget(widget)
    dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
    window.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    window.manager.finished.connect(widget.experiment_finished)
    widget.store = str(store)

    experiment = window.new_experiment(make_results(tmp_path))
    with qtbot.waitSignal(widget.saved, timeout=5000):
        window.manager.finished.emit(experiment)
    assert load_group(store, "DATA_1")["Current (A)"].size == 3
