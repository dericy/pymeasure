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
import time

import pytest

pytest.importorskip("zarr")
pytest.importorskip("xarray")

from pymeasure.display.Qt import QtCore, QtWidgets
from pymeasure.display.widgets import ZarrInputWidget
from pymeasure.display.windows.managed_window import ManagedWindow
from pymeasure.experiment import IntegerParameter, Procedure
from pymeasure.experiment.zarr_store import load_zarr_group


class _Procedure(Procedure):
    iterations = IntegerParameter("Iterations", default=8)
    DATA_COLUMNS = ["Iteration", "Voltage (V)"]

    def execute(self):
        for i in range(self.iterations):
            self.emit("results", {"Iteration": i, "Voltage (V)": 0.5 * i})
            time.sleep(0.03)


class _Window(ManagedWindow):
    def __init__(self, store):
        super().__init__(
            _Procedure, inputs=["iterations"], x_axis="Iteration", y_axis="Voltage (V)",
            enable_file_input=False,
        )
        self.zarr_input = ZarrInputWidget(_Procedure, parent=self)
        self.zarr_input.store = str(store)

    def queue(self, procedure=None):
        if procedure is None:
            procedure = self.make_procedure()
        results = self.zarr_input.new_results(procedure)
        if results is None:
            return
        self.manager.queue(self.new_experiment(results))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.delenv("PYMEASURE_ZARR_STORE", raising=False)
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)


def test_plot_and_store_without_csv(qtbot, tmp_path):
    store = tmp_path / "store.zarr"
    window = _Window(store)
    qtbot.addWidget(window)
    window.show()
    shown = []
    with qtbot.waitSignal(window.zarr_input.saved, timeout=60000):
        window.queue()
        deadline = time.time() + 60
        while window.zarr_input.status_label.text().startswith("Recording"):
            QtWidgets.QApplication.processEvents()
            for experiment in window.manager.experiments:
                for curve in filter(None, experiment.curve_list):
                    x = curve.getData()[0]
                    shown.append(0 if x is None else len(x))
            time.sleep(0.02)
            if time.time() > deadline:
                break
    experiment = window.manager.experiments[0]
    experiment.curve_list[0].update_data()
    assert len(experiment.curve_list[0].getData()[0]) == 8
    assert max(shown) >= 1  # the plot was filled while the experiment ran
    assert sorted(p.name for p in tmp_path.iterdir()) == ["store.zarr"]  # no CSV anywhere
    dataset = load_zarr_group(store, "DATA_1")
    assert dataset["Voltage (V)"].values.tolist() == [0.5 * i for i in range(8)]
    assert dataset.attrs["status"] == "Finished"
    item = window.browser.topLevelItem(0)
    assert item.text(1) == f"{store}#DATA_1"
    assert item.toolTip(1).splitlines() == [f"Store: {store}", "Group: DATA_1"]


def test_queued_experiments_are_appended_to_one_store(qtbot, tmp_path):
    store = tmp_path / "store.zarr"
    window = _Window(store)
    qtbot.addWidget(window)
    saved = []
    window.zarr_input.saved.connect(saved.append)
    window.queue()
    window.queue()  # queued while the first runs: the groups are different
    qtbot.waitUntil(lambda: len(saved) == 2, timeout=60000)
    assert sorted(r.group for r in saved) == ["DATA_1", "DATA_2"]
    assert all(load_zarr_group(store, g).sizes["record"] == 8 for g in ("DATA_1", "DATA_2"))


def test_queue_without_a_store_does_nothing(qtbot, tmp_path):
    window = _Window(tmp_path / "store.zarr")
    qtbot.addWidget(window)
    window.zarr_input.store = ""
    window.queue()
    assert not list(window.manager.experiments)
    assert "No zarr store" in window.zarr_input.status_label.text()


def test_the_file_menu_actions_are_disabled(qtbot, tmp_path, monkeypatch):
    from pymeasure.display.windows import managed_window
    window = _Window(tmp_path / "store.zarr")
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.zarr_input.saved, timeout=60000):
        window.queue()
    item = window.browser.topLevelItem(0)
    seen = {}

    class RecordingMenu(QtWidgets.QMenu):
        def exec(self, *args, **kwargs):
            seen.update({a.text(): a.isEnabled() for a in self.actions() if a.text()})

    monkeypatch.setattr(managed_window.QtWidgets, "QMenu", RecordingMenu)
    monkeypatch.setattr(window.browser, "itemAt", lambda pos: item)
    window.browser.setCurrentItem(item)
    window.browser_item_menu(QtCore.QPoint(0, 0))
    assert seen["Open Data Externally"] is False and seen["Save Data File Copy"] is False
