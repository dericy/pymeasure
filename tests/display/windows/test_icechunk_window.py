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

pytest.importorskip("icechunk", reason="needs Python >= 3.11 and pymeasure[icechunk]")
pytest.importorskip("xarray")
pytest.importorskip("zarr")

from pymeasure.display.Qt import QtWidgets  # noqa: E402
from pymeasure.display.widgets import IcechunkInputWidget  # noqa: E402
from pymeasure.display.windows.managed_window import ManagedWindow  # noqa: E402
from pymeasure.experiment import IntegerParameter, Procedure  # noqa: E402
from pymeasure.experiment.icechunk_store import load_group  # noqa: E402


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
        self.icechunk_input = IcechunkInputWidget(_Procedure, parent=self)
        self.icechunk_input.store = str(store)

    def queue(self, procedure=None):
        if procedure is None:
            procedure = self.make_procedure()
        results = self.icechunk_input.new_results(procedure)
        if results is None:
            return
        self.manager.queue(self.new_experiment(results))


@pytest.fixture(autouse=True)
def _clean_root_logging_handlers():
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)


def test_plot_and_store_without_csv(qtbot, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYMEASURE_ICECHUNK_STORE", raising=False)
    store = tmp_path / "store"
    window = _Window(store)
    qtbot.addWidget(window)
    window.show()
    shown = []  # points of the curve, read while the experiment runs
    with qtbot.waitSignal(window.icechunk_input.saved, timeout=60000):
        window.queue()
        deadline = time.time() + 60
        while not window.manager.experiments.has_next() and time.time() < deadline:
            QtWidgets.QApplication.processEvents()
        while window.icechunk_input.status_label.text().startswith("Recording"):
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
    assert sorted(p.name for p in tmp_path.iterdir()) == ["store"]  # no CSV anywhere
    dataset = load_group(store, "DATA_1")
    assert dataset["Voltage (V)"].values.tolist() == [0.5 * i for i in range(8)]
    assert dataset.attrs["status"] == "Finished"
    assert window.icechunk_input.status_label.text().startswith("Saved as 'DATA_1'")


def test_queue_without_a_store_does_nothing(qtbot, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYMEASURE_ICECHUNK_STORE", raising=False)
    window = _Window(tmp_path / "store")
    qtbot.addWidget(window)
    window.icechunk_input.store = ""
    window.queue()
    assert not list(window.manager.experiments)
    assert "No icechunk store" in window.icechunk_input.status_label.text()
