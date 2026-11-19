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
import threading

import pytest

pytest.importorskip("zarr")
pytest.importorskip("xarray")

from pymeasure.display.Qt import QtWidgets
from pymeasure.display.widgets import ZarrInputWidget
from pymeasure.experiment import FloatParameter, Parameter, Procedure, Results
from pymeasure.experiment.zarr_store import ZarrResults, load_zarr_group


class _Procedure(Procedure):
    voltage = FloatParameter("Voltage", units="V", default=2.5)
    sample = Parameter("Sample", default="S1")

    DATA_COLUMNS = ["Voltage (V)", "Current (A)"]

    def execute(self):
        pass


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path_factory):
    """No default store from the environment or from a .env file of the machine."""
    for variable in ("PYMEASURE_ZARR_STORE", "PYMEASURE_ICECHUNK_STORE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


@pytest.fixture
def widget(qtbot):
    wdg = ZarrInputWidget(_Procedure)
    qtbot.addWidget(wdg)
    wdg.show()
    return wdg


@pytest.fixture
def store(tmp_path):
    return tmp_path / "store.zarr"


def make_results(tmp_path, rows=3):
    results = Results(_Procedure(), str(tmp_path / "data.csv"))
    with open(results.data_filename, "a", encoding="utf-8") as file:
        file.writelines(
            results.format({"Voltage (V)": 0.5 * i, "Current (A)": 1e-6 * i}) + "\n"
            for i in range(rows)
        )
    return results


def test_fields(widget):
    assert widget.store == "" and widget.group == "DATA" and widget.store_measurement is True
    assert widget._rows["Branch"][0].isHidden() and widget._rows["Branch"][1].isHidden()
    assert widget._rows["Commit message"][1].isHidden()
    assert not widget._rows["Group"][1].isHidden()
    assert not widget._rows["Extra metadata"][1].isHidden()
    assert widget.store_input.placeholderText() == "Directory of the zarr store"
    assert "zarr" in widget.save_toggle.toolTip()


def test_default_store_from_the_environment_or_a_dotenv_file(qtbot, monkeypatch, tmp_path):
    monkeypatch.setenv("PYMEASURE_ZARR_STORE", str(tmp_path / "env_store"))
    wdg = ZarrInputWidget(_Procedure)
    qtbot.addWidget(wdg)
    assert wdg.store == str(tmp_path / "env_store")
    monkeypatch.delenv("PYMEASURE_ZARR_STORE")
    (tmp_path / ".env").write_text("PYMEASURE_ZARR_STORE=lab/zarr\n")
    monkeypatch.chdir(tmp_path)
    other = ZarrInputWidget(_Procedure)
    qtbot.addWidget(other)
    assert other.store == str(tmp_path / "lab" / "zarr")


def test_the_icechunk_setting_is_not_used(qtbot, monkeypatch, tmp_path):
    monkeypatch.setenv("PYMEASURE_ICECHUNK_STORE", str(tmp_path / "icechunk"))
    wdg = ZarrInputWidget(_Procedure)
    qtbot.addWidget(wdg)
    assert wdg.store == ""


def test_new_results_use_the_fields(widget, store):
    widget.store = str(store)
    widget.group = "{Sample}/run"
    widget.metadata_text = "operator=me"
    results = widget.new_results(_Procedure())
    assert isinstance(results, ZarrResults)
    assert results.group == "S1/run_1" and results.extra_metadata == {"operator": "me"}
    assert widget.status_label.text() == "Recording in 'S1/run_1'"


def test_new_results_without_a_store(widget):
    failures = []
    widget.failed.connect(failures.append)
    assert widget.new_results(_Procedure()) is None
    assert "No zarr store is selected" in widget.status_label.text()
    assert widget.status_label.styleSheet() == "color: red"
    assert "PYMEASURE_ZARR_STORE" in failures[0]


def test_an_empty_field_falls_back_to_the_default(qtbot, monkeypatch, store):
    monkeypatch.setenv("PYMEASURE_ZARR_STORE", str(store))
    wdg = ZarrInputWidget(_Procedure)
    qtbot.addWidget(wdg)
    wdg.store = ""
    assert wdg.new_results(_Procedure()).store_name == str(store)


@pytest.mark.parametrize("group", ["{Nope}", "bad:name"])
def test_new_results_with_an_invalid_group(widget, store, group, caplog):
    failures = []
    widget.failed.connect(failures.append)
    widget.store = str(store)
    widget.group = group
    with caplog.at_level(logging.ERROR):
        assert widget.new_results(_Procedure()) is None
    assert "Not saved" in widget.status_label.text() and failures
    assert "Failed to prepare the zarr store" in caplog.text
    assert not store.exists()


def test_new_results_when_saving_is_off_use_memory(widget, store):
    widget.store = str(store)
    widget.store_measurement = False
    assert widget.new_results(_Procedure()).store_name == "in-memory"
    assert not store.exists()


def test_a_storage_factory(qtbot):
    import zarr
    seen = []

    def factory(text):
        seen.append(text)
        return zarr.storage.MemoryStore()

    wdg = ZarrInputWidget(_Procedure, storage_factory=factory)
    qtbot.addWidget(wdg)
    wdg.store = "s3://bucket/store"
    assert wdg.new_results(_Procedure()) is not None and seen == ["s3://bucket/store"]


def test_the_end_of_a_run_is_shown_from_the_worker_thread(widget, store):
    widget.store = str(store)
    results = widget.new_results(_Procedure())
    results.store({"Voltage (V)": 1.0, "Current (A)": 2.0})
    saved = []
    widget.saved.connect(saved.append)
    thread = threading.Thread(target=results.finish)
    thread.start()
    thread.join()
    QtWidgets.QApplication.processEvents()
    assert saved and saved[0].group == "DATA_1"
    assert widget.status_label.text() == "Saved as 'DATA_1'"


def test_a_failure_in_the_worker_thread_is_shown(widget, store):
    widget.store = str(store)
    results = widget.new_results(_Procedure())
    failures = []
    widget.failed.connect(failures.append)
    thread = threading.Thread(target=lambda: results.on_error(RuntimeError("disk full")))
    thread.start()
    thread.join()
    QtWidgets.QApplication.processEvents()
    assert failures == ["disk full"]
    assert widget.status_label.text() == "Not saved: disk full"


def test_save_appends_a_finished_experiment(widget, store, tmp_path):
    widget.store = str(store)
    widget.group = "{Sample}"
    widget.metadata_text = "operator=me"
    saved = []
    widget.saved.connect(saved.append)
    result = widget.save(make_results(tmp_path))
    assert result.group == "S1_1" and saved == [result]
    assert widget.status_label.text() == "Saved as 'S1_1'"
    dataset = load_zarr_group(store, "S1_1")
    assert dataset["Voltage (V)"].values.tolist() == [0.0, 0.5, 1.0]
    assert dataset.attrs["extra_metadata"] == {"operator": "me"}
    assert widget.save(make_results(tmp_path)).group == "S1_2"


def test_save_errors_are_shown_not_raised(widget, store, tmp_path, caplog):
    failures = []
    widget.failed.connect(failures.append)
    with caplog.at_level(logging.ERROR):
        assert widget.save(make_results(tmp_path)) is None  # no store
    assert "Failed to save the data to the zarr store" in caplog.text
    widget.store = str(store)
    widget.group = "a:b"
    assert widget.save(make_results(tmp_path)) is None
    assert len(failures) == 2 and widget.status_label.styleSheet() == "color: red"


class _Experiment:
    def __init__(self, results):
        self.results = results


def test_experiment_finished_saves_if_switched_on(widget, store, tmp_path):
    widget.store = str(store)
    widget.experiment_finished(_Experiment(make_results(tmp_path)))
    assert load_zarr_group(store, "DATA_1").sizes["record"] == 3
    widget.store_measurement = False
    widget.experiment_finished(_Experiment(make_results(tmp_path)))
    assert not widget.store_input.isEnabled()
    with pytest.raises(Exception):  # noqa: B017  (no second group)
        load_zarr_group(store, "DATA_2")
