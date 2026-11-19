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
import shutil
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest

zarr = pytest.importorskip("zarr")
xarray = pytest.importorskip("xarray")

from pymeasure.experiment import (  # noqa: E402
    IntegerParameter,
    Metadata,
    Parameter,
    Procedure,
    Results,
    zarr_store,
)
from pymeasure.experiment.procedure import ProcedureStatus  # noqa: E402
from pymeasure.experiment.workers import Worker  # noqa: E402
from pymeasure.experiment.zarr_store import (  # noqa: E402
    INSTALL_HINT,
    ZARR_STORE_VARIABLE,
    ZarrRecorder,
    ZarrResults,
    ZarrSaveResult,
    default_store,
    load_zarr_group,
    lookup_setting,
    save_results_to_zarr,
    validate_group_path,
)


class _Procedure(Procedure):
    iterations = IntegerParameter("Iterations", default=6)
    sample = Parameter("Sample", default="S1")
    version = Metadata("Version", default="1.2")

    DATA_COLUMNS = ["Iteration", "Voltage (V)", "Note"]

    def execute(self):
        for i in range(self.iterations):
            self.emit("results", {"Iteration": i, "Voltage (V)": 0.5 * i, "Note": f"row {i}"})
            time.sleep(0.02)


@pytest.fixture(autouse=True)
def no_reserved_groups():
    zarr_store._RESERVED.clear()
    yield
    zarr_store._RESERVED.clear()


@pytest.fixture
def store(tmp_path):
    return tmp_path / "store.zarr"


def record(i):
    return {"Iteration": i, "Voltage (V)": 0.5 * i, "Note": f"row {i}"}


def groups_in(store):
    root = zarr.open_group(str(store), mode="r")
    return sorted(name for name, member in root.members(max_depth=None)
                  if isinstance(member, zarr.Group) and "Iteration" in dict(member.members()))


def finish_with(results, status=ProcedureStatus.FINISHED):
    results.procedure.status = status
    return results.finish()


class TestZarrResults:
    def test_data_is_read_from_the_store_while_recording(self, store):
        results = ZarrResults(_Procedure(), store, group="run")
        assert len(results.data) == 0
        assert list(results.data.columns) == _Procedure.DATA_COLUMNS
        for i in range(3):
            results.store(record(i))
        data = results.data
        assert data["Voltage (V)"].tolist() == [0.0, 0.5, 1.0]
        # the data is in the directory already, and a different reader sees it
        live = zarr.open_group(zarr.storage.LocalStore(str(store)), mode="r")["run_1"]
        assert live["Voltage (V)"][:].tolist() == [0.0, 0.5, 1.0]
        assert live.attrs["status"] == "Running"
        result = finish_with(results)
        assert result == ZarrSaveResult("run_1", str(store))
        assert groups_in(store) == ["run_1"]
        assert load_zarr_group(store, "run_1").attrs["status"] == "Finished"

    def test_growing_data_keeps_earlier_values(self, store):
        results = ZarrResults(_Procedure(), store)
        results.store(record(0))
        first = results.data.copy()
        results.store(record(1))
        results.store(record(2))
        data = results.data
        assert data.iloc[:1].equals(first)
        assert data["Note"].tolist() == ["row 0", "row 1", "row 2"]
        results.reload()
        assert results.data.equals(data)

    def test_what_is_saved(self, store):
        results = ZarrResults(
            _Procedure(), store, group="{Sample}", metadata={"operator": "me", "s": "{Sample}"})
        results.procedure.evaluate_metadata()
        results.store_metadata()
        for i in range(4):
            results.store(record(i))
        finish_with(results)
        dataset = load_zarr_group(store, "S1_1")
        assert dict(dataset.sizes) == {"record": 4}
        assert dataset["Iteration"].dtype == np.int64
        assert dataset["Voltage (V)"].attrs["units"] == "V"
        assert dataset["Note"].values.tolist() == ["row 0", "row 1", "row 2", "row 3"]
        assert dataset.attrs["status"] == "Finished"
        assert dataset.attrs["procedure"] == "_Procedure"
        assert dataset.attrs["metadata"] == {"Version": "1.2"}
        assert dataset.attrs["extra_metadata"] == {"operator": "me", "s": "S1"}
        assert dataset.attrs["parameters"]["Sample"] == "S1"

    def test_no_extra_metadata_attribute_without_metadata(self, store):
        results = ZarrResults(_Procedure(), store)
        results.store(record(0))
        finish_with(results)
        assert "extra_metadata" not in load_zarr_group(store, "DATA_1").attrs

    @pytest.mark.parametrize("status", [ProcedureStatus.ABORTED, ProcedureStatus.FAILED])
    def test_partial_data_is_kept_with_the_status(self, store, status):
        results = ZarrResults(_Procedure(), store)
        results.store(record(0))
        finish_with(results, status)
        assert load_zarr_group(store, "DATA_1").attrs["status"] == str(status)

    def test_groups_are_appended_to_the_store(self, store):
        for sample in ("a", "b", "a"):
            results = ZarrResults(_Procedure(), store, group=sample)
            results.store(record(0))
            finish_with(results)
        assert groups_in(store) == ["a_1", "a_2", "b_1"]
        assert load_zarr_group(store, "a_1").sizes["record"] == 1

    def test_a_commit_interval_changes_nothing(self, store, monkeypatch):
        monkeypatch.setattr(ZarrResults, "FLUSH_INTERVAL", 0)
        results = ZarrResults(_Procedure(), store, commit_interval=0)
        for i in range(3):
            results.store(record(i))
            assert len(results.data) == i + 1
        finish_with(results)
        assert load_zarr_group(store, "DATA_1").sizes["record"] == 3

    def test_columns_widen_when_needed(self, store, monkeypatch):
        monkeypatch.setattr(ZarrResults, "FLUSH_INTERVAL", 0)
        results = ZarrResults(_Procedure(), store)
        results.store({"Iteration": 0, "Voltage (V)": 1, "Note": "a"})
        results.data  # noqa: B018  (written as integers)
        results.store({"Iteration": 1, "Voltage (V)": 1.5, "Note": 7})
        results.store({"Iteration": 2, "Voltage (V)": True, "Note": "c"})
        data = results.data
        assert data["Voltage (V)"].tolist() == [1.0, 1.5, 1.0]
        assert data["Note"].tolist() == ["a", "7", "c"]
        finish_with(results)
        dataset = load_zarr_group(store, "DATA_1")
        assert dataset["Voltage (V)"].dtype == np.float64
        assert dataset["Voltage (V)"].attrs["units"] == "V"

    def test_a_column_without_units_widens(self, store, monkeypatch):
        monkeypatch.setattr(ZarrResults, "FLUSH_INTERVAL", 0)
        results = ZarrResults(_Procedure(), store)
        results.store({"Iteration": 0, "Voltage (V)": 1.0, "Note": "a"})
        results.store({"Iteration": 0.5, "Voltage (V)": 1.0, "Note": "b"})
        assert results.data["Iteration"].tolist() == [0.0, 0.5]
        finish_with(results)
        assert "units" not in load_zarr_group(store, "DATA_1")["Iteration"].attrs

    def test_a_boolean_column_widens_to_numbers(self, store, monkeypatch):
        monkeypatch.setattr(ZarrResults, "FLUSH_INTERVAL", 0)
        results = ZarrResults(_Procedure(), store)
        results.store({"Iteration": 0, "Voltage (V)": True, "Note": "a"})
        results.store({"Iteration": 1, "Voltage (V)": 3, "Note": "b"})
        assert results.data["Voltage (V)"].tolist() == [1, 3]

    def test_text_in_a_numeric_column_is_an_error_and_the_record_is_kept(self, store, monkeypatch):
        monkeypatch.setattr(ZarrResults, "FLUSH_INTERVAL", 0)
        errors = []
        results = ZarrResults(_Procedure(), store, on_error=errors.append)
        results.store(record(0))
        results.store({"Iteration": 1, "Voltage (V)": "high", "Note": "b"})
        assert len(errors) == 1 and isinstance(errors[0], TypeError)
        assert len(results.data) == 1

    def test_missing_columns_are_nan(self, store):
        results = ZarrResults(_Procedure(), store)
        results.store({"Iteration": 0, "Note": "a"})
        results.store({"Iteration": 1, "Voltage (V)": 2.0, "Note": "b"})
        data = results.data
        assert np.isnan(data["Voltage (V)"].iloc[0]) and data["Voltage (V)"].iloc[1] == 2.0

    def test_groups_are_new_and_reserved(self, store):
        first = ZarrResults(_Procedure(), store, group="g")
        second = ZarrResults(_Procedure(), store, group="g")  # queued meanwhile
        assert (first.group, second.group) == ("g_1", "g_2")
        for results in (second, first):
            results.store(record(0))
            finish_with(results)
        assert groups_in(store) == ["g_1", "g_2"]
        third = ZarrResults(_Procedure(), store, group="g")
        assert third.group == "g_3" and third.data_filename.endswith("#g_3")

    def test_without_index_an_existing_group_is_an_error(self, store):
        results = ZarrResults(_Procedure(), store, group="g", index=False)
        assert results.group == "g"
        results.store(record(0))
        finish_with(results)
        with pytest.raises(FileExistsError):
            ZarrResults(_Procedure(), store, group="g", index=False)

    def test_a_group_taken_by_another_writer_is_replaced(self, store, caplog):
        results = ZarrResults(_Procedure(), store, group="g")
        zarr_store._RESERVED.clear()  # as if another process chose the same name
        other = ZarrResults(_Procedure(), store, group="g")
        other.store(record(0))
        finish_with(other)
        assert other.group == "g_1" and results.group == "g_1"
        with caplog.at_level(logging.WARNING):
            results.store(record(5))
            finish_with(results)
        assert results.group == "g_2" and "taken" in caplog.text
        assert groups_in(store) == ["g_1", "g_2"]

    def test_the_taken_group_is_an_error_without_index(self, store):
        results = ZarrResults(_Procedure(), store, group="g", index=False)
        zarr_store._RESERVED.clear()
        zarr.open_group(str(store), mode="a").create_group("g")  # someone else
        results.store(record(0))
        errors = []
        results.on_error = errors.append
        finish_with(results)
        assert isinstance(errors[0], FileExistsError)

    def test_nested_groups(self, store):
        results = ZarrResults(_Procedure(), store, group="sample/{Sample}/run")
        results.store(record(0))
        finish_with(results)
        assert load_zarr_group(store, "sample/S1/run_1").sizes["record"] == 1

    def test_a_store_in_memory(self):
        saved = []
        results = ZarrResults(_Procedure(), None, on_commit=saved.append)
        results.store(record(0))
        assert len(results.data) == 1
        finish_with(results)
        assert saved[0] == ZarrSaveResult("DATA_1", "in-memory")

    def test_a_zarr_store_object(self):
        memory = zarr.storage.MemoryStore()
        results = ZarrResults(_Procedure(), memory, group="x")
        results.store(record(0))
        finish_with(results)
        assert "x_1" in zarr.open_group(memory, mode="r")
        assert load_zarr_group(memory, "x_1").sizes["record"] == 1
        assert results.store_name == repr(memory)

    def test_the_directory_is_created_or_not(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no zarr store"):
            ZarrResults(_Procedure(), tmp_path / "missing", create=False)
        assert not (tmp_path / "missing").exists()
        results = ZarrResults(_Procedure(), tmp_path / "deep" / "new" / "store")
        results.store(record(0))
        finish_with(results)
        assert (tmp_path / "deep" / "new" / "store" / "zarr.json").is_file()
        again = ZarrResults(_Procedure(), tmp_path / "deep" / "new" / "store", create=False)
        assert again.group == "DATA_2"

    def test_only_the_store_directory_is_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        results = ZarrResults(_Procedure(), "store.zarr")
        results.store(record(0))
        finish_with(results)
        assert [p.name for p in tmp_path.iterdir()] == ["store.zarr"]

    def test_a_copy_of_the_directory_is_a_store(self, tmp_path):
        results = ZarrResults(_Procedure(), tmp_path / "a.zarr", group="run")
        for i in range(5):
            results.store(record(i))
        finish_with(results)
        shutil.copytree(tmp_path / "a.zarr", tmp_path / "elsewhere" / "b.zarr")
        assert load_zarr_group(tmp_path / "elsewhere" / "b.zarr", "run_1").sizes["record"] == 5
        more = ZarrResults(_Procedure(), tmp_path / "elsewhere" / "b.zarr", group="run")
        assert more.group == "run_2"  # and more can be appended to the copy

    def test_nothing_emitted_nothing_created(self, store):
        results = ZarrResults(_Procedure(), store)
        assert finish_with(results) is None
        assert not store.exists()
        assert results._read_root() is None

    def test_records_after_the_end_are_dropped(self, store, caplog):
        results = ZarrResults(_Procedure(), store)
        results.store(record(0))
        finish_with(results)
        with caplog.at_level(logging.WARNING):
            results.store(record(1))
        assert "dropped" in caplog.text
        assert len(results.data) == 1
        assert results.finish() is results.save_result

    def test_reload_after_the_end_reads_the_store(self, store):
        results = ZarrResults(_Procedure(), store)
        for i in range(4):
            results.store(record(i))
        finish_with(results)
        before = results.data.copy()
        results.reload()
        assert results.data.equals(before) and len(before) == 4

    @pytest.mark.parametrize("kwargs, error", [
        ({"group": "{Nope}"}, ValueError),
        ({"metadata": {"k": "{Nope}"}}, ValueError),
        ({"group": "a//b"}, ValueError),
        ({"group": "a:b"}, ValueError),  # not a valid directory name on Windows
        ({"group": "CON"}, ValueError),
        ({"group": "run."}, ValueError),
        ({"group": "good/bad?"}, ValueError),
    ])
    def test_invalid_arguments(self, store, kwargs, error):
        with pytest.raises(error):
            ZarrResults(_Procedure(), store, **kwargs)
        assert not store.exists()

    def test_a_procedure_is_required(self, store):
        with pytest.raises(TypeError):
            ZarrResults("not a procedure", store)

    def test_the_missing_packages_are_reported(self, store, monkeypatch):
        monkeypatch.setitem(sys.modules, "zarr", None)
        with pytest.raises(ImportError, match="pymeasure\\[zarr\\]"):
            ZarrResults(_Procedure(), store)
        assert "zarr" in INSTALL_HINT

    def test_display(self, store):
        results = ZarrResults(_Procedure(), store, group="g")
        assert results.display_name == f"{store}#g_1"
        assert results.display_details == f"Store: {store}\nGroup: g_1"
        assert results.is_file is False
        assert "g_1" in repr(results)

    def test_metadata_is_not_written_without_metadata_objects(self, store):
        class NoMeta(Procedure):
            DATA_COLUMNS = ["x"]
        results = ZarrResults(NoMeta(), store)
        results.store_metadata()
        assert not results._created

    def test_a_failing_write_is_reported_and_keeps_the_records(self, store, monkeypatch):
        errors = []
        results = ZarrResults(_Procedure(), store, on_error=errors.append)

        def failing(*_):
            raise OSError("disk full")
        monkeypatch.setattr(results, "_ensure_open", failing)
        monkeypatch.setattr(ZarrResults, "FLUSH_INTERVAL", 0)
        results.store(record(0))
        assert isinstance(errors[0], OSError) and len(results._buffer) == 1
        len(results.data)
        assert len(errors) >= 2
        monkeypatch.undo()
        results.store(record(1))
        assert len(results.data) == 2

    def test_a_failing_metadata_write_is_reported(self, store, monkeypatch):
        errors = []
        results = ZarrResults(_Procedure(), store, on_error=errors.append)
        monkeypatch.setattr(results, "_ensure_open", lambda: (_ for _ in ()).throw(OSError("x")))
        results.store_metadata()
        assert isinstance(errors[0], OSError)

    def test_failing_to_finish_writes_a_csv_file(self, store, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr(zarr_store.tempfile, "tempdir", str(tmp_path))
        errors = []
        results = ZarrResults(_Procedure(), store, on_error=errors.append)
        for i in range(3):
            results.store(record(i))
        monkeypatch.setattr(results, "_commit", lambda final: (_ for _ in ()).throw(OSError("no")))
        with caplog.at_level(logging.ERROR):
            assert finish_with(results) is None
        assert isinstance(errors[0], OSError)
        (path,) = list(tmp_path.glob("pymeasure_zarr_fallback_*.csv"))
        assert len(pd.read_csv(path)) == 3 and str(path) in caplog.text

    def test_unwritable_fallback_is_logged(self, store, monkeypatch, caplog):
        results = ZarrResults(_Procedure(), store)
        results.store(record(0))
        monkeypatch.setattr(results, "_commit", lambda final: (_ for _ in ()).throw(OSError("no")))
        monkeypatch.setattr(zarr_store.tempfile, "mkstemp",
                            lambda **_: (_ for _ in ()).throw(OSError("no space")))
        with caplog.at_level(logging.ERROR):
            assert finish_with(results) is None
        assert "fallback file" in caplog.text

    def test_the_fallback_includes_records_not_yet_written(self, store, monkeypatch, tmp_path):
        monkeypatch.setattr(zarr_store.tempfile, "tempdir", str(tmp_path))
        monkeypatch.setattr(ZarrResults, "FLUSH_INTERVAL", 0)
        results = ZarrResults(_Procedure(), store)
        results.store(record(0))
        results.store(record(1))
        monkeypatch.setattr(results, "_flush", lambda: None)
        results._buffer.append(record(2))
        monkeypatch.setattr(results, "_commit", lambda final: (_ for _ in ()).throw(OSError("no")))
        finish_with(results)
        (path,) = list(tmp_path.glob("pymeasure_zarr_fallback_*.csv"))
        assert len(pd.read_csv(path)) == 3

    def test_threads_can_write_and_read(self, store):
        results = ZarrResults(_Procedure(), store)
        stop = threading.Event()
        lengths = []

        def reader():
            while not stop.is_set():
                lengths.append(len(results.data))
                time.sleep(0.005)
        thread = threading.Thread(target=reader)
        thread.start()
        for i in range(60):
            results.store(record(i))
            time.sleep(0.005)
        stop.set()
        thread.join()
        finish_with(results)
        assert lengths == sorted(lengths)
        assert len(results.data) == 60
        assert load_zarr_group(store, "DATA_1").sizes["record"] == 60


class TestWithWorker:
    def run(self, results):
        worker = Worker(results)
        worker.start()
        worker.join(timeout=30)
        threading.Thread.join(worker, timeout=30)
        assert not worker.is_alive()

    def test_make_recorder(self, store):
        assert isinstance(ZarrResults(_Procedure(), store).make_recorder(None), ZarrRecorder)

    def test_an_experiment_in_the_store_without_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        saved = []
        results = ZarrResults(_Procedure(), tmp_path / "store.zarr", group="{Sample}",
                              on_commit=saved.append)
        self.run(results)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["store.zarr"]
        assert results.data["Iteration"].tolist() == list(range(6))
        assert saved[0].group == "S1_1"
        dataset = load_zarr_group(tmp_path / "store.zarr", "S1_1")
        assert dataset.attrs["status"] == "Finished"
        assert dataset.attrs["metadata"] == {"Version": "1.2"}

    def test_an_aborted_experiment_keeps_the_data(self, store):
        class Endless(_Procedure):
            def execute(self):
                for i in range(1000):
                    self.emit("results", record(i))
                    time.sleep(0.01)
                    if self.should_stop():
                        break

        results = ZarrResults(Endless(), store)
        worker = Worker(results)
        worker.start()
        deadline = time.time() + 20
        while len(results.data) < 5 and time.time() < deadline:
            time.sleep(0.02)
        worker.stop()
        threading.Thread.join(worker, timeout=30)
        assert load_zarr_group(store, "DATA_1").attrs["status"] == "Aborted"
        assert load_zarr_group(store, "DATA_1").sizes["record"] >= 5

    def test_a_failing_experiment_is_marked_failed(self, store):
        class Failing(_Procedure):
            def execute(self):
                self.emit("results", record(0))
                raise RuntimeError("boom")

        self.run(ZarrResults(Failing(), store))
        assert load_zarr_group(store, "DATA_1").attrs["status"] == "Failed"


class TestSaveFinishedResults:
    @pytest.fixture
    def csv_results(self, tmp_path):
        results = Results(_Procedure(), str(tmp_path / "data.csv"))
        with open(results.data_filename, "a", encoding="utf-8") as file:
            file.writelines(results.format(record(i)) + "\n" for i in range(3))
        return results

    def test_a_csv_run_is_appended_as_a_group(self, csv_results, store):
        first = save_results_to_zarr(csv_results, store, group="{Sample}",
                                     metadata={"operator": "me"})
        second = save_results_to_zarr(csv_results, store, group="{Sample}")
        assert (first.group, second.group) == ("S1_1", "S1_2")
        assert first.store == str(store)
        dataset = load_zarr_group(store, "S1_1")
        assert dataset["Voltage (V)"].values.tolist() == [0.0, 0.5, 1.0]
        assert dataset["Voltage (V)"].attrs["units"] == "V"
        assert dataset.attrs["extra_metadata"] == {"operator": "me"}
        assert dataset.attrs["status"] == str(csv_results.procedure.status)
        assert "extra_metadata" not in load_zarr_group(store, "S1_2").attrs

    def test_without_index(self, csv_results, store):
        save_results_to_zarr(csv_results, store, group="g", index=False)
        with pytest.raises(FileExistsError):
            save_results_to_zarr(csv_results, store, group="g", index=False)

    def test_invalid_arguments_write_nothing(self, csv_results, store):
        for group in ("{Nope}", "a:b"):
            with pytest.raises(ValueError):
                save_results_to_zarr(csv_results, store, group=group)
        with pytest.raises(ValueError):
            save_results_to_zarr(csv_results, store, metadata={"k": "{Nope}"})
        empty = Results(_Procedure(), str(store.parent / "empty.csv"))
        with pytest.raises(ValueError, match="no data"):
            save_results_to_zarr(empty, store)
        assert not store.exists()

    def test_other_stores(self, csv_results, tmp_path):
        memory = zarr.storage.MemoryStore()
        assert save_results_to_zarr(csv_results, memory).group == "DATA_1"
        assert save_results_to_zarr(csv_results, None).store == "in-memory"
        with pytest.raises(FileNotFoundError):
            save_results_to_zarr(csv_results, tmp_path / "missing", create=False)

    def test_the_missing_packages_are_reported(self, csv_results, store, monkeypatch):
        monkeypatch.setitem(sys.modules, "xarray", None)
        with pytest.raises(ImportError, match="pymeasure\\[zarr\\]"):
            save_results_to_zarr(csv_results, store)


class TestGroupNames:
    @pytest.mark.parametrize("name", ["DATA", "a/b/c", "Sample 1", "x_1.5", "ünï", "COM10", "con_"])
    def test_portable_names(self, name):
        assert validate_group_path(name, portable=True) == name

    @pytest.mark.parametrize("name", [
        "a:b", "a<b", 'a"b', "a|b", "a?b", "a*b", "a\x01b", "CON", "nul", "Com1", "lpt9.txt",
        "end ", "end.", "ok/bad:", "ok/aux/x",
    ])
    def test_names_that_are_not_valid_on_windows(self, name):
        validate_group_path(name)  # fine for an object store
        with pytest.raises(ValueError, match="Windows"):
            validate_group_path(name, portable=True)

    @pytest.mark.parametrize("name", ["", "/a", "a/", "a//b", ".", "a/..", "a\\b"])
    def test_names_that_are_never_valid(self, name):
        for portable in (False, True):
            with pytest.raises(ValueError, match="Invalid group path"):
                validate_group_path(name, portable=portable)


class TestDefaultStore:
    @pytest.fixture(autouse=True)
    def clean_environment(self, monkeypatch):
        monkeypatch.delenv(ZARR_STORE_VARIABLE, raising=False)

    def test_nothing_configured(self, tmp_path):
        assert default_store(tmp_path) is None

    def test_environment_variable_and_file(self, monkeypatch, tmp_path):
        (tmp_path / ".env").write_text(f"{ZARR_STORE_VARIABLE}=data/zarr_store\n")
        (tmp_path / "sub").mkdir()
        assert default_store(tmp_path / "sub") == str(tmp_path / "data/zarr_store")
        monkeypatch.setenv(ZARR_STORE_VARIABLE, "/from/environment")
        assert default_store(tmp_path) == "/from/environment"

    def test_it_is_a_different_setting_from_the_icechunk_store(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PYMEASURE_ICECHUNK_STORE", "/icechunk")
        assert default_store(tmp_path) is None
        assert lookup_setting("PYMEASURE_ICECHUNK_STORE", tmp_path) == "/icechunk"

    def test_default_start_is_the_working_directory(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text(f"{ZARR_STORE_VARIABLE}=/cwd/store\n")
        monkeypatch.chdir(tmp_path)
        assert default_store() == "/cwd/store"
