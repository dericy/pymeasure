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
import time

import numpy as np
import pandas as pd
import pytest

icechunk = pytest.importorskip("icechunk", reason="needs Python >= 3.11 and pymeasure[icechunk]")
xarray = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

from pymeasure.experiment import (  # noqa: E402
    IntegerParameter, Metadata, Parameter, Procedure, Results,
)
from pymeasure.experiment import icechunk_store  # noqa: E402
from pymeasure.experiment.icechunk_store import (  # noqa: E402
    IcechunkRecorder,
    IcechunkResults,
    STORE_VARIABLE,
    IcechunkSaveResult,
    default_store,
    load_group,
)
from pymeasure.experiment.listeners import Recorder  # noqa: E402
from pymeasure.experiment.procedure import ProcedureStatus  # noqa: E402
from pymeasure.experiment.workers import Worker  # noqa: E402


class _Procedure(Procedure):
    iterations = IntegerParameter("Iterations", default=6)
    sample = Parameter("Sample", default="S1")
    version = Metadata("Version", default="1.2")

    DATA_COLUMNS = ["Iteration", "Voltage (V)", "Note"]

    def execute(self):
        for i in range(self.iterations):
            self.emit("results", {"Iteration": i, "Voltage (V)": 0.5 * i, "Note": f"row {i}"})
            time.sleep(0.02)


@pytest.fixture
def storage():
    return icechunk.in_memory_storage()


@pytest.fixture(autouse=True)
def no_reserved_groups():
    icechunk_store._RESERVED.clear()
    yield
    icechunk_store._RESERVED.clear()


def record(i):
    return {"Iteration": i, "Voltage (V)": 0.5 * i, "Note": f"row {i}"}


def history(storage, branch="main"):
    return list(icechunk.Repository.open(storage).ancestry(branch=branch))


def committed_groups(storage, branch="main"):
    session = icechunk.Repository.open(storage).readonly_session(branch)
    try:
        return sorted(zarr.open_group(session.store, mode="r").group_keys())
    except zarr.errors.GroupNotFoundError:
        return []


def finish_with(results, status=ProcedureStatus.FINISHED):
    results.procedure.status = status
    return results.finish()


class TestIcechunkResults:
    def test_data_is_read_from_the_store_before_any_commit(self, storage):
        results = IcechunkResults(_Procedure(), storage, group="run")
        assert len(results.data) == 0
        assert list(results.data.columns) == _Procedure.DATA_COLUMNS
        for i in range(3):
            results.store(record(i))
        data = results.data  # the records go to the open session, then are read back from it
        assert len(data) == 3
        assert data["Voltage (V)"].tolist() == [0.0, 0.5, 1.0]
        assert committed_groups(storage) == []  # nothing is committed yet
        result = finish_with(results)
        assert isinstance(result, IcechunkSaveResult)
        assert result.group == "run_1" and result.branch == "main"
        assert committed_groups(storage) == ["run_1"]

    def test_growing_data_keeps_earlier_values(self, storage):
        results = IcechunkResults(_Procedure(), storage)
        results.store(record(0))
        first = results.data.copy()
        results.store(record(1))
        results.store(record(2))
        data = results.data
        assert data.iloc[:1].equals(first)
        assert data["Note"].tolist() == ["row 0", "row 1", "row 2"]
        results.reload()
        assert results.data.equals(data)

    def test_what_is_saved(self, storage):
        results = IcechunkResults(
            _Procedure(), storage, group="{Sample}", message="Run of {Sample} ({Iterations})",
            metadata={"operator": "me", "sample": "{Sample}"}, branch="dev",
        )
        results.procedure.evaluate_metadata()
        results.store_metadata()
        for i in range(4):
            results.store(record(i))
        finish_with(results)
        dataset = load_group(storage, "S1_1", branch="dev")
        assert dict(dataset.sizes) == {"record": 4}
        assert dataset["Iteration"].dtype == np.int64
        assert dataset["Voltage (V)"].attrs["units"] == "V"
        assert dataset["Note"].values.tolist() == ["row 0", "row 1", "row 2", "row 3"]
        assert dataset.attrs["status"] == "Finished"
        assert dataset.attrs["procedure"] == "_Procedure"
        assert dataset.attrs["metadata"] == {"Version": "1.2"}
        assert dataset.attrs["parameters"]["Sample"] == "S1"
        commit = history(storage, "dev")[0]
        assert commit.message == "Run of S1 (6)"
        assert commit.metadata["status"] == "Finished"
        assert commit.metadata["operator"] == "me" and commit.metadata["sample"] == "S1"
        assert commit.metadata["group"] == "S1_1"

    @pytest.mark.parametrize("status", [ProcedureStatus.ABORTED, ProcedureStatus.FAILED])
    def test_partial_data_is_kept_with_the_status(self, storage, status):
        results = IcechunkResults(_Procedure(), storage)
        results.store(record(0))
        finish_with(results, status)
        assert load_group(storage, "DATA_1").attrs["status"] == str(status)
        assert history(storage)[0].metadata["status"] == str(status)

    def test_commits_while_running(self, storage, monkeypatch):
        monkeypatch.setattr(IcechunkResults, "FLUSH_INTERVAL", 0)
        results = IcechunkResults(_Procedure(), storage, commit_interval=0)
        for i in range(3):
            results.store(record(i))
            assert len(results.data) == i + 1  # also right after a commit
        finish_with(results)
        messages = [(c.message.endswith("(in progress)"), c.metadata.get("status"))
                    for c in history(storage)]
        assert messages[0] == (False, "Finished")
        assert (True, "Running") in messages
        assert load_group(storage, "DATA_1").sizes["record"] == 3
        assert len(results.data) == 3  # after the end, the data is still there

    def test_commit_interval_none_commits_only_at_the_end(self, storage, monkeypatch):
        monkeypatch.setattr(IcechunkResults, "FLUSH_INTERVAL", 0)
        results = IcechunkResults(_Procedure(), storage, commit_interval=None)
        for i in range(3):
            results.store(record(i))
        finish_with(results)
        assert len(history(storage)) == 2  # initialisation and the final commit

    def test_columns_widen_when_needed(self, storage, monkeypatch):
        monkeypatch.setattr(IcechunkResults, "FLUSH_INTERVAL", 0)
        results = IcechunkResults(_Procedure(), storage)
        results.store({"Iteration": 0, "Voltage (V)": 1, "Note": "a"})
        results.data  # noqa: B018  (written as integers)
        results.store({"Iteration": 1, "Voltage (V)": 1.5, "Note": 7})  # float, and a number as text
        results.store({"Iteration": 2, "Voltage (V)": True, "Note": "c"})
        data = results.data
        assert data["Voltage (V)"].tolist() == [1.0, 1.5, 1.0]
        assert data["Note"].tolist() == ["a", "7", "c"]
        finish_with(results)
        assert load_group(storage, "DATA_1")["Voltage (V)"].dtype == np.float64

    def test_a_column_without_units_widens(self, storage, monkeypatch):
        monkeypatch.setattr(IcechunkResults, "FLUSH_INTERVAL", 0)
        results = IcechunkResults(_Procedure(), storage)
        results.store({"Iteration": 0, "Voltage (V)": 1.0, "Note": "a"})
        results.store({"Iteration": 0.5, "Voltage (V)": 1.0, "Note": "b"})  # not a whole number
        assert results.data["Iteration"].tolist() == [0.0, 0.5]
        finish_with(results)
        assert "units" not in load_group(storage, "DATA_1")["Iteration"].attrs

    def test_reload_after_the_end_reads_the_committed_data(self, storage):
        results = IcechunkResults(_Procedure(), storage)
        for i in range(4):
            results.store(record(i))
        finish_with(results)
        before = results.data.copy()
        results.reload()
        assert results.data.equals(before) and len(before) == 4

    def test_a_boolean_column_widens_to_numbers(self, storage, monkeypatch):
        monkeypatch.setattr(IcechunkResults, "FLUSH_INTERVAL", 0)
        results = IcechunkResults(_Procedure(), storage)
        results.store({"Iteration": 0, "Voltage (V)": True, "Note": "a"})
        results.store({"Iteration": 1, "Voltage (V)": 3, "Note": "b"})
        assert results.data["Voltage (V)"].tolist() == [1, 3]

    def test_text_in_a_numeric_column_is_an_error_and_the_record_is_kept(self, storage,
                                                                         monkeypatch):
        monkeypatch.setattr(IcechunkResults, "FLUSH_INTERVAL", 0)
        errors = []
        results = IcechunkResults(_Procedure(), storage, on_error=errors.append)
        results.store(record(0))
        results.store({"Iteration": 1, "Voltage (V)": "high", "Note": "b"})
        assert len(errors) == 1 and isinstance(errors[0], TypeError)
        assert len(results.data) == 1  # the bad record stays in the buffer, nothing is lost

    def test_missing_columns_are_nan(self, storage):
        results = IcechunkResults(_Procedure(), storage)
        results.store({"Iteration": 0, "Note": "a"})
        results.store({"Iteration": 1, "Voltage (V)": 2.0, "Note": "b"})
        data = results.data
        assert np.isnan(data["Voltage (V)"].iloc[0]) and data["Voltage (V)"].iloc[1] == 2.0

    def test_groups_are_new_and_reserved(self, storage):
        first = IcechunkResults(_Procedure(), storage, group="g")
        second = IcechunkResults(_Procedure(), storage, group="g")  # queued meanwhile
        assert (first.group, second.group) == ("g_1", "g_2")
        for results in (second, first):
            results.store(record(0))
            finish_with(results)
        assert committed_groups(storage) == ["g_1", "g_2"]
        third = IcechunkResults(_Procedure(), storage, group="g")
        assert third.group == "g_3"
        assert third.data_filename.endswith("#g_3")

    def test_without_index_an_existing_group_is_an_error(self, storage):
        results = IcechunkResults(_Procedure(), storage, group="g", index=False)
        assert results.group == "g"
        results.store(record(0))
        finish_with(results)
        with pytest.raises(FileExistsError):
            IcechunkResults(_Procedure(), storage, group="g", index=False)

    def test_a_group_taken_by_another_writer_is_replaced(self, storage, caplog):
        results = IcechunkResults(_Procedure(), storage, group="g")
        icechunk_store._RESERVED.clear()  # as if another process chose the same name
        other = IcechunkResults(_Procedure(), storage, group="g")
        other.store(record(0))
        finish_with(other)
        assert other.group == "g_1" and results.group == "g_1"
        with caplog.at_level(logging.WARNING):
            results.store(record(5))
            finish_with(results)
        assert results.group == "g_2" and "taken" in caplog.text
        assert committed_groups(storage) == ["g_1", "g_2"]

    def test_the_taken_group_is_an_error_without_index(self, storage):
        results = IcechunkResults(_Procedure(), storage, group="g", index=False)
        icechunk_store._RESERVED.clear()
        repository = icechunk.Repository.open(storage)
        session = repository.writable_session("main")
        zarr.open_group(session.store, mode="a").create_group("g")
        session.commit("someone else")
        results.store(record(0))
        errors = []
        results.on_error = errors.append
        finish_with(results)
        assert isinstance(errors[0], FileExistsError)

    def test_nested_groups(self, storage):
        results = IcechunkResults(_Procedure(), storage, group="sample/{Sample}/run")
        results.store(record(0))
        finish_with(results)
        assert load_group(storage, "sample/S1/run_1").sizes["record"] == 1

    def test_a_store_in_memory(self):
        saved = []
        results = IcechunkResults(_Procedure(), None, on_commit=saved.append)
        results.store(record(0))
        assert len(results.data) == 1
        finish_with(results)
        assert saved[0].store == "in-memory" and saved[0].group == "DATA_1"

    def test_a_local_directory(self, tmp_path):
        results = IcechunkResults(_Procedure(), tmp_path / "store", group="run")
        results.store(record(0))
        finish_with(results)
        assert load_group(tmp_path / "store", "run_1").sizes["record"] == 1
        assert results.data_filename.startswith(str(tmp_path / "store"))

    def test_nothing_emitted_nothing_committed(self, storage):
        results = IcechunkResults(_Procedure(), storage)
        assert finish_with(results) is None
        assert committed_groups(storage) == []
        assert len(history(storage)) == 1

    def test_records_after_the_end_are_dropped(self, storage, caplog):
        results = IcechunkResults(_Procedure(), storage)
        results.store(record(0))
        finish_with(results)
        with caplog.at_level(logging.WARNING):
            results.store(record(1))
        assert "dropped" in caplog.text
        assert len(results.data) == 1
        assert results.finish() is results.save_result  # a second finish changes nothing

    def test_unknown_branches(self, storage):
        icechunk.Repository.create(storage)
        with pytest.raises(ValueError, match="does not exist"):
            IcechunkResults(_Procedure(), storage, branch="dev", create_branch=False)
        with pytest.raises(FileNotFoundError):
            IcechunkResults(_Procedure(), icechunk.in_memory_storage(), create=False)

    @pytest.mark.parametrize("kwargs, error", [
        ({"group": "{Nope}"}, ValueError),
        ({"message": "{Nope}"}, ValueError),
        ({"metadata": {"k": "{Nope}"}}, ValueError),
        ({"branch": "a b"}, ValueError),
        ({"group": "a//b"}, ValueError),
    ])
    def test_invalid_arguments(self, storage, kwargs, error):
        with pytest.raises(error):
            IcechunkResults(_Procedure(), storage, **kwargs)

    def test_a_procedure_is_required(self, storage):
        with pytest.raises(TypeError):
            IcechunkResults("not a procedure", storage)

    def test_repr(self, storage):
        assert "DATA_1" in repr(IcechunkResults(_Procedure(), storage))

    def test_metadata_is_not_written_without_metadata_objects(self, storage):
        class NoMeta(Procedure):
            DATA_COLUMNS = ["x"]
        results = IcechunkResults(NoMeta(), storage)
        results.store_metadata()  # does nothing, and does not create the group
        assert not results._created

    def test_a_failing_write_is_reported_and_keeps_the_records(self, storage, monkeypatch):
        errors = []
        results = IcechunkResults(_Procedure(), storage, on_error=errors.append)
        calls = []

        def failing(*_):
            calls.append(1)
            raise OSError("disk full")
        monkeypatch.setattr(results, "_ensure_open", failing)
        monkeypatch.setattr(IcechunkResults, "FLUSH_INTERVAL", 0)
        results.store(record(0))  # does not raise
        assert isinstance(errors[0], OSError)
        assert len(results._buffer) == 1
        len(results.data)  # reading does not raise either, and reports again
        assert len(errors) >= 2
        monkeypatch.undo()
        results.store(record(1))
        assert len(results.data) == 2  # the record was kept and is written now
        results.store_metadata()  # no metadata objects with values: nothing to do

    def test_failing_metadata_write_is_reported(self, storage, monkeypatch):
        errors = []
        results = IcechunkResults(_Procedure(), storage, on_error=errors.append)
        monkeypatch.setattr(results, "_ensure_open", lambda: (_ for _ in ()).throw(OSError("x")))
        results.store_metadata()
        assert isinstance(errors[0], OSError)

    def test_failing_commit_writes_a_csv_file(self, storage, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr(icechunk_store.tempfile, "tempdir", str(tmp_path))
        errors = []
        results = IcechunkResults(_Procedure(), storage, on_error=errors.append)
        for i in range(3):
            results.store(record(i))
        monkeypatch.setattr(results, "_commit", lambda final: (_ for _ in ()).throw(OSError("no")))
        with caplog.at_level(logging.ERROR):
            assert finish_with(results) is None
        assert isinstance(errors[0], OSError)
        (path,) = list(tmp_path.glob("pymeasure_icechunk_fallback_*.csv"))
        assert len(pd.read_csv(path)) == 3
        assert str(path) in caplog.text

    def test_unwritable_fallback_is_logged(self, storage, monkeypatch, caplog):
        results = IcechunkResults(_Procedure(), storage)
        results.store(record(0))
        monkeypatch.setattr(results, "_commit", lambda final: (_ for _ in ()).throw(OSError("no")))
        monkeypatch.setattr(icechunk_store.tempfile, "mkstemp",
                            lambda **_: (_ for _ in ()).throw(OSError("no space")))
        with caplog.at_level(logging.ERROR):
            assert finish_with(results) is None
        assert "fallback file" in caplog.text

    def test_the_fallback_includes_records_not_yet_written(self, storage, monkeypatch, tmp_path):
        monkeypatch.setattr(icechunk_store.tempfile, "tempdir", str(tmp_path))
        monkeypatch.setattr(IcechunkResults, "FLUSH_INTERVAL", 0)
        results = IcechunkResults(_Procedure(), storage)
        results.store(record(0))
        results.store(record(1))
        monkeypatch.setattr(results, "_flush", lambda: None)
        results._buffer.append(record(2))
        monkeypatch.setattr(results, "_commit", lambda final: (_ for _ in ()).throw(OSError("no")))
        finish_with(results)
        (path,) = list(tmp_path.glob("pymeasure_icechunk_fallback_*.csv"))
        assert len(pd.read_csv(path)) == 3

    def test_threads_can_write_and_read(self, storage):
        results = IcechunkResults(_Procedure(), storage, commit_interval=0.05)
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
        assert lengths == sorted(lengths)  # never goes backwards
        assert len(results.data) == 60
        assert load_group(storage, "DATA_1").sizes["record"] == 60


class TestWithWorker:
    def run(self, results):
        worker = Worker(results)
        worker.start()
        worker.join(timeout=30)
        assert not worker.is_alive()

    def test_make_recorder(self, storage):
        results = IcechunkResults(_Procedure(), storage)
        assert isinstance(results.make_recorder(None), IcechunkRecorder)

    def test_default_results_use_the_normal_recorder(self, tmp_path):
        results = Results(_Procedure(), str(tmp_path / "data.csv"))
        assert isinstance(results.make_recorder(None), Recorder)

    def test_an_experiment_in_the_store_without_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        saved = []
        results = IcechunkResults(_Procedure(), tmp_path / "store", group="{Sample}",
                                  on_commit=saved.append)
        self.run(results)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["store"]  # no CSV anywhere
        assert len(results.data) == 6
        assert results.data["Iteration"].tolist() == list(range(6))
        assert saved[0].group == "S1_1"
        dataset = load_group(tmp_path / "store", "S1_1")
        assert dataset.attrs["status"] == "Finished"
        assert dataset.attrs["metadata"] == {"Version": "1.2"}

    def test_an_aborted_experiment_keeps_the_data(self, storage):
        class Endless(_Procedure):
            def execute(self):
                for i in range(1000):
                    self.emit("results", record(i))
                    time.sleep(0.01)
                    if self.should_stop():
                        break

        results = IcechunkResults(Endless(), storage)
        worker = Worker(results)
        worker.start()
        deadline = time.time() + 20
        while len(results.data) < 5 and time.time() < deadline:
            time.sleep(0.02)
        worker.stop()
        threading.Thread.join(worker, timeout=30)  # Worker.join returns at once after a stop
        assert not worker.is_alive()
        assert load_group(storage, "DATA_1").attrs["status"] == "Aborted"
        assert load_group(storage, "DATA_1").sizes["record"] >= 5

    def test_a_failing_experiment_is_marked_failed(self, storage):
        class Failing(_Procedure):
            def execute(self):
                self.emit("results", record(0))
                raise RuntimeError("boom")

        results = IcechunkResults(Failing(), storage)
        self.run(results)
        assert load_group(storage, "DATA_1").attrs["status"] == "Failed"


class TestDefaultStore:
    @pytest.fixture(autouse=True)
    def clean_environment(self, monkeypatch):
        monkeypatch.delenv(STORE_VARIABLE, raising=False)

    def test_nothing_configured(self, tmp_path):
        assert default_store(tmp_path) is None

    def test_environment_variable(self, monkeypatch, tmp_path):
        monkeypatch.setenv(STORE_VARIABLE, "  /data/store  ")
        assert default_store(tmp_path) == "/data/store"

    def test_blank_environment_variable_is_ignored(self, monkeypatch, tmp_path):
        monkeypatch.setenv(STORE_VARIABLE, "   ")
        (tmp_path / ".env").write_text(f"{STORE_VARIABLE}=/from/env/file\n")
        assert default_store(tmp_path) == "/from/env/file"

    def test_environment_wins_over_the_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv(STORE_VARIABLE, "/from/environment")
        (tmp_path / ".env").write_text(f"{STORE_VARIABLE}=/from/env/file\n")
        assert default_store(tmp_path) == "/from/environment"

    def test_the_file_is_not_loaded_into_the_environment(self, tmp_path):
        import os
        (tmp_path / ".env").write_text(f"{STORE_VARIABLE}=/x\n")
        default_store(tmp_path)
        assert STORE_VARIABLE not in os.environ

    @pytest.mark.parametrize("line, expected", [
        ("PYMEASURE_ICECHUNK_STORE=/data/store", "/data/store"),
        ("PYMEASURE_ICECHUNK_STORE = /data/store", "/data/store"),
        ('PYMEASURE_ICECHUNK_STORE="/data/my store"', "/data/my store"),
        ("PYMEASURE_ICECHUNK_STORE='/data/my store'", "/data/my store"),
        ("export PYMEASURE_ICECHUNK_STORE=/data/store", "/data/store"),
        ("PYMEASURE_ICECHUNK_STORE=/data/store  # my lab store", "/data/store"),
        ('PYMEASURE_ICECHUNK_STORE="/data/#1"', "/data/#1"),
        ("PYMEASURE_ICECHUNK_STORE=s3://bucket/store", "s3://bucket/store"),
    ])
    def test_syntax(self, tmp_path, line, expected):
        (tmp_path / ".env").write_text(f"# comment\n\nOTHER=1\nnot an entry\n{line}\n")
        assert default_store(tmp_path) == expected

    def test_empty_entry_is_ignored(self, tmp_path):
        (tmp_path / ".env").write_text(f"{STORE_VARIABLE}=\n=novalue\n")
        assert default_store(tmp_path) is None

    def test_relative_paths_are_relative_to_the_file(self, tmp_path):
        (tmp_path / ".env").write_text(f"{STORE_VARIABLE}=data/store\n")
        sub = tmp_path / "work" / "deep"
        sub.mkdir(parents=True)
        assert default_store(sub) == str(tmp_path / "data/store")

    def test_home_directory_is_kept_for_later_expansion(self, tmp_path):
        (tmp_path / ".env").write_text(f"{STORE_VARIABLE}=~/store\n")
        assert default_store(tmp_path) == "~/store"

    def test_the_nearest_file_with_the_entry_wins(self, tmp_path):
        (tmp_path / ".env").write_text(f"{STORE_VARIABLE}=/outer\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / ".env").write_text("OTHER=1\n")  # no entry: the search goes on
        assert default_store(sub) == "/outer"
        (sub / ".env").write_text(f"{STORE_VARIABLE}=/inner\n")
        assert default_store(sub) == "/inner"

    def test_unreadable_files_are_ignored(self, tmp_path):
        (tmp_path / ".env").write_bytes(b"\xff\xfe\x00bad")
        assert default_store(tmp_path) is None
        (tmp_path / ".env").unlink()
        (tmp_path / ".env").mkdir()  # a directory called .env
        assert default_store(tmp_path) is None

    def test_default_start_is_the_working_directory(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text(f"{STORE_VARIABLE}=/cwd/store\n")
        monkeypatch.chdir(tmp_path)
        assert default_store() == "/cwd/store"

    def test_used_by_results(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text(f"{STORE_VARIABLE}=data/store\n")
        monkeypatch.chdir(tmp_path)
        results = IcechunkResults(_Procedure(), default_store())
        results.store(record(0))
        finish_with(results)
        assert load_group(tmp_path / "data" / "store", "DATA_1").sizes["record"] == 1


class TestDisplay:
    def test_default_results_show_the_file(self, tmp_path):
        results = Results(_Procedure(), str(tmp_path / "data.csv"))
        assert results.display_name == "data.csv"
        assert results.display_details == str(tmp_path / "data.csv")
        assert results.is_file is True

    def test_icechunk_results_show_the_commit_message_and_the_store(self, tmp_path):
        results = IcechunkResults(
            _Procedure(), tmp_path / "store", group="{Sample}", message="Run of {Sample}",
            branch="dev",
        )
        store = str(tmp_path / "store")
        assert results.display_name == f"Run of S1 \u2014 {store}#S1_1"
        details = results.display_details.splitlines()
        assert details == [f"Store: {store}", "Group: S1_1", "Branch: dev",
                           "Commit message: Run of S1"]
        assert results.is_file is False

    def test_a_store_in_memory(self):
        results = IcechunkResults(_Procedure(), None, message="Test")
        assert results.display_name == "Test \u2014 in-memory#DATA_1"

    def test_the_name_follows_a_group_that_had_to_change(self, storage):
        results = IcechunkResults(_Procedure(), storage, group="g", message="m")
        icechunk_store._RESERVED.clear()
        other = IcechunkResults(_Procedure(), storage, group="g")
        other.store(record(0))
        finish_with(other)
        results.store(record(1))
        finish_with(results)
        assert results.display_name.endswith("#g_2")
