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

import sys

import numpy as np
import pytest

icechunk = pytest.importorskip("icechunk", reason="needs Python >= 3.11 and pymeasure[icechunk]")
xarray = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

from pymeasure.experiment import (  # noqa: E402  # noqa: E402
    FloatParameter,
    Metadata,
    Parameter,
    Procedure,
    Results,
    icechunk_store,
    zarr_store,
)
from pymeasure.experiment.icechunk_store import (  # noqa: E402
    INSTALL_HINT,
    IcechunkSaveResult,
    load_group,
    parse_metadata_text,
    results_to_dataset,
    save_results_to_icechunk,
    validate_branch_name,
    validate_group_path,
)


class _Procedure(Procedure):
    voltage = FloatParameter("Voltage", units="V", default=2.5)
    sample = Parameter("Sample", default="S1")
    version = Metadata("Version", default="1.2")

    DATA_COLUMNS = ["Voltage (V)", "Current (A)", "Note"]

    def execute(self):
        pass


def make_results(tmp_path, rows=3, name="data.csv", procedure=None):
    results = Results(procedure or _Procedure(), str(tmp_path / name))
    with open(results.data_filename, "a", encoding="utf-8") as file:
        for i in range(rows):
            record = {"Voltage (V)": 0.5 * i, "Current (A)": 1e-6 * i, "Note": f"row {i}"}
            file.write(results.format(record) + "\n")
    return results


@pytest.fixture
def storage():
    return icechunk.in_memory_storage()


def history(storage, branch="main"):
    return list(icechunk.Repository.open(storage).ancestry(branch=branch))


def groups(storage, branch="main"):
    session = icechunk.Repository.open(storage).readonly_session(branch)
    try:
        return sorted(zarr.open_group(session.store, mode="r").group_keys())
    except zarr.errors.GroupNotFoundError:    # nothing has been written yet
        return []


# ---- what is stored ----------------------------------------------------------------------------

def test_round_trip_data_and_units(tmp_path, storage):
    results = make_results(tmp_path)
    saved = save_results_to_icechunk(results, storage)
    assert isinstance(saved, IcechunkSaveResult)
    assert (saved.branch, saved.group) == ("main", "DATA_1")
    dataset = load_group(storage, saved.group)
    assert dataset["Voltage (V)"].dims == ("record",)
    np.testing.assert_allclose(dataset["Voltage (V)"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(dataset["Current (A)"], [0.0, 1e-6, 2e-6])
    assert dataset["Note"].values.tolist() == ["row 0", "row 1", "row 2"]
    assert dataset["Voltage (V)"].attrs["units"] == "V"
    assert dataset["Current (A)"].attrs["units"] == "A"
    assert "units" not in dataset["Note"].attrs


def test_dataset_attributes_describe_the_run(tmp_path):
    results = make_results(tmp_path)
    attrs = results_to_dataset(results).attrs
    assert attrs["procedure"] == "_Procedure"
    assert attrs["procedure_module"] == _Procedure.__module__
    assert attrs["parameters"] == {"Voltage": 2.5, "Sample": "S1"}
    assert attrs["parameter_units"] == {"Voltage": "V"}
    assert attrs["metadata"] == {"Version": "1.2"}
    assert attrs["data_filename"] == str(tmp_path / "data.csv")
    assert attrs["pymeasure_version"] and attrs["saved_at"].endswith("+00:00")


def test_unset_parameter_is_stored_as_null(tmp_path):
    class Unset(_Procedure):
        threshold = FloatParameter("Threshold")             # no default: reading it raises

    attrs = results_to_dataset(make_results(tmp_path, procedure=Unset())).attrs
    assert attrs["parameters"]["Threshold"] is None
    assert attrs["parameters"]["Voltage"] == 2.5 and "Threshold" not in attrs["parameter_units"]


def test_nan_values_survive(tmp_path, storage):
    results = Results(_Procedure(), str(tmp_path / "nan.csv"))
    with open(results.data_filename, "a", encoding="utf-8") as file:
        file.write("1.0,nan,x\n2.0,3.0,y\n")
    saved = save_results_to_icechunk(results, storage)
    current = load_group(storage, saved.group)["Current (A)"].values
    assert np.isnan(current[0]) and current[1] == 3.0


def test_no_data_raises_before_touching_the_store(tmp_path, storage):
    results = make_results(tmp_path, rows=0)
    with pytest.raises(ValueError, match="no data"):
        save_results_to_icechunk(results, storage)
    assert not icechunk.Repository.exists(storage)


# ---- commit message, metadata, placeholders ----------------------------------------------------

def test_commit_message_and_metadata(tmp_path, storage):
    results = make_results(tmp_path)
    saved = save_results_to_icechunk(results, storage, message="Cooldown 3",
                                     metadata={"operator": "me", "group": "mine"})
    latest = history(storage)[0]
    assert latest.id == saved.snapshot_id and latest.message == "Cooldown 3"
    assert latest.metadata["operator"] == "me"
    assert latest.metadata["group"] == "mine"                 # user keys take precedence
    assert latest.metadata["procedure"] == "_Procedure"
    assert latest.metadata["pymeasure_version"]


def test_default_commit_metadata_names_the_group(tmp_path, storage):
    save_results_to_icechunk(make_results(tmp_path), storage)
    assert history(storage)[0].metadata["group"] == "DATA_1"


def test_placeholders_in_group_message_and_metadata(tmp_path, storage):
    results = make_results(tmp_path)
    saved = save_results_to_icechunk(
        results, storage, group="{Sample}/V{Voltage:.1f}", message="Run of {Sample} at {Voltage} V",
        metadata={"sample": "{Sample}", "when": "{date}"})
    assert saved.group == "S1/V2.5_1"
    commit = history(storage)[0]
    assert commit.message == "Run of S1 at 2.5 V"
    assert commit.metadata["sample"] == "S1" and "{" not in commit.metadata["when"]


@pytest.mark.parametrize("kwargs, where", (
    ({"group": "{Nope}"}, "group name"),
    ({"message": "{Nope}"}, "commit message"),
    ({"metadata": {"k": "{Nope}"}}, "metadata value of 'k'"),
    ({"group": "a{"}, "group name"),
    ({"message": "b}"}, "commit message"),
))
def test_invalid_placeholders_raise_and_write_nothing(tmp_path, storage, kwargs, where):
    with pytest.raises(ValueError, match=where):
        save_results_to_icechunk(make_results(tmp_path), storage, **kwargs)
    assert not icechunk.Repository.exists(storage)


# ---- group names -------------------------------------------------------------------------------

def test_group_index_makes_names_unique(tmp_path, storage):
    names = [save_results_to_icechunk(make_results(tmp_path, name=f"{i}.csv"), storage).group
             for i in range(3)]
    assert names == ["DATA_1", "DATA_2", "DATA_3"]
    assert groups(storage) == names
    assert len(history(storage)) == 4                          # initial commit + 3 saves


def test_nested_group_path(tmp_path, storage):
    saved = save_results_to_icechunk(make_results(tmp_path), storage, group="sample A/run")
    assert saved.group == "sample A/run_1"
    assert load_group(storage, "sample A/run_1")["Voltage (V)"].size == 3


def test_index_false_keeps_the_name_and_refuses_to_overwrite(tmp_path, storage):
    first = save_results_to_icechunk(make_results(tmp_path), storage, group="fixed", index=False)
    assert first.group == "fixed"
    before = history(storage)[0].id
    with pytest.raises(FileExistsError):
        save_results_to_icechunk(make_results(tmp_path, name="2.csv"), storage, group="fixed",
                                 index=False)
    assert history(storage)[0].id == before                    # nothing was committed


@pytest.mark.parametrize("path", ("", "/a", "a/", "a//b", ".", "..", "a/../b", "a\\b"))
def test_invalid_group_paths(tmp_path, storage, path):
    with pytest.raises(ValueError, match="group"):
        validate_group_path(path)
    with pytest.raises(ValueError):
        save_results_to_icechunk(make_results(tmp_path), storage, group=path)
    assert not icechunk.Repository.exists(storage)


def test_valid_group_paths():
    for path in ("DATA", "a/b", "sample 1/run (2)", "_x", "ünï"):
        assert validate_group_path(path) == path


# ---- branches ----------------------------------------------------------------------------------

def test_new_branch_is_created_from_main(tmp_path, storage):
    save_results_to_icechunk(make_results(tmp_path), storage)
    saved = save_results_to_icechunk(make_results(tmp_path, name="2.csv"), storage, branch="dev")
    assert saved.branch == "dev"
    repository = icechunk.Repository.open(storage)
    assert repository.list_branches() == {"main", "dev"}
    assert groups(storage, "main") == ["DATA_1"]               # main is untouched
    assert groups(storage, "dev") == ["DATA_1", "DATA_2"]      # dev started from main's data


def test_missing_branch_without_create_branch_raises(tmp_path, storage):
    with pytest.raises(ValueError, match="does not exist"):
        save_results_to_icechunk(make_results(tmp_path), storage, branch="dev",
                                 create_branch=False)
    assert groups(storage) == []


@pytest.mark.parametrize("branch", ("", "a b", "feature/x", " "))
def test_invalid_branch_names(tmp_path, storage, branch):
    with pytest.raises(ValueError, match="branch"):
        validate_branch_name(branch)
    with pytest.raises(ValueError, match="branch"):
        save_results_to_icechunk(make_results(tmp_path), storage, branch=branch)
    assert not icechunk.Repository.exists(storage)


# ---- stores ------------------------------------------------------------------------------------

def test_local_directory_store_is_created_and_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    saved = save_results_to_icechunk(make_results(tmp_path), "~/measurements/store")
    assert saved.store == str(tmp_path / "measurements" / "store")
    assert load_group(tmp_path / "measurements" / "store", saved.group)["Voltage (V)"].size == 3


def test_pathlib_path_store(tmp_path):
    store = tmp_path / "store"
    save_results_to_icechunk(make_results(tmp_path), store)
    save_results_to_icechunk(make_results(tmp_path, name="2.csv"), store)
    assert load_group(store, "DATA_2")["Voltage (V)"].size == 3


def test_create_false_requires_an_existing_repository(tmp_path, storage):
    with pytest.raises(FileNotFoundError, match="no icechunk repository"):
        save_results_to_icechunk(make_results(tmp_path), storage, create=False)
    assert not icechunk.Repository.exists(storage)
    save_results_to_icechunk(make_results(tmp_path), storage)
    save_results_to_icechunk(make_results(tmp_path, name="2.csv"), storage, create=False)
    assert groups(storage) == ["DATA_1", "DATA_2"]


# ---- concurrent writers ------------------------------------------------------------------------

class _RacingRepository:
    """Proxy for a repository: a rival writer commits between our opening a session and our
    commit, ``times`` times, to the group returned by ``rival_group(n)`` (n = 1, 2, ...)."""

    def __init__(self, repository, rival_group, times):
        self._repository, self._rival_group, self.times, self.count = (
            repository, rival_group, times, 0)

    def __getattr__(self, name):
        return getattr(self._repository, name)

    def writable_session(self, branch):
        session = self._repository.writable_session(branch)
        if self.count < self.times:
            self.count += 1
            rival = self._repository.writable_session(branch)
            dataset = xarray.Dataset({"x": ("i", [1.0])})
            dataset.to_zarr(rival.store, group=self._rival_group(self.count), mode="w-",
                            consolidated=False)
            rival.commit("rival")
        return session


def _race(monkeypatch, rival_group, times):
    real = icechunk_store._open_repository
    monkeypatch.setattr(icechunk_store, "_open_repository",
                        lambda storage, create: _RacingRepository(real(storage, create),
                                                                  rival_group, times))


def test_rival_writing_another_group_does_not_conflict(tmp_path, storage, monkeypatch):
    save_results_to_icechunk(make_results(tmp_path), storage)   # root group exists from here on
    _race(monkeypatch, lambda n: "OTHER", times=1)
    saved = save_results_to_icechunk(make_results(tmp_path, name="2.csv"), storage)
    assert saved.group == "DATA_2"
    assert groups(storage) == ["DATA_1", "DATA_2", "OTHER"]
    assert len(history(storage)) == 4                           # initial, first, rival, ours


def test_rival_taking_the_same_group_name_makes_us_retry(tmp_path, storage, monkeypatch):
    save_results_to_icechunk(make_results(tmp_path), storage)
    _race(monkeypatch, lambda n: "DATA_2", times=1)
    saved = save_results_to_icechunk(make_results(tmp_path, name="2.csv"), storage)
    assert saved.group == "DATA_3"
    assert groups(storage) == ["DATA_1", "DATA_2", "DATA_3"]
    assert load_group(storage, "DATA_3")["Voltage (V)"].size == 3   # ours, not the rival's


def test_gives_up_after_the_given_number_of_attempts(tmp_path, storage, monkeypatch):
    save_results_to_icechunk(make_results(tmp_path), storage)
    _race(monkeypatch, lambda n: f"DATA_{n + 1}", times=10)
    with pytest.raises(RuntimeError, match="after 3 attempts") as excinfo:
        save_results_to_icechunk(make_results(tmp_path, name="2.csv"), storage, attempts=3)
    assert isinstance(excinfo.value.__cause__, icechunk.RebaseFailedError)
    assert "DATA_2" in groups(storage)                          # only the rivals' groups


# ---- helpers -----------------------------------------------------------------------------------

def test_missing_dependencies_give_an_install_hint(tmp_path, storage, monkeypatch):
    monkeypatch.setitem(sys.modules, "icechunk", None)
    with pytest.raises(ImportError, match=r"pip install pymeasure\[icechunk\]") as excinfo:
        save_results_to_icechunk(make_results(tmp_path), storage)
    assert str(excinfo.value) == INSTALL_HINT


@pytest.mark.parametrize("text, expected", (
    ("", {}),
    ("   ", {}),
    ("a=1", {"a": "1"}),
    ("a=1; b = two words ;c=x=y", {"a": "1", "b": "two words", "c": "x=y"}),
    ("a=1\nb=2", {"a": "1", "b": "2"}),
    ("a=1;;a=2;", {"a": "2"}),
    ("empty=", {"empty": ""}),
))
def test_parse_metadata_text(text, expected):
    assert parse_metadata_text(text) == expected


@pytest.mark.parametrize("text", ("novalue", "=1", " = 1", "a=1; b"))
def test_parse_metadata_text_rejects_bad_entries(text):
    with pytest.raises(ValueError, match="key=value"):
        parse_metadata_text(text)


def test_jsonable():
    assert zarr_store._jsonable(np.float64(1.5)) == 1.5
    assert type(zarr_store._jsonable(np.int64(3))) is int
    assert zarr_store._jsonable(np.array([1, 2])) == [1, 2]
    assert zarr_store._jsonable((1, "a", None)) == [1, "a", None]
    assert zarr_store._jsonable({1: np.float32(2.0)}) == {"1": 2.0}
    assert zarr_store._jsonable(float("nan")) == "nan"
    assert zarr_store._jsonable(float("inf")) == "inf"
    assert zarr_store._jsonable(object).startswith("<class")
