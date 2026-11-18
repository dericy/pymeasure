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

"""Save the data of an experiment to an `icechunk <https://icechunk.io>`_ store.

Icechunk is a transactional storage engine for Zarr data: every save is a *commit* with a
message and metadata on a *branch*, so the history of a store of measurements is versioned.

The optional dependencies (``icechunk``, ``zarr``, ``xarray``; Python 3.11 or newer) are only
imported when a save is requested: install them with ``pip install pymeasure[icechunk]``.
"""

import logging
import math
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ..log import QueueListener
from .procedure import Procedure
from .results import CSVFormatter, Results, replace_placeholders

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

#: Name of the environment variable (or entry of a ``.env`` file) with the default store.
STORE_VARIABLE = "PYMEASURE_ICECHUNK_STORE"

INSTALL_HINT = (
    "Saving to an icechunk store requires the optional packages 'icechunk', 'zarr' and "
    "'xarray' (Python 3.11 or newer): install them with 'pip install pymeasure[icechunk]'."
)


class IcechunkSaveResult(NamedTuple):
    """Where a save ended up."""

    snapshot_id: str  #: Id of the commit that was created.
    branch: str  #: Branch that the commit was made on.
    group: str  #: Path of the group in the store that holds the data.
    store: str  #: Description of the store (the path for a local store).


def _read_env_file(path: Path) -> dict[str, str]:
    """Read the ``KEY=value`` entries of a ``.env`` file (comments, ``export`` and quotes are
    understood; there is no variable expansion)."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()  # trailing comment
        if key:
            values[key] = value
    return values


def default_store(start: "str | os.PathLike | None" = None) -> str | None:
    """Return the default icechunk store, or ``None`` if none is configured.

    It is the value of the environment variable ``PYMEASURE_ICECHUNK_STORE``; if that is not set,
    the entry ``PYMEASURE_ICECHUNK_STORE=...`` of the first ``.env`` file found in ``start``
    (default: the current directory) or its parent directories. A relative path in a ``.env``
    file is relative to the directory of that file (a value with ``://``, such as an S3 URL, is
    returned as it is). The ``.env`` file is only read, ``os.environ`` is not changed.

    :param start: Directory where the search for a ``.env`` file begins.
    """
    value = os.environ.get(STORE_VARIABLE, "").strip()
    if value:
        return value
    directory = Path(start if start is not None else Path.cwd()).expanduser().resolve()
    for folder in (directory, *directory.parents):
        value = _read_env_file(folder / ".env").get(STORE_VARIABLE, "").strip()
        if value:
            if "://" in value or Path(value).expanduser().is_absolute():
                return value
            return str(folder / value)
    return None


def _import_icechunk():
    try:
        import icechunk
        import xarray
        import zarr
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return icechunk, xarray, zarr


def validate_group_path(path: str) -> str:
    """Check that ``path`` is usable as the path of a zarr group and return it.

    Groups are separated by ``/``. No part may be empty, ``.`` or ``..``, and ``\\`` is not
    allowed. Raises :class:`ValueError` otherwise.
    """
    if not path or any(part in ("", ".", "..") for part in path.split("/")) or "\\" in path:
        raise ValueError(
            f"Invalid group path '{path}': use names separated by '/', without empty parts, "
            "'.', '..' or backslashes."
        )
    return path


def validate_branch_name(branch: str) -> str:
    """Check that ``branch`` is a usable icechunk branch name and return it.

    Icechunk rejects ``/`` in branch names but accepts an empty name, which would be
    unusable, so this is checked here. Raises :class:`ValueError` otherwise.
    """
    if not branch or re.search(r"[\s/]", branch):
        raise ValueError(
            f"Invalid branch name '{branch}': it must not be empty or contain spaces or '/'."
        )
    return branch


def parse_metadata_text(text: str) -> dict[str, str]:
    """Parse ``"key=value; key2=value2"`` (entries separated by ``;`` or newlines).

    Keys and values are stripped; the value is everything after the first ``=``. Raises
    :class:`ValueError` for an entry without a key or without ``=``.
    """
    metadata: dict[str, str] = {}
    for entry in re.split(r"[;\n]", text):
        entry = entry.strip()
        if not entry:
            continue
        key, separator, value = entry.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"Invalid metadata entry '{entry}': use 'key=value'.")
        metadata[key] = value.strip()
    return metadata


def _jsonable(value: Any) -> Any:
    """Convert ``value`` to something that can be stored as a zarr attribute (JSON)."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return str(value)


def _fill_placeholders(text: str, procedure: Procedure, what: str) -> str:
    try:
        return replace_placeholders(text, procedure)
    except KeyError as exc:
        raise ValueError(f"Invalid placeholder in the {what}: {exc.args[0]}") from None
    except ValueError as exc:
        raise ValueError(f"Invalid placeholder syntax in the {what}: {exc}") from None


def _run_attributes(results: Results) -> dict[str, Any]:
    """Attributes that describe a run: procedure, parameters, metadata, data file, version."""
    from pymeasure import __version__

    procedure = results.procedure
    parameters, parameter_units = {}, {}
    for parameter in procedure.parameter_objects().values():
        try:
            parameters[parameter.name] = _jsonable(parameter.value)
        except ValueError:  # value not set
            parameters[parameter.name] = None
        units = getattr(parameter, "units", None)
        if units:
            parameter_units[parameter.name] = str(units)
    metadata = {m.name: _jsonable(m.value) for m in procedure.metadata_objects().values()}
    return {
        "pymeasure_version": __version__,
        "procedure": type(procedure).__name__,
        "procedure_module": type(procedure).__module__,
        "parameters": parameters,
        "parameter_units": parameter_units,
        "metadata": metadata,
        "data_filename": str(results.data_filename),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }


def results_to_dataset(results: Results):
    """Convert the data of ``results`` to an :class:`xarray.Dataset`.

    Each data column becomes a variable along the dimension ``record``. Columns with a unit in
    their header (``"Voltage (V)"``) get the attribute ``units``. The attributes of the dataset
    describe the run: the procedure, its parameters and metadata (by display name, with their
    units in ``parameter_units``), the data file, the pymeasure version and the time of saving.
    """
    _, xarray, _ = _import_icechunk()

    data = results.data
    if len(data) == 0:
        raise ValueError("The results contain no data.")
    dataset = xarray.Dataset.from_dataframe(data.rename_axis("record"))
    for column, unit in Procedure.parse_columns([str(c) for c in data.columns]).items():
        dataset[column].attrs["units"] = f"{unit:~}"

    dataset.attrs.update(_run_attributes(results))
    return dataset


def _open_repository(storage, create: bool):
    icechunk, _, _ = _import_icechunk()
    if icechunk.Repository.exists(storage):
        return icechunk.Repository.open(storage)
    if not create:
        raise FileNotFoundError(
            "There is no icechunk repository at this store (use create=True to create one)."
        )
    return icechunk.Repository.create(storage)


def _ensure_branch(repository, branch: str, create_branch: bool) -> None:
    branches = repository.list_branches()
    if branch not in branches:
        if not create_branch or "main" not in branches:
            raise ValueError(f"The branch '{branch}' does not exist in the store.")
        repository.create_branch(branch, repository.lookup_branch("main"))


def _unique_group(root, base: str, index: bool) -> str:
    """Return ``base`` with a ``_#`` suffix that makes it new in ``root`` (like filenames)."""
    if not index:
        return base
    i = 1
    while f"{base}_{i}" in root:
        i += 1
    return f"{base}_{i}"


def save_results_to_icechunk(
    results: Results,
    store: "str | os.PathLike | Any",
    group: str = "DATA",
    message: str = "Add measurement {date} {time}",
    branch: str = "main",
    metadata: Mapping[str, str] | None = None,
    index: bool = True,
    create: bool = True,
    create_branch: bool = True,
    attempts: int = 5,
) -> IcechunkSaveResult:
    """Write the data of ``results`` to an icechunk store and commit it.

    The data is written as a zarr group (see :func:`results_to_dataset`) and committed in one
    transaction. Writes of other processes to *other* groups do not get in the way; if another
    writer used the same group name in the meantime, the save is retried with a new name.

    ``group``, ``message`` and the values of ``metadata`` may contain placeholders for the
    parameters and metadata of the procedure and for ``{date}`` and ``{time}``, exactly as the
    filename in the graphical interface (see :func:`.replace_placeholders`).

    :param results: Results to save.
    :param store: Local directory of the store, or an ``icechunk.Storage`` (for example for
        S3 or an in-memory store). A local directory is created and initialised if needed
        (see ``create``). Note that icechunk's local-filesystem storage is not safe for
        several processes committing at the same time.
    :param group: Template of the group path, groups separated by ``/``.
    :param message: Template of the commit message.
    :param branch: Branch to commit to (no spaces or ``/``).
    :param metadata: Template values of extra commit metadata. The group, the procedure and
        the pymeasure version are always added (your keys take precedence).
    :param index: Add a ``_#`` suffix to the group name so that it is new. With ``False``, an
        existing group raises :class:`FileExistsError`.
    :param create: Create the repository if there is none at ``store``.
    :param create_branch: Create ``branch`` from the tip of ``main`` if it does not exist.
    :param attempts: Number of tries if other writers conflict.
    :raises ValueError: For invalid names or placeholders, or if there is no data.
    :raises ImportError: If the optional dependencies are not installed.
    :returns: Where the data was saved.
    """
    icechunk, _, zarr = _import_icechunk()
    from pymeasure import __version__

    procedure = results.procedure
    base = validate_group_path(_fill_placeholders(group, procedure, "group name"))
    commit_message = _fill_placeholders(message, procedure, "commit message")
    extra = {
        key: _fill_placeholders(value, procedure, f"metadata value of '{key}'")
        for key, value in (metadata or {}).items()
    }
    validate_branch_name(branch)
    dataset = results_to_dataset(results)  # raises if there is no data, before touching the store

    storage, store_name = _storage_for(store)
    repository = _open_repository(storage, create)

    _ensure_branch(repository, branch, create_branch)

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        session = repository.writable_session(branch)
        name = _unique_group(zarr.open_group(session.store, mode="a"), base, index)
        dataset.to_zarr(session.store, group=name, mode="w-", consolidated=False)
        commit_metadata = {
            "group": name,
            "procedure": type(procedure).__name__,
            "pymeasure_version": __version__,
            **extra,
        }
        try:
            snapshot_id = session.commit(
                commit_message,
                metadata=commit_metadata,
                rebase_with=icechunk.ConflictDetector(),
            )
        except (icechunk.ConflictError, icechunk.RebaseFailedError) as exc:
            last_error = exc
            log.info(f"Conflict while committing '{name}' (attempt {attempt}), retrying")
            continue
        log.info(f"Saved the results as '{name}' on branch '{branch}' ({snapshot_id})")
        return IcechunkSaveResult(snapshot_id, branch, name, store_name)
    raise RuntimeError(
        f"Could not commit to the icechunk store after {attempts} attempts because of "
        "conflicting writes by other processes."
    ) from last_error


# ---- live storage: the icechunk store instead of a CSV file ------------------------------------

_RESERVED: set[tuple[str, str, str]] = set()  # groups promised to queued results of this process
_RESERVED_LOCK = threading.Lock()


def _storage_for(store) -> tuple[Any, str]:
    """Return ``(icechunk storage, description)`` for the path of a local store, an
    ``icechunk.Storage`` or ``None`` (a store in memory that is lost with the process)."""
    icechunk, _, _ = _import_icechunk()
    if store is None:
        return icechunk.in_memory_storage(), "in-memory"
    if isinstance(store, (str, os.PathLike)):
        path = os.path.abspath(os.path.expanduser(os.fspath(store)))
        return icechunk.local_filesystem_storage(path), path
    return store, repr(store)


def _normalized(values) -> np.ndarray:
    """Array of one of the dtypes of a stored column: bool, int64, float64 or text."""
    array = np.asarray(values)
    if array.dtype.kind == "b":
        return array
    if array.dtype.kind in "iu":
        return array.astype(np.int64)
    if array.dtype.kind == "f":
        return array.astype(np.float64)
    return array.astype(str)


def _is_text(dtype) -> bool:
    return np.dtype(dtype).kind not in "biuf"


class IcechunkResults(Results):
    """Results that live in an `icechunk <https://icechunk.io>`_ store instead of a CSV file.

    While the procedure runs, the records that it emits are appended to the arrays of one group
    of the store (one array per data column, along the dimension ``record``, like
    :func:`results_to_dataset`), and :attr:`data` -- which the curves, images and tables of the
    graphical interface read -- is read back from the store. Nothing is written to a file.

    The data is committed every ``commit_interval`` seconds (so that other processes can follow
    the run) and when the run ends; the final commit has the status of the run in its metadata
    (``status``: ``Finished``, ``Aborted`` or ``Failed``) and in the attributes of the group.
    Records are buffered and written in batches (at most every :attr:`FLUSH_INTERVAL` seconds,
    and whenever :attr:`data` is read), so that emitting a record is cheap.
    If the final commit fails, the data is written to a temporary CSV file, whose name is logged,
    so that a measurement is not lost.

    Use it where a :class:`.Results` object is created, for example in the ``queue`` method of a
    window (see :meth:`.IcechunkInputWidget.new_results`)::

        results = IcechunkResults(procedure, "path/to/store", group="{Sample name}")
        experiment = self.new_experiment(results)
        self.manager.queue(experiment)

    The group is chosen when the results are created (an index ``_#`` is added so that it is
    new, as for a filename), so that several queued experiments get different groups.

    :param procedure: Procedure of the experiment.
    :param store: Local directory of the store, an ``icechunk.Storage``, or ``None`` for a store
        in memory (the data can be plotted but is lost with the process).
    :param group: Template of the group path (placeholders as for the filename).
    :param message: Template of the commit message.
    :param branch: Branch to commit to.
    :param metadata: Template values of extra commit metadata.
    :param index: Add a ``_#`` suffix to the group so that it is new. With ``False``, an existing
        group raises :class:`FileExistsError`.
    :param create: Create the repository if there is none at ``store``.
    :param create_branch: Create ``branch`` from the tip of ``main`` if it does not exist.
    :param commit_interval: Seconds between commits while running (``None``: only at the end).
    :param on_commit: Called with the :class:`IcechunkSaveResult` of the final commit (from the
        thread of the worker).
    :param on_error: Called with the exception if writing to the store failed (from the thread
        that notices the failure).
    """

    #: Longest time (in seconds) that records stay in memory before they are written.
    FLUSH_INTERVAL = 0.5
    #: Chunk length of the arrays.
    CHUNK_LENGTH = 4096

    def __init__(
        self,
        procedure: Procedure,
        store=None,
        group: str = "DATA",
        message: str = "Add measurement {date} {time}",
        branch: str = "main",
        metadata: Mapping[str, str] | None = None,
        index: bool = True,
        create: bool = True,
        create_branch: bool = True,
        commit_interval: float | None = 10.0,
        on_commit: Callable[[IcechunkSaveResult], Any] | None = None,
        on_error: Callable[[Exception], Any] | None = None,
    ):
        if not isinstance(procedure, Procedure):
            raise TypeError("Results require a Procedure object")
        _import_icechunk()  # fail early if the optional packages are missing

        # what a Results object has (see Results.__init__), but no files
        self.procedure = procedure
        self.procedure_class = procedure.__class__
        self.parameters = procedure.parameter_objects()
        self._header_count = -1
        self._metadata_count = -1
        self._last_file_size = 0
        self.formatter = CSVFormatter(columns=procedure.DATA_COLUMNS)
        self.data_filenames = []
        self._data = None

        base = validate_group_path(_fill_placeholders(group, procedure, "group name"))
        self.message = _fill_placeholders(message, procedure, "commit message")
        self.extra_metadata = {
            key: _fill_placeholders(value, procedure, f"metadata value of '{key}'")
            for key, value in (metadata or {}).items()
        }
        self.branch = validate_branch_name(branch)
        self.index = index
        self.commit_interval = commit_interval
        self.on_commit = on_commit
        self.on_error = on_error

        storage, self.store_name = _storage_for(store)
        self._repository = _open_repository(storage, create)
        _ensure_branch(self._repository, self.branch, create_branch)
        self._base = base
        self.group = self._reserve(base)
        self.data_filename = f"{self.store_name}#{self.group}"

        self._lock = threading.RLock()
        self._buffer: list[Mapping[str, Any]] = []
        self._n_flushed = 0  # records in the store
        self._session = None  # writable session while there are uncommitted changes
        self._group = None
        self._arrays: dict[str, Any] = {}
        self._created = False  # the group exists (in a session or a commit)
        self._units = {
            column: f"{unit:~}"
            for column, unit in Procedure.parse_columns(list(procedure.DATA_COLUMNS)).items()
        }
        self._last_flush = self._last_commit = time.monotonic()
        self._finished = False
        self.save_result: IcechunkSaveResult | None = None

    # ---- names ---------------------------------------------------------------------------------

    def _reserve(self, base: str) -> str:
        _, _, zarr = _import_icechunk()
        session = self._repository.readonly_session(self.branch)
        try:
            root = zarr.open_group(session.store, mode="r")
        except zarr.errors.GroupNotFoundError:  # nothing was ever saved to this store
            root = ()
        with _RESERVED_LOCK:
            def taken(name):
                return name in root or (self.store_name, self.branch, name) in _RESERVED

            if not self.index:
                if taken(base):
                    raise FileExistsError(f"The group '{base}' exists already in the store.")
                name = base
            else:
                i = 1
                while taken(f"{base}_{i}"):
                    i += 1
                name = f"{base}_{i}"
            _RESERVED.add((self.store_name, self.branch, name))
        return name

    def _release(self) -> None:
        with _RESERVED_LOCK:
            _RESERVED.discard((self.store_name, self.branch, self.group))

    # ---- writing -------------------------------------------------------------------------------

    def _ensure_open(self) -> None:
        """Make sure that there is a writable session with the group and its arrays."""
        if self._session is not None:
            return
        _, _, zarr = _import_icechunk()
        session = self._repository.writable_session(self.branch)
        root = zarr.open_group(session.store, mode="a")
        if not self._created:
            if self.group in root:  # another process took the name in the meantime
                if not self.index:
                    raise FileExistsError(f"The group '{self.group}' exists already.")
                self._release()
                self.group = self._reserve(self._base)
                self.data_filename = f"{self.store_name}#{self.group}"
                log.warning(f"The planned group was taken, using '{self.group}' instead")
            group = root.create_group(self.group)
            group.attrs.update(_run_attributes(self))
            group.attrs["status"] = "Running"
            self._arrays = {}
            self._created = True
        else:
            group = root[self.group]
            self._arrays = {name: group[name] for name in group.array_keys()}
        self._session, self._group = session, group

    def _append(self, name: str, batch: np.ndarray, start: int) -> None:
        """Write ``batch`` to the column ``name`` from record ``start`` on (idempotent)."""
        array = self._arrays.get(name)
        batch = _normalized(batch)
        if array is None:
            array = self._group.create_array(
                name, shape=(start,), chunks=(self.CHUNK_LENGTH,),
                dtype=str if _is_text(batch.dtype) else batch.dtype, dimension_names=("record",),
            )
            if name in self._units:
                array.attrs["units"] = self._units[name]
            self._arrays[name] = array
        else:
            stored = np.dtype(array.dtype)
            if _is_text(stored):
                batch = batch.astype(str)
            elif _is_text(batch.dtype):
                raise TypeError(f"The column '{name}' has numbers, but a record has text.")
            elif stored.kind == "b" and batch.dtype.kind != "b" or (
                    stored.kind == "i" and batch.dtype.kind == "f"):
                array = self._widen(name, array, batch.dtype)  # the column needs a wider type
        if array.shape[0] != start + len(batch):
            array.resize((start + len(batch),))
        array[start:] = batch

    def _widen(self, name: str, array, dtype):
        old = array[:]
        del self._group[name]
        new = self._group.create_array(
            name, shape=(len(old),), chunks=(self.CHUNK_LENGTH,), dtype=dtype,
            dimension_names=("record",),
        )
        if name in self._units:
            new.attrs["units"] = self._units[name]
        new[:] = old.astype(dtype)
        self._arrays[name] = new
        return new

    def _flush(self) -> None:
        """Write the buffered records to the arrays of the session."""
        self._last_flush = time.monotonic()
        if not self._buffer:
            return
        rows = self._buffer
        self._ensure_open()
        for column in self.procedure.DATA_COLUMNS:
            values = [row.get(column, np.nan) for row in rows]
            self._append(column, np.asarray(values), self._n_flushed)
        self._n_flushed += len(rows)
        self._buffer = []

    def store(self, record: Mapping[str, Any]) -> None:
        """Add a record (a dictionary with a value for each data column) to the data."""
        with self._lock:
            if self._finished:
                log.warning("A record arrived after the results were finished, it is dropped.")
                return
            self._buffer.append(record)
            now = time.monotonic()
            try:
                if now - self._last_flush >= self.FLUSH_INTERVAL:
                    self._flush()
                if (self.commit_interval is not None
                        and now - self._last_commit >= self.commit_interval):
                    self._commit(final=False)
            except Exception as exc:  # noqa: BLE001  the records stay in the buffer
                self._failed(exc, "write")

    def _failed(self, exc: Exception, what: str) -> None:
        log.error(f"Could not {what} the data in the icechunk store: {exc!r}")
        if self.on_error is not None:
            self.on_error(exc)

    def store_metadata(self) -> None:
        """Store the metadata of the procedure (evaluated after its start-up) in the group."""
        with self._lock:
            if not self.procedure.metadata_objects():
                return
            try:
                self._ensure_open()
                self._group.attrs["metadata"] = _run_attributes(self)["metadata"]
            except Exception as exc:  # noqa: BLE001
                self._failed(exc, "write the metadata of")

    def _commit(self, final: bool) -> IcechunkSaveResult | None:
        icechunk, _, _ = _import_icechunk()
        from pymeasure import __version__

        self._flush()
        self._sync_cache()
        if final:
            if not self._created:
                return None  # the run emitted nothing
            self._ensure_open()
            status = str(self.procedure.status)
            self._group.attrs.update({
                "status": status, "saved_at": datetime.now(timezone.utc).isoformat(),
            })
        else:
            status = "Running"
        message = self.message if final else f"{self.message} (in progress)"
        snapshot = self._session.commit(
            message,
            metadata={
                "group": self.group,
                "procedure": type(self.procedure).__name__,
                "pymeasure_version": __version__,
                "status": status,
                **self.extra_metadata,
            },
            rebase_with=icechunk.ConflictDetector(),
        )
        self._session = self._group = None
        self._arrays = {}
        self._last_commit = time.monotonic()
        return IcechunkSaveResult(snapshot, self.branch, self.group, self.store_name)

    def finish(self) -> IcechunkSaveResult | None:
        """Write what is left, make the final commit and release the group (done by the
        recorder at the end of the run). Returns where the data was saved (``None`` if the
        procedure emitted no data or the commit failed)."""
        with self._lock:
            if self._finished:
                return self.save_result
            result = None
            try:
                result = self._commit(final=True)
            except Exception as exc:  # noqa: BLE001
                self._failed(exc, "commit")
                self._dump_to_csv()
            self._finished = True
            self._session = self._group = None
            self._release()
            self.save_result = result
        if result is not None:
            log.info(f"Saved the results as '{result.group}' on branch '{result.branch}'")
            if self.on_commit is not None:
                self.on_commit(result)
        return result

    def _dump_to_csv(self) -> None:
        """Last resort if the store cannot be written: keep the data in a temporary CSV file."""
        try:
            frame = self.data
            if self._buffer:
                frame = pd.concat([frame, pd.DataFrame(self._buffer)], ignore_index=True)
            handle, path = tempfile.mkstemp(prefix="pymeasure_icechunk_fallback_", suffix=".csv")
            os.close(handle)
            frame.to_csv(path, index=False)
            log.error(f"The data was written to the file {path} instead.")
        except Exception:
            log.exception("The data could not be saved to a fallback file either.")

    # ---- reading -------------------------------------------------------------------------------

    def _source_group(self):
        """The group to read from: that of the open session, or the last commit."""
        _, _, zarr = _import_icechunk()
        if self._session is not None:
            return self._group
        session = self._repository.readonly_session(self.branch)
        return zarr.open_group(session.store, mode="r")[self.group]

    def _sync_cache(self) -> None:
        """Read the records that were written since the last read from the store."""
        if self._data is None:
            self._data = pd.DataFrame(columns=list(self.procedure.DATA_COLUMNS))
        known = len(self._data)
        if self._n_flushed <= known:
            return
        group = self._source_group()
        columns = {}
        for name in self.procedure.DATA_COLUMNS:
            values = group[name][known:self._n_flushed]
            columns[name] = values.astype(object) if _is_text(values.dtype) else values
        new = pd.DataFrame(columns)
        self._data = new if known == 0 else pd.concat([self._data, new], ignore_index=True)

    @property
    def data(self) -> pd.DataFrame:
        """The data so far, as a DataFrame; the records that were not yet written are written to
        the store first, and the new ones are read back from it."""
        with self._lock:
            if not self._finished:
                try:
                    self._flush()
                except Exception as exc:  # noqa: BLE001
                    self._failed(exc, "write")
            self._sync_cache()
            return self._data  # type: ignore[return-value]

    def reload(self) -> None:
        """Read all the data again from the store."""
        with self._lock:
            self._data = None
            self._sync_cache()

    def make_recorder(self, queue):
        return IcechunkRecorder(self, queue)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(group='{self.group}', store='{self.store_name}')>"


class _IcechunkHandler:
    """Handler of a recorder: hands the records to the results."""

    level = 0

    def __init__(self, results: IcechunkResults):
        self.results = results

    def handle(self, record) -> bool:
        self.results.store(record)
        return True

    def close(self) -> None:
        self.results.finish()


class IcechunkRecorder(QueueListener):
    """Recorder (see :class:`.Recorder`) that stores the records in :class:`IcechunkResults`
    and makes the final commit when it is stopped."""

    def __init__(self, results: IcechunkResults, queue):
        super().__init__(queue, _IcechunkHandler(results))

    def stop(self) -> None:
        for handler in self.handlers:
            handler.close()
        super().stop()


def load_group(store: "str | os.PathLike | Any", group: str, branch: str = "main"):
    """Open one saved run as an :class:`xarray.Dataset` (convenience for checking a save).

    :param store: Local directory of the store or an ``icechunk.Storage``.
    :param group: Path of the group, as returned in :attr:`IcechunkSaveResult.group`.
    :param branch: Branch to read.
    """
    icechunk, xarray, _ = _import_icechunk()
    if isinstance(store, (str, os.PathLike)):
        store = icechunk.local_filesystem_storage(os.path.abspath(os.path.expanduser(store)))
    session = icechunk.Repository.open(store).readonly_session(branch)
    return xarray.open_zarr(session.store, group=group, consolidated=False).load()


__all__ = [
    "INSTALL_HINT",
    "STORE_VARIABLE",
    "IcechunkRecorder",
    "IcechunkResults",
    "IcechunkSaveResult",
    "default_store",
    "load_group",
    "parse_metadata_text",
    "results_to_dataset",
    "save_results_to_icechunk",
    "validate_branch_name",
    "validate_group_path",
]
