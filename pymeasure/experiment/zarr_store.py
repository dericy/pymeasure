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

"""Record the data of an experiment in a `zarr <https://zarr.dev>`_ store.

Every experiment becomes a *group* of one zarr store (a directory), appended to the groups of
the earlier experiments: one array per data column, along the dimension ``record``, and the
procedure, its parameters and its metadata as attributes. The data is written while the
experiment runs and the live plots read it from the store (see :class:`ZarrResults`), instead of
a CSV file.

A zarr store in a directory is made of plain files, so it works on every operating system
(unlike the local-file storage of icechunk, see :mod:`pymeasure.experiment.icechunk_store`), and
a copy of the directory, made with any tool, is a valid store. There are no transactions,
commits or branches: for that, use icechunk.

The optional dependencies (``zarr`` and ``xarray``) are only imported when they are needed:
install them with ``pip install pymeasure[zarr]``.
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

#: Name of the environment variable (or entry of a ``.env`` file) with the default zarr store.
ZARR_STORE_VARIABLE = "PYMEASURE_ZARR_STORE"

INSTALL_HINT = (
    "Saving to a zarr store requires the optional packages 'zarr' and 'xarray': install them "
    "with 'pip install pymeasure[zarr]'."
)

_RESERVED: set[tuple[str, str, str]] = set()  # groups promised to queued results of this process
_RESERVED_LOCK = threading.Lock()


class ZarrSaveResult(NamedTuple):
    """Where a save ended up."""

    group: str  #: Path of the group in the store that holds the data.
    store: str  #: Description of the store (the path for a local store).


def _import_zarr():
    try:
        import xarray
        import zarr
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return xarray, zarr


_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10)),
}
_WINDOWS_FORBIDDEN = re.compile(r'[<>:"|?*\x00-\x1f]')


def validate_group_path(path: str, portable: bool = False) -> str:
    """Check that ``path`` is usable as the path of a zarr group and return it.

    Groups are separated by ``/``. No part may be empty, ``.`` or ``..``, and ``\\`` is not
    allowed. With ``portable=True`` (for stores in a directory, which can be copied to any
    operating system) the parts must also be valid names of files on Windows: no
    ``< > : " | ? *`` or control characters, no trailing space or dot, and not a reserved name
    such as ``CON`` or ``NUL``. Raises :class:`ValueError` otherwise.
    """
    if not path or any(part in ("", ".", "..") for part in path.split("/")) or "\\" in path:
        raise ValueError(
            f"Invalid group path '{path}': use names separated by '/', without empty parts, "
            "'.', '..' or backslashes."
        )
    if portable:
        for part in path.split("/"):
            if (_WINDOWS_FORBIDDEN.search(part) or part[-1] in " ."
                    or part.split(".")[0].upper() in _WINDOWS_RESERVED):
                raise ValueError(
                    f"Invalid group path '{path}': the part '{part}' cannot be the name of a "
                    "directory on Windows (no < > : \" | ? * or control characters, no trailing "
                    "space or dot, no reserved names such as CON or NUL)."
                )
    return path


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
    xarray, _ = _import_zarr()

    data = results.data
    if len(data) == 0:
        raise ValueError("The results contain no data.")
    dataset = xarray.Dataset.from_dataframe(data.rename_axis("record"))
    for column, unit in Procedure.parse_columns([str(c) for c in data.columns]).items():
        dataset[column].attrs["units"] = f"{unit:~}"

    dataset.attrs.update(_run_attributes(results))
    return dataset


def _unique_group(root, base: str, index: bool) -> str:
    """Return ``base`` with a ``_#`` suffix that makes it new in ``root`` (like filenames)."""
    if not index:
        return base
    i = 1
    while f"{base}_{i}" in root:
        i += 1
    return f"{base}_{i}"


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


def lookup_setting(variable: str, start: "str | os.PathLike | None" = None) -> str | None:
    """Return the value of a setting, or ``None`` if it is not set.

    It is the value of the environment variable ``variable``; if that is not set (or blank), the
    entry ``variable=...`` of the first ``.env`` file found in ``start`` (default: the current
    directory) or its parent directories. A relative path in a ``.env`` file is relative to the
    directory of that file (a value with ``://``, such as an S3 URL, is returned as it is). The
    ``.env`` file is only read, ``os.environ`` is not changed.
    """
    value = os.environ.get(variable, "").strip()
    if value:
        return value
    directory = Path(start if start is not None else Path.cwd()).expanduser().resolve()
    for folder in (directory, *directory.parents):
        value = _read_env_file(folder / ".env").get(variable, "").strip()
        if value:
            if "://" in value or Path(value).expanduser().is_absolute():
                return value
            return str(folder / value)
    return None


def default_store(start: "str | os.PathLike | None" = None) -> str | None:
    """Return the default zarr store, or ``None`` if none is configured.

    It is the environment variable ``PYMEASURE_ZARR_STORE`` or the entry of that name in a
    ``.env`` file, see :func:`lookup_setting`.

    :param start: Directory where the search for a ``.env`` file begins.
    """
    return lookup_setting(ZARR_STORE_VARIABLE, start)


def _store_for(store, create: bool = True) -> tuple[Any, str]:
    """Return ``(zarr store, description)`` for a directory (the path of a local store), a zarr
    store object (for example one for S3) or ``None`` (a store in memory that is lost with the
    process)."""
    _, zarr = _import_zarr()
    if store is None:
        return zarr.storage.MemoryStore(), "in-memory"
    if isinstance(store, (str, os.PathLike)):
        path = os.path.abspath(os.path.expanduser(os.fspath(store)))
        if not create and not os.path.isdir(path):
            raise FileNotFoundError(f"There is no zarr store at '{path}'.")
        return zarr.storage.LocalStore(path), path
    return store, repr(store)


class ZarrResults(Results):
    """Results that live in a zarr store instead of a CSV file.

    While the procedure runs, the records that it emits are appended to the arrays of one new
    group of the store (one array per data column, along the dimension ``record``, like
    :func:`results_to_dataset`), and :attr:`data` -- which the curves, images and tables of the
    graphical interface read -- is read back from the store. Nothing is written to a file.
    Records are buffered and written in batches (at most every :attr:`FLUSH_INTERVAL` seconds,
    and whenever :attr:`data` is read), so that emitting a record is cheap.

    The group is new: it is appended to the groups that the store has already (an index ``_#``
    is added to its name so that no earlier measurement is overwritten, as for a filename). Its
    attributes describe the run, with the ``status`` ``Running`` until the run ends and then
    ``Finished``, ``Aborted`` or ``Failed``, so the data of aborted and failed experiments is
    kept. A zarr store has no transactions: other programs can read the data while it is
    written, and a group that is left at ``Running`` belongs to a run that was interrupted.
    If the data cannot be written at the end, it is written to a temporary CSV file, whose name
    is logged, so that a measurement is not lost.

    Use it where a :class:`.Results` object is created, for example in the ``queue`` method of a
    window (see :meth:`.ZarrInputWidget.new_results`)::

        results = ZarrResults(procedure, "path/to/store.zarr", group="{Sample name}")
        experiment = self.new_experiment(results)
        self.manager.queue(experiment)

    The group is chosen when the results are created, so that several queued experiments get
    different groups. Group names are checked to be valid on every operating system (see
    :func:`validate_group_path`), so that the store can be copied to Windows.

    :param procedure: Procedure of the experiment.
    :param store: Directory of the store (created if needed, see ``create``), a zarr store
        object (for example for S3), or ``None`` for a store in memory (the data can be plotted
        but is lost with the process).
    :param group: Template of the group path (placeholders as for the filename).
    :param metadata: Template values of extra metadata, stored in the attribute
        ``extra_metadata`` of the group.
    :param index: Add a ``_#`` suffix to the group so that it is new. With ``False``, an existing
        group raises :class:`FileExistsError`.
    :param create: Create the directory of the store if it does not exist.
    :param commit_interval: Not used by a zarr store (there is nothing to commit); for the
        versioned stores of subclasses.
    :param on_commit: Called with the :class:`ZarrSaveResult` when the data is complete (from the
        thread of the worker).
    :param on_error: Called with the exception if writing to the store failed (from the thread
        that notices the failure).
    """

    #: Longest time (in seconds) that records stay in memory before they are written.
    FLUSH_INTERVAL = 0.5
    #: Chunk length of the arrays.
    CHUNK_LENGTH = 4096

    _kind = "zarr store"  # for messages
    _fallback_prefix = "pymeasure_zarr_fallback_"
    _portable_names = True  # group names must be valid file names everywhere
    _metadata_in_attributes = True  # extra metadata goes to the attributes of the group

    def __init__(
        self,
        procedure: Procedure,
        store=None,
        group: str = "DATA",
        metadata: Mapping[str, str] | None = None,
        index: bool = True,
        create: bool = True,
        commit_interval: float | None = None,
        on_commit: Callable[[Any], Any] | None = None,
        on_error: Callable[[Exception], Any] | None = None,
    ):
        if not isinstance(procedure, Procedure):
            raise TypeError("Results require a Procedure object")
        self._import_backend()  # fail early if the optional packages are missing

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

        base = validate_group_path(
            _fill_placeholders(group, procedure, "group name"), portable=self._portable_names)
        self.extra_metadata = {
            key: _fill_placeholders(value, procedure, f"metadata value of '{key}'")
            for key, value in (metadata or {}).items()
        }
        self.index = index
        self.commit_interval = commit_interval
        self.on_commit = on_commit
        self.on_error = on_error

        self._open_backend(store, create)
        self._base = base
        self.group = self._reserve(base)
        self.data_filename = f"{self.store_name}#{self.group}"

        self._lock = threading.RLock()
        self._buffer: list[Mapping[str, Any]] = []
        self._n_flushed = 0  # records in the store
        self._group: Any = None  # the group while it is open for writing
        self._arrays: dict[str, Any] = {}
        self._created = False  # the group exists in the store
        self._units = {
            column: f"{unit:~}"
            for column, unit in Procedure.parse_columns(list(procedure.DATA_COLUMNS)).items()
        }
        self._last_flush = self._last_commit = time.monotonic()
        self._finished = False
        self.save_result: Any = None

    # ---- the store: what a subclass replaces ---------------------------------------------------

    @property
    def _scope(self) -> str:
        """Part of the identity of a group besides the store (for example the branch)."""
        return ""

    def _import_backend(self) -> None:
        _import_zarr()

    def _open_backend(self, store, create: bool) -> None:
        """Open the store: set :attr:`store_name` and what the other methods need."""
        self._zstore, self.store_name = _store_for(store, create)

    def _read_root(self) -> Any:
        """The root group of the store for reading, or ``None`` if nothing was ever saved."""
        _, zarr = _import_zarr()
        try:
            return zarr.open_group(self._zstore, mode="r")
        except (zarr.errors.GroupNotFoundError, FileNotFoundError):
            return None

    def _writable_root(self) -> Any:
        """The root group of the store for writing (created if needed)."""
        _, zarr = _import_zarr()
        return zarr.open_group(self._zstore, mode="a")

    def _commit(self, final: bool) -> Any:
        """Write everything that is pending; at the end (``final``) mark the run as done and
        return where the data is. A zarr store has nothing to commit while running."""
        self._flush()
        self._sync_cache()
        if not final or not self._created:  # nothing to finish if the run emitted nothing
            return None
        self._set_final_attributes()
        self._group = None
        self._arrays = {}
        return ZarrSaveResult(self.group, self.store_name)

    def _set_final_attributes(self) -> str:
        self._ensure_open()
        status = str(self.procedure.status)
        self._group.attrs.update({
            "status": status, "saved_at": datetime.now(timezone.utc).isoformat(),
        })
        return status

    # ---- names ---------------------------------------------------------------------------------

    def _reserve(self, base: str) -> str:
        root = self._read_root()
        with _RESERVED_LOCK:
            def taken(name):
                return ((root is not None and name in root)
                        or (self.store_name, self._scope, name) in _RESERVED)

            if not self.index:
                if taken(base):
                    raise FileExistsError(f"The group '{base}' exists already in the store.")
                name = base
            else:
                i = 1
                while taken(f"{base}_{i}"):
                    i += 1
                name = f"{base}_{i}"
            _RESERVED.add((self.store_name, self._scope, name))
        return name

    def _release(self) -> None:
        with _RESERVED_LOCK:
            _RESERVED.discard((self.store_name, self._scope, self.group))

    # ---- writing -------------------------------------------------------------------------------

    def _ensure_open(self) -> None:
        """Make sure that the group and its arrays are open for writing."""
        if self._group is not None:
            return
        root = self._writable_root()
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
            if self._metadata_in_attributes and self.extra_metadata:
                group.attrs["extra_metadata"] = self.extra_metadata
            group.attrs["status"] = "Running"
            self._arrays = {}
            self._created = True
        else:
            group = root[self.group]
            self._arrays = {name: group[name] for name in group.array_keys()}
        self._group = group

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
        """Write the buffered records to the arrays of the group."""
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
        log.error(f"Could not {what} the data in the {self._kind}: {exc!r}")
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

    def finish(self) -> Any:
        """Write what is left, mark the run as done and release the group (done by the recorder
        at the end of the run). Returns where the data was saved (``None`` if the procedure
        emitted no data or the data could not be saved)."""
        with self._lock:
            if self._finished:
                return self.save_result
            result = None
            try:
                result = self._commit(final=True)
            except Exception as exc:  # noqa: BLE001
                self._failed(exc, "finish")
                self._dump_to_csv()
            self._finished = True
            self._group = None
            self._release()
            self.save_result = result
        if result is not None:
            log.info(f"Saved the results as '{result.group}' in {self.store_name}")
            if self.on_commit is not None:
                self.on_commit(result)
        return result

    def _dump_to_csv(self) -> None:
        """Last resort if the store cannot be written: keep the data in a temporary CSV file."""
        try:
            frame = self.data
            if self._buffer:
                frame = pd.concat([frame, pd.DataFrame(self._buffer)], ignore_index=True)
            handle, path = tempfile.mkstemp(prefix=self._fallback_prefix, suffix=".csv")
            os.close(handle)
            frame.to_csv(path, index=False)
            log.error(f"The data was written to the file {path} instead.")
        except Exception:
            log.exception("The data could not be saved to a fallback file either.")

    # ---- reading -------------------------------------------------------------------------------

    def _source_group(self):
        """The group to read from: the open one, or the one in the store."""
        if self._group is not None:
            return self._group
        return self._read_root()[self.group]

    def _sync_cache(self) -> None:
        """Read the records that were written since the last read from the store."""
        if self._data is None:
            self._data = pd.DataFrame(
                columns=list(self.procedure.DATA_COLUMNS))  # pyright: ignore[reportArgumentType]
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
        return ZarrRecorder(self, queue)

    @property
    def display_name(self) -> str:
        """The store with the group, as shown in the list of experiments."""
        return self.data_filename

    @property
    def display_details(self) -> str:
        return f"Store: {self.store_name}\nGroup: {self.group}"

    @property
    def is_file(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(group='{self.group}', store='{self.store_name}')>"


class _RecorderHandler(logging.Handler):
    """Handler of a recorder: hands the records (dictionaries, not log records) to the
    results, and completes them when it is closed."""

    def __init__(self, results: ZarrResults):
        super().__init__(logging.NOTSET)
        self.results = results

    def emit(self, record: Any) -> None:  # a dictionary of the data, see ZarrRecorder
        self.results.store(record)

    def close(self) -> None:
        self.results.finish()
        super().close()


class ZarrRecorder(QueueListener):
    """Recorder (see :class:`.Recorder`) that stores the records in :class:`ZarrResults` and
    completes the group when it is stopped."""

    def __init__(self, results: ZarrResults, queue):
        super().__init__(queue, _RecorderHandler(results))

    def stop(self) -> None:
        for handler in self.handlers:
            handler.close()
        super().stop()


def save_results_to_zarr(
    results: Results,
    store: "str | os.PathLike | Any",
    group: str = "DATA",
    metadata: Mapping[str, str] | None = None,
    index: bool = True,
    create: bool = True,
) -> ZarrSaveResult:
    """Append the data of finished ``results`` (for example loaded from a CSV file) to a zarr
    store, as a new group.

    The group is written as described for :func:`results_to_dataset`. ``group`` and the values of
    ``metadata`` may contain placeholders for the parameters and metadata of the procedure and
    for ``{date}`` and ``{time}``, as the filename in the graphical interface.

    :param results: Results to save.
    :param store: Directory of the store, a zarr store object, or ``None`` (a store in memory).
    :param group: Template of the group path, groups separated by ``/``.
    :param metadata: Template values of extra metadata (attribute ``extra_metadata``).
    :param index: Add a ``_#`` suffix to the group name so that it is new. With ``False``, an
        existing group raises :class:`FileExistsError`.
    :param create: Create the directory of the store if it does not exist.
    :raises ValueError: For invalid names or placeholders, or if there is no data.
    :raises ImportError: If the optional dependencies are not installed.
    :returns: Where the data was saved.
    """
    _, zarr = _import_zarr()
    procedure = results.procedure
    base = validate_group_path(_fill_placeholders(group, procedure, "group name"), portable=True)
    extra = {
        key: _fill_placeholders(value, procedure, f"metadata value of '{key}'")
        for key, value in (metadata or {}).items()
    }
    dataset = results_to_dataset(results)  # raises if there is no data, before touching the store
    dataset.attrs["status"] = str(procedure.status)
    if extra:
        dataset.attrs["extra_metadata"] = extra
    zstore, store_name = _store_for(store, create)
    root = zarr.open_group(zstore, mode="a")
    name = _unique_group(root, base, index)
    if name in root:
        raise FileExistsError(f"The group '{name}' exists already in the store.")
    dataset.to_zarr(zstore, group=name, mode="w-", consolidated=False)
    log.info(f"Saved the results as '{name}' in {store_name}")
    return ZarrSaveResult(name, store_name)


def load_zarr_group(store: "str | os.PathLike | Any", group: str):
    """Open one saved run as an :class:`xarray.Dataset`.

    :param store: Directory of the store or a zarr store object.
    :param group: Path of the group, as returned in :attr:`ZarrSaveResult.group`.
    """
    xarray, _ = _import_zarr()
    if isinstance(store, (str, os.PathLike)):
        store = os.path.abspath(os.path.expanduser(os.fspath(store)))
    return xarray.open_zarr(store, group=group, consolidated=False).load()


__all__ = [
    "INSTALL_HINT",
    "ZARR_STORE_VARIABLE",
    "ZarrRecorder",
    "ZarrResults",
    "ZarrSaveResult",
    "default_store",
    "load_zarr_group",
    "lookup_setting",
    "parse_metadata_text",
    "results_to_dataset",
    "save_results_to_zarr",
    "validate_group_path",
]
