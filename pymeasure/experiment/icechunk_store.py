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
import os
import re
import time
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from .procedure import Procedure
from .results import Results
from .zarr_store import (
    ZarrRecorder,
    ZarrResults,
    _fill_placeholders,
    _unique_group,
    lookup_setting,
    parse_metadata_text,
    results_to_dataset,
    validate_group_path,
)

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


def default_store(start: "str | os.PathLike | None" = None) -> str | None:
    """Return the default icechunk store, or ``None`` if none is configured.

    It is the value of the environment variable ``PYMEASURE_ICECHUNK_STORE``; if that is not set,
    the entry ``PYMEASURE_ICECHUNK_STORE=...`` of the first ``.env`` file found in ``start``
    (default: the current directory) or its parent directories. A relative path in a ``.env``
    file is relative to the directory of that file (a value with ``://``, such as an S3 URL, is
    returned as it is). The ``.env`` file is only read, ``os.environ`` is not changed.

    :param start: Directory where the search for a ``.env`` file begins.
    """
    return lookup_setting(STORE_VARIABLE, start)


def _import_icechunk():
    try:
        import icechunk
        import xarray
        import zarr
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return icechunk, xarray, zarr


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


class IcechunkResults(ZarrResults):
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

    This is a :class:`~pymeasure.experiment.zarr_store.ZarrResults` that keeps the store in an
    icechunk repository: the data is written to a session of the branch, and a commit is a
    consistent version of the store that readers see.
    """

    _kind = "icechunk store"
    _fallback_prefix = "pymeasure_icechunk_fallback_"
    _portable_names = False  # object stores have no problem with the names of groups
    _metadata_in_attributes = False  # it goes to the metadata of the commit

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
        self.message = _fill_placeholders(message, procedure, "commit message")
        self.branch = validate_branch_name(branch)
        self._create_branch = create_branch
        self._session: Any = None  # writable session while there are uncommitted changes
        super().__init__(
            procedure, store, group, metadata, index, create, commit_interval, on_commit,
            on_error)

    # ---- the store -----------------------------------------------------------------------------

    @property
    def _scope(self) -> str:
        return self.branch

    def _import_backend(self) -> None:
        _import_icechunk()

    def _open_backend(self, store, create: bool) -> None:
        storage, self.store_name = _storage_for(store)
        self._repository = _open_repository(storage, create)
        _ensure_branch(self._repository, self.branch, self._create_branch)

    def _read_root(self):
        _, _, zarr = _import_icechunk()
        session = self._repository.readonly_session(self.branch)
        try:
            return zarr.open_group(session.store, mode="r")
        except zarr.errors.GroupNotFoundError:  # nothing was ever saved to this store
            return None

    def _writable_root(self):
        _, _, zarr = _import_icechunk()
        self._session = self._repository.writable_session(self.branch)
        return zarr.open_group(self._session.store, mode="a")

    def _commit(self, final: bool) -> IcechunkSaveResult | None:
        icechunk, _, _ = _import_icechunk()
        from pymeasure import __version__

        self._flush()
        self._sync_cache()
        if final:
            if not self._created:
                return None  # the run emitted nothing
            status = self._set_final_attributes()
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

    def make_recorder(self, queue):
        return IcechunkRecorder(self, queue)

    @property
    def display_name(self) -> str:
        """The commit message and the store with the group, as shown in the list of experiments."""
        return f"{self.message} \u2014 {self.data_filename}"

    @property
    def display_details(self) -> str:
        return (f"Store: {self.store_name}\nGroup: {self.group}\nBranch: {self.branch}\n"
                f"Commit message: {self.message}")


class IcechunkRecorder(ZarrRecorder):
    """Recorder (see :class:`.Recorder`) that stores the records in :class:`IcechunkResults`
    and makes the final commit when it is stopped."""


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
