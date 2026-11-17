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
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, NamedTuple

import numpy as np

from .procedure import Procedure
from .results import Results, replace_placeholders

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

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


def results_to_dataset(results: Results):
    """Convert the data of ``results`` to an :class:`xarray.Dataset`.

    Each data column becomes a variable along the dimension ``record``. Columns with a unit in
    their header (``"Voltage (V)"``) get the attribute ``units``. The attributes of the dataset
    describe the run: the procedure, its parameters and metadata (by display name, with their
    units in ``parameter_units``), the data file, the pymeasure version and the time of saving.
    """
    _, xarray, _ = _import_icechunk()
    from pymeasure import __version__

    data = results.data
    if len(data) == 0:
        raise ValueError("The results contain no data.")
    dataset = xarray.Dataset.from_dataframe(data.rename_axis("record"))
    for column, unit in Procedure.parse_columns([str(c) for c in data.columns]).items():
        dataset[column].attrs["units"] = f"{unit:~}"

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
    dataset.attrs.update({
        "pymeasure_version": __version__,
        "procedure": type(procedure).__name__,
        "procedure_module": type(procedure).__module__,
        "parameters": parameters,
        "parameter_units": parameter_units,
        "metadata": metadata,
        "data_filename": str(results.data_filename),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    })
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

    if isinstance(store, (str, os.PathLike)):
        path = os.path.abspath(os.path.expanduser(os.fspath(store)))
        storage = icechunk.local_filesystem_storage(path)
        store_name = path
    else:
        storage, store_name = store, repr(store)
    repository = _open_repository(storage, create)

    branches = repository.list_branches()
    if branch not in branches:
        if not create_branch or "main" not in branches:
            raise ValueError(f"The branch '{branch}' does not exist in the store.")
        repository.create_branch(branch, repository.lookup_branch("main"))

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
    "IcechunkSaveResult",
    "INSTALL_HINT",
    "load_group",
    "parse_metadata_text",
    "results_to_dataset",
    "save_results_to_icechunk",
    "validate_branch_name",
    "validate_group_path",
]
