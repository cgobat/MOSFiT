"""Neural and tabulated emulator *weights* live here, separate from SED code.

Each subdirectory (e.g. ``sampson/``) holds artifacts for one emulator that is
driven from a ``kind: sed`` module under ``mosfit.modules.seds``.

Override the search path at runtime::

    export MOSFIT_EMULATOR_DATA=/path/to/root   # uses ``<root>/<name>/``

If unset or ``<root>/<name>`` is missing, MOSFiT uses the copy shipped inside
``mosfit/emulators/<name>/`` next to the installed package.
"""
from __future__ import annotations

import os
from pathlib import Path


def emulator_weights_dir(name: str) -> Path:
    """Directory containing weight files for emulator ``name`` (e.g. ``'sampson'``)."""
    root = os.environ.get("MOSFIT_EMULATOR_DATA")
    if root:
        candidate = Path(root).expanduser() / name
        if candidate.is_dir():
            return candidate.resolve()
    # This file is ``mosfit/emulators/__init__.py``; bundled weights live in subdirs.
    return (Path(__file__).resolve().parent / name).resolve()
