"""Path resolution.

Single rule for the whole project: **no drive letters, no absolute paths anywhere
else in the codebase.** Everything goes through here, so the same code runs on the
Windows workstation and the Linux A4000 box without edits.

Override the data location with the ``EPCG_DATA_ROOT`` environment variable:

    Windows : set EPCG_DATA_ROOT=D:\\datasets\\epcg
    Linux   : export EPCG_DATA_ROOT=/data/epcg
"""

from __future__ import annotations

import os
from pathlib import Path

# epcg/paths.py -> epcg/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """Root of all datasets. Defaults to ``<repo>/data``."""
    env = os.environ.get("EPCG_DATA_ROOT")
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "data"


def raw_dir() -> Path:
    """PhysioNet downloads, kept byte-identical to upstream. Never written to."""
    return data_root() / "raw"


def interim_dir() -> Path:
    """Parsed + preprocessed HDF5, one file per source dataset."""
    return data_root() / "interim"


def processed_dir() -> Path:
    """Training-ready window indices and split manifests."""
    return data_root() / "processed"


def experiments_dir() -> Path:
    return REPO_ROOT / "experiments"


def ensure_dirs() -> None:
    for d in (raw_dir(), interim_dir(), processed_dir()):
        d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    src = "EPCG_DATA_ROOT" if os.environ.get("EPCG_DATA_ROOT") else "default (<repo>/data)"
    return (
        f"repo root : {REPO_ROOT}\n"
        f"data root : {data_root()}   [{src}]\n"
        f"  raw       : {raw_dir()}\n"
        f"  interim   : {interim_dir()}\n"
        f"  processed : {processed_dir()}"
    )


if __name__ == "__main__":
    print(describe())
