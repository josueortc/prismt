"""Reading and writing MATLAB v7.3 (HDF5) files without guessing.

MATLAB stores arrays column-major, so h5py reports the dimensions of a MATLAB ``[N R T M]``
array reversed, as ``(M, T, R, N)``. MATLAB also drops trailing singleton dimensions: an
``[N R T 1]`` array is stored as ``[N R T]``. Both are undone here deterministically, using
the shape that the file itself declares.

:func:`write_mat73` writes files that MATLAB's ``load`` opens directly: a 512-byte MATLAB
header in the HDF5 user block, arrays with reversed dimensions, and a ``MATLAB_class``
attribute on every variable.
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np

_MATLAB_CLASS = {
    np.dtype("float32"): "single",
    np.dtype("float64"): "double",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
}

_HEADER_TEXT_BYTES = 116
_USERBLOCK = 512


def is_hdf5(path: str | os.PathLike) -> bool:
    try:
        return h5py.is_hdf5(str(path))
    except OSError:
        return False


def mat_header(path: str | os.PathLike) -> str:
    """The first line of a .mat file's text header, or '' if there is none."""
    with open(path, "rb") as fh:
        head = fh.read(_HEADER_TEXT_BYTES)
    return head.split(b"\x00")[0].decode("ascii", errors="replace").strip()


def matlab_array(raw: np.ndarray, shape: Sequence[int] | None = None) -> np.ndarray:
    """Return an h5py-read array in MATLAB's dimension order, restoring dropped singletons.

    ``shape`` is the MATLAB size declared by the file. Without it the reversed array is
    returned as stored.
    """
    arr = np.asarray(raw)
    arr = np.ascontiguousarray(np.transpose(arr, tuple(range(arr.ndim))[::-1]))
    if shape is None:
        return arr
    shape = tuple(int(s) for s in shape)
    stored = arr.shape
    # MATLAB may drop trailing singleton dimensions (and always stores at least 2-D).
    trimmed = list(shape)
    while len(trimmed) > 2 and trimmed[-1] == 1:
        trimmed.pop()
    if stored != tuple(shape) and stored != tuple(trimmed):
        # A 1-D declared shape is stored as 1xN or Nx1 by MATLAB.
        if int(np.prod(stored)) != int(np.prod(shape)):
            raise ValueError(f"stored size {list(stored)} does not match declared size {list(shape)}")
        if [s for s in stored if s != 1] != [s for s in shape if s != 1]:
            raise ValueError(f"stored size {list(stored)} does not match declared size {list(shape)}")
    return arr.reshape(shape)


def read_text(dataset: h5py.Dataset) -> str:
    """Decode a MATLAB uint8 (UTF-8 bytes) or char (UTF-16 code units) variable."""
    raw = np.asarray(dataset[()])
    cls = dataset.attrs.get("MATLAB_class", b"")
    cls = cls.decode() if isinstance(cls, bytes) else str(cls)
    if cls == "char" or raw.dtype == np.uint16:
        return "".join(chr(c) for c in raw.ravel(order="F"))
    return bytes(raw.astype(np.uint8).ravel(order="F")).decode("utf-8")


def read_scalar(dataset: h5py.Dataset) -> float:
    raw = np.asarray(dataset[()]).ravel()
    if raw.size != 1:
        raise ValueError(f"expected a scalar, found {raw.size} values")
    return float(raw[0])


def text_to_uint8(text: str) -> np.ndarray:
    """UTF-8 bytes as a MATLAB 1xB uint8 row vector."""
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).reshape(1, -1)


def write_mat73(path: str | os.PathLike, variables: Mapping[str, np.ndarray]) -> Path:
    """Write ``variables`` as a MATLAB v7.3 file that ``load`` opens directly.

    Arrays are given in MATLAB dimension order (e.g. ``X`` as ``[N R T M]``). 1-D arrays
    become MATLAB row vectors and scalars become 1x1. The file is written next to its
    destination and renamed when complete.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    with h5py.File(tmp, "w", userblock_size=_USERBLOCK, libver="earliest") as fh:
        for name, value in variables.items():
            arr = np.asarray(value)
            if arr.dtype == np.bool_:
                raise TypeError(f"{name}: write logical arrays as uint8")
            cls = _MATLAB_CLASS.get(arr.dtype)
            if cls is None:
                raise TypeError(f"{name}: unsupported dtype {arr.dtype}")
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            data = np.ascontiguousarray(np.transpose(arr, tuple(range(arr.ndim))[::-1]))
            ds = fh.create_dataset(name, data=data)
            ds.attrs["MATLAB_class"] = np.bytes_(cls)
    _write_header(tmp)
    os.replace(tmp, path)
    return path


def _write_header(path: Path) -> None:
    created = time.strftime("%a %b %d %H:%M:%S %Y")
    plat = {"Darwin": "MACA64", "Linux": "GLNXA64", "Windows": "PCWIN64"}.get(platform.system(), "GLNXA64")
    text = f"MATLAB 7.3 MAT-file, Platform: {plat}, Created on: {created} HDF5 schema 1.00 ."
    header = text.ljust(_HEADER_TEXT_BYTES).encode("ascii")[:_HEADER_TEXT_BYTES]
    header += b"\x00" * 8  # subsystem data offset
    header += b"\x00\x02"  # version 0x0200
    header += b"IM"  # endian indicator (little-endian writer)
    header = header.ljust(_USERBLOCK, b"\x00")
    with open(path, "r+b") as fh:
        fh.write(header)
