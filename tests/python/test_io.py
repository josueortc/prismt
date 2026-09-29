"""The dataset contract: strict reading, validation messages and round trips."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io

from conftest import FIXTURES, value_coded
from prismt.errors import DatasetError
from prismt.io import read_dataset, validate_file, write_dataset
from prismt.io import matv73
from prismt.io.dataset import build_meta


def _write(path: Path, X: np.ndarray, **kw) -> Path:
    trials = kw.pop("trials", {"mouse": np.array([f"m{i % 3}" for i in range(X.shape[0])], dtype=object)})
    kw.setdefault("subject", "mouse")
    kw.setdefault("fs_hz", 10.0)
    return write_dataset(path, X, trials, **kw)


@pytest.mark.parametrize("suffix", [".mat", ".npz"])
@pytest.mark.parametrize("shape", [(5, 4, 3, 2), (5, 4, 3, 1), (1, 4, 3, 2), (3, 1, 1, 1)])
def test_round_trip_restores_every_value_by_index(tmp_path, suffix, shape):
    X = value_coded(*shape)
    X[0, 0, 0, 0] = np.nan
    ds = read_dataset(_write(tmp_path / f"d{suffix}", X))
    assert ds.X.shape == shape
    np.testing.assert_array_equal(ds.X, X)


def test_mat_and_npz_give_the_same_fingerprint(tmp_path):
    X = value_coded(6, 3, 4, 2)
    a = read_dataset(_write(tmp_path / "a.mat", X))
    b = read_dataset(_write(tmp_path / "b.npz", X))
    assert a.fingerprint == b.fingerprint and len(a.fingerprint) == 32


def test_python_written_mat_has_the_matlab_header_and_reversed_dims(tmp_path):
    X = value_coded(5, 4, 3, 2)
    path = _write(tmp_path / "d.mat", X)
    assert matv73.mat_header(path).startswith("MATLAB 7.3 MAT-file")
    with h5py.File(path, "r") as fh:
        assert fh["X"].shape == (2, 3, 4, 5)
        assert fh["X"].attrs["MATLAB_class"] == b"single"
        assert fh["meta_json"].attrs["MATLAB_class"] == b"uint8"


def test_matlab_written_fixture_reads_exactly(fixtures_dir):
    path = fixtures_dir / "dataset_v1_matlab.mat"
    if not path.exists():
        pytest.skip("MATLAB fixture not generated yet (tests/matlab/makeFixtures.m)")
    ds = read_dataset(path)
    N, R, T, M = ds.X.shape
    expected = value_coded(N, R, T, M)
    expected[1, 2, 3, 0] = np.nan  # the planted NaN, X(2,3,4,1) in MATLAB
    np.testing.assert_array_equal(ds.X, expected)
    assert ds.channel_names[0] == "Visual L" and ds.modality_names == ("calcium", "ach")
    assert ds.columns["phase"].as_text().tolist()[:2] == ["early", "early"]
    assert ds.columns["stim"].labels == {0.0: "CS-", 1.0: "CS+"}
    assert ds.subject == "mouse" and ds.session == "session"
    assert ds.columns["note"].values[0] == "café µ"


def test_matlab_single_modality_fixture_restores_trailing_singleton(fixtures_dir):
    path = fixtures_dir / "dataset_v1_matlab_m1.mat"
    if not path.exists():
        pytest.skip("MATLAB fixture not generated yet")
    ds = read_dataset(path)
    assert ds.X.ndim == 4 and ds.X.shape[3] == 1
    np.testing.assert_array_equal(ds.X, value_coded(*ds.X.shape))


def test_infinite_values_are_rejected_with_the_matlab_index(tmp_path):
    X = value_coded(4, 3, 2, 1)
    X[2, 1, 0, 0] = np.inf
    with pytest.raises(DatasetError) as err:
        _write(tmp_path / "d.mat", X)
    assert err.value.code == "E_DATA_INF" and "X(3,2,1,1)" in err.value.message


def test_older_mat_file_is_explained(tmp_path):
    path = tmp_path / "old.mat"
    scipy.io.savemat(path, {"processed_data": {"dff": np.zeros((2, 3, 4))}})
    with pytest.raises(DatasetError) as err:
        read_dataset(path)
    assert err.value.code == "E_DATA_NOT_PRISMT" and "older" in err.value.message
    assert "Import" in err.value.hint


def test_untagged_hdf5_file_is_rejected(tmp_path):
    path = tmp_path / "x.mat"
    with h5py.File(path, "w") as fh:
        fh["T"] = np.zeros(3)
    with pytest.raises(DatasetError) as err:
        read_dataset(path)
    assert err.value.code == "E_DATA_NOT_PRISMT" and "T" in err.value.message


def test_newer_format_version_asks_for_an_update(tmp_path):
    X = value_coded(2, 2, 2, 1)
    meta = build_meta(X, {}, fs_hz=10.0)
    path = tmp_path / "future.mat"
    matv73.write_mat73(path, {"prismt_format": matv73.text_to_uint8("prismt.dataset"),
                              "prismt_version": np.array(99.0), "X": X,
                              "meta_json": matv73.text_to_uint8(json.dumps(meta))})
    with pytest.raises(DatasetError) as err:
        read_dataset(path)
    assert err.value.code == "E_DATA_NEWER_FORMAT"


def test_orientation_probes_catch_swapped_axes_even_when_sizes_match(tmp_path):
    X = value_coded(3, 4, 4, 1)  # channels == time, so a shape check cannot tell them apart
    meta = build_meta(X, {}, fs_hz=10.0)
    swapped = np.ascontiguousarray(X.transpose(0, 2, 1, 3))
    path = tmp_path / "swapped.mat"
    matv73.write_mat73(path, {"prismt_format": matv73.text_to_uint8("prismt.dataset"),
                              "prismt_version": np.array(1.0), "X": swapped,
                              "meta_json": matv73.text_to_uint8(json.dumps(meta))})
    with pytest.raises(DatasetError) as err:
        read_dataset(path)
    assert err.value.code == "E_DATA_ORIENTATION"


@pytest.mark.parametrize(
    "kwargs, code",
    [
        ({"channel_names": ["a", "a", "b"]}, "E_DATA_CHANNELS"),
        ({"channel_names": ["a", "b"]}, "E_DATA_CHANNELS"),
        ({"modality_channels": [[1, 5]]}, "E_DATA_MODALITIES"),
        ({"modality_kinds": ["brainwaves"]}, "E_DATA_MODALITIES"),
        ({"times_s": [0.0, 0.0]}, "E_DATA_TIME"),
        ({"subject": "animal"}, "E_DATA_ROLES"),
    ],
)
def test_invalid_descriptions_raise_actionable_errors(tmp_path, kwargs, code):
    X = value_coded(4, 3, 2, 1)
    with pytest.raises(DatasetError) as err:
        _write(tmp_path / "d.mat", X, **kwargs)
    codes = [d["code"] for d in err.value.details] or [err.value.code]
    assert code in codes
    assert err.value.message


def test_column_with_wrong_length_is_rejected(tmp_path):
    X = value_coded(4, 3, 2, 1)
    with pytest.raises(DatasetError, match="has 3 values but there are 4 trials"):
        write_dataset(tmp_path / "d.mat", X, {"stim": [0, 1, 0]}, fs_hz=10)


def test_validate_collects_all_errors_and_warnings_without_raising(tmp_path):
    X = value_coded(4, 3, 2, 1)
    meta = build_meta(X, {"stim": [0, 1, 0, 1]})
    meta["channels"]["names"] = ["a", "a"]
    meta["modalities"]["names"] = []
    path = tmp_path / "bad.mat"
    matv73.write_mat73(path, {"prismt_format": matv73.text_to_uint8("prismt.dataset"),
                              "prismt_version": np.array(1.0), "X": X,
                              "meta_json": matv73.text_to_uint8(json.dumps(meta))})
    report = validate_file(path)
    assert not report["ok"]
    assert {e["code"] for e in report["errors"]} >= {"E_DATA_CHANNELS", "E_DATA_MODALITIES"}
    assert any(w["code"] == "W_DATA_NO_SUBJECT" for w in report["warnings"])


def test_scalar_values_from_matlab_jsonencode_are_accepted(tmp_path):
    """MATLAB writes a 1-element list as a scalar; the reader must not care."""
    X = value_coded(1, 1, 1, 1)
    meta = build_meta(X, {"stim": [1]}, fs_hz=10.0, channel_names=["only"], modality_names=["calcium"])
    meta["channels"]["names"] = "only"
    meta["modalities"]["names"] = "calcium"
    meta["modalities"]["units"] = "dF/F"
    meta["trials"]["columns"][0]["values"] = 1
    path = tmp_path / "scalar.mat"
    matv73.write_mat73(path, {"prismt_format": matv73.text_to_uint8("prismt.dataset"),
                              "prismt_version": np.array(1.0), "X": X,
                              "meta_json": matv73.text_to_uint8(json.dumps(meta))})
    ds = read_dataset(path)
    assert ds.channel_names == ("only",) and ds.columns["stim"].values.tolist() == [1.0]


def test_metadata_types_and_labels_survive(tmp_path):
    X = value_coded(4, 2, 2, 1)
    trials = {
        "mouse": np.array(["M1", "M1", "M2", None], dtype=object),
        "stim": np.array([0, 1, 0, np.nan]),
        "hit": np.array([True, False, True, True]),
        "note": np.array(["café µ", "", "x", "y"], dtype=object),
    }
    path = write_dataset(tmp_path / "d.mat", X, trials, subject="mouse", fs_hz=10,
                         value_labels={"stim": {0: "CS-", 1: "CS+"}})
    ds = read_dataset(path)
    assert ds.columns["mouse"].kind == "categorical" and ds.columns["mouse"].values[3] is None
    assert ds.columns["stim"].as_text().tolist() == ["CS-", "CS+", "CS-", ""]
    assert ds.columns["hit"].kind == "bool" and ds.columns["hit"].values.dtype == bool
    assert ds.columns["note"].values[0] == "café µ"


def test_summary_reports_structure_for_the_gui(fast_dataset):
    report = validate_file(fast_dataset)
    assert report["ok"], report["errors"]
    s = report["summary"]
    assert (s["n_trials"], s["n_channels"], s["n_time"], s["n_modalities"]) == (480, 12, 10, 2)
    assert s["n_subjects"] == 8 and s["n_sessions"] == 16
    phase = next(c for c in s["columns"] if c["name"] == "phase")
    assert phase["constant_within_session"] is True and phase["values"] == ["early", "late"]
    stim = next(c for c in s["columns"] if c["name"] == "stim")
    assert stim["constant_within_session"] is False and set(stim["values"]) == {"CS-", "CS+"}
    assert s["n_tokens_full"] == 12 * 10 * 2 + 1


def test_session_keys_are_unique_per_subject(fast_dataset):
    ds = read_dataset(fast_dataset)
    keys = ds.session_keys()
    assert len(set(keys)) == 16  # 8 mice x 2 sessions, although session numbers repeat
    assert keys[0] == "M01/1"


def test_reading_data_never_imports_torch(tiny_dataset):
    code = ("import sys; from prismt.io import read_dataset; "
            f"read_dataset({str(tiny_dataset)!r}); print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_missing_file_error(tmp_path):
    with pytest.raises(DatasetError) as err:
        read_dataset(tmp_path / "nope.mat")
    assert err.value.code == "E_DATA_MISSING"
