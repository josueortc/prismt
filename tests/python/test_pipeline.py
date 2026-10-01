"""Config, selection, preprocessing, splits, and whole runs on synthetic data."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import scipy.io

from prismt.config import load_config
from prismt.data.preprocess import Normalizer, make_time_bins
from prismt.data.splits import LeakageError, assert_no_leakage, plan_splits
from prismt.errors import ConfigError, SplitError
from prismt.io import read_dataset
from prismt.run import check, prepare, run


def cfg_for(path, **over):
    base = {"task": "classify", "dataset": {"path": str(path)}, "labels": {"column": "phase"},
            "train": {"epochs": 3, "min_epochs": 0, "device": "cpu"}, "model": {"d_model": 16, "n_layers": 1}}
    for k, v in over.items():
        base.setdefault(k, {}) if isinstance(v, dict) else None
        if isinstance(v, dict):
            base[k] = {**base.get(k, {}), **v}
        else:
            base[k] = v
    return load_config(base)


# --- config -------------------------------------------------------------------------------

def test_config_merges_preset_and_user_values(tmp_path):
    c = load_config({"dataset": {"path": "d.mat"}, "labels": {"column": "x"}, "preset": "standard",
                     "train": {"lr": 0.01}}, base_dir=tmp_path)
    assert c["model"]["d_model"] == 128 and c["train"]["lr"] == 0.01
    assert c["dataset"]["path"] == str((tmp_path / "d.mat").resolve())


def test_config_reports_every_problem_with_suggestions():
    with pytest.raises(ConfigError) as err:
        load_config({"dataset": {"path": "d"}, "trian": {}, "model": {"n_heads": 3}})
    codes = [d["code"] for d in err.value.details]
    assert {"E_CFG_UNKNOWN", "E_CFG_HEADS", "E_CFG_NO_LABEL"} <= set(codes)
    assert "Did you mean 'train'" in err.value.details[0]["hint"]


def test_one_element_lists_from_matlab_are_accepted():
    c = load_config({"dataset": {"path": "d"}, "labels": {"column": "stim", "classes": {"name": "a", "values": 1}},
                     "selection": {"filters": {"column": "phase", "values": "late"}}})
    assert c["labels"]["classes"] == [{"name": "a", "values": [1]}]
    assert c["selection"]["filters"][0]["values"] == ["late"]


# --- selection and preprocessing ------------------------------------------------------------

def test_filters_classes_and_labels(fast_dataset):
    c = cfg_for(fast_dataset, labels={"column": "response", "classes": [{"name": "correct", "values": ["hit", "CR"]},
                                                                        {"name": "error", "values": [0, 3]}]},
                selection={"filters": [{"column": "phase", "values": ["late"]}]})
    prep = prepare(c)
    ds = read_dataset(fast_dataset)
    resp = ds.columns["response"].values[prep.sel.trial_index]
    assert set(ds.columns["phase"].as_text()[prep.sel.trial_index]) == {"late"}
    assert np.array_equal(prep.sel.y == 0, np.isin(resp, [1, 2]))


def test_unknown_value_lists_what_is_there(fast_dataset):
    with pytest.raises(ConfigError) as err:
        prepare(cfg_for(fast_dataset, selection={"filters": [{"column": "phase", "values": ["middle"]}]}))
    assert "early, late" in err.value.hint


def test_time_bins_average_exactly():
    times = np.arange(10) / 10.0
    bins = make_time_bins(times, [0.0, 0.5], 0.2)
    assert [list(m) for m in bins.members] == [[0, 1], [2, 3], [4, 5]]
    with pytest.raises(ConfigError):
        make_time_bins(times, None, 0.01)


def test_normalization_ignores_test_trials():
    X = np.random.default_rng(0).standard_normal((20, 2, 3, 1)).astype(np.float32)
    a = Normalizer.fit(X, np.arange(10), "zscore_train", None)
    X[10:] = 1e6
    b = Normalizer.fit(X, np.arange(10), "zscore_train", None)
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_allclose(b.inverse(b.transform(X[:10])), X[:10], rtol=1e-5)


# --- splits -------------------------------------------------------------------------------

def _groups(n_mice=8, sessions=2, trials=10):
    mouse = np.repeat([f"M{i}" for i in range(n_mice)], sessions * trials).astype(object)
    sess = np.array([f"{m}/{(i // trials) % sessions}" for i, m in enumerate(mouse)], dtype=object)
    phase = np.array([(i // trials) % sessions for i in range(len(mouse))])
    return mouse, sess, phase


SPLIT = load_config({"dataset": {"path": "x"}, "labels": {"column": "x"}})["split"]


@pytest.mark.parametrize("seed", range(5))
def test_mice_never_cross_between_parts(seed):
    mouse, sess, phase = _groups(12)
    plan = plan_splits(phase, mouse, sess, dict(SPLIT, seed=seed))
    for f in plan.folds:
        assert not set(mouse[f.test]) & set(mouse[f.train]) and not set(mouse[f.test]) & set(mouse[f.val])
        assert set(phase[f.train]) == {0, 1}


def test_assignment_does_not_depend_on_row_order():
    mouse, sess, phase = _groups(12)
    a = plan_splits(phase, mouse, sess, SPLIT)
    perm = np.random.default_rng(1).permutation(len(mouse))
    b = plan_splits(phase[perm], mouse[perm], sess[perm], SPLIT)
    assert set(mouse[a.folds[0].test]) == set(mouse[perm][b.folds[0].test])


def test_session_level_label_refuses_trial_split():
    mouse, sess, phase = _groups()
    with pytest.raises(SplitError) as err:
        plan_splits(phase, mouse, sess, dict(SPLIT, test_on="trial"), label_name="phase")
    assert err.value.code == "E_LEAKY_SPLIT"
    plan = plan_splits(phase, mouse, sess, dict(SPLIT, test_on="trial", allow_leaky=True))
    assert plan.flags["leaky_split"]


def test_few_mice_use_leave_one_out_and_test_each_once():
    mouse, sess, phase = _groups(5)
    plan = plan_splits(phase, mouse, sess, SPLIT)
    assert plan.scheme == "loo" and len(plan.folds) == 5
    tested = np.concatenate([f.test for f in plan.folds])
    assert sorted(tested.tolist()) == list(range(len(mouse)))


def test_subject_level_label_needs_two_animals_per_class():
    mouse, sess, _ = _groups(3)
    genotype = (mouse == "M0").astype(int)
    with pytest.raises(SplitError) as err:
        plan_splits(genotype, mouse, sess, SPLIT)
    assert err.value.code == "E_SPLIT_INFEASIBLE"


def test_leakage_check_catches_a_shared_mouse():
    mouse, sess, phase = _groups(12)
    plan = plan_splits(phase, mouse, sess, SPLIT)
    f = plan.folds[0]
    f.train = np.append(f.train, f.test[0])
    with pytest.raises(LeakageError):
        assert_no_leakage(plan, {"subject": mouse, "session": sess, "trial": None})


# --- whole runs ---------------------------------------------------------------------------

def test_check_reports_plan_and_size(fast_dataset):
    rep = check(cfg_for(fast_dataset))
    assert rep["selection"]["class_counts"] == [240, 240]
    assert rep["split"]["test_on"] == "subject" and rep["model"]["tokens_per_trial"] == 241


@pytest.mark.slow
def test_classification_run_writes_the_contract(tmp_path, tiny_dataset):
    out = run(cfg_for(tiny_dataset, output={"root": str(tmp_path)}, labels={"column": "stim"}))
    for f in ("config.json", "status.json", "history.csv", "metrics.json", "splits.csv", "predictions.csv",
              "results.mat", "run_info.json", "log.txt"):
        assert (out / f).exists(), f
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "finished"
    m = json.loads((out / "metrics.json").read_text())
    assert m["headline"]["name"] == "balanced_accuracy" and m["summary_lines"]
    mat = scipy.io.loadmat(out / "results.mat")
    assert mat["prob"].shape[1] == 2 and mat["y_true"].min() >= 1
    # one summary space per fold model: PCs are computed within folds and labelled by fold
    folds = mat["embedding_fold"].ravel()
    assert set(folds) == set(range(1, int(folds.max()) + 1)) and len(folds) == mat["embedding_pcs"].shape[0]
    assert mat["embedding_explained"].shape[0] == int(folds.max())
    for f in set(folds):
        assert abs(mat["embedding_pcs"][folds == f, 0].mean()) < 1e-6, "centred within each fold"
    with open(out / "splits.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert min(int(r["trial_index"]) for r in rows) >= 1


def test_shuffled_labels_move_whole_groups():
    from prismt.eval.baselines import permuted_labels

    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 5)
    y = np.repeat(np.arange(10) % 2, 5)           # label constant within each group
    for _ in range(5):
        p = permuted_labels(y, groups, rng)
        assert all(len(set(p[groups == g])) == 1 for g in range(10)), "a group was split"
        assert np.bincount(p).tolist() == np.bincount(y).tolist()
    assert sorted(permuted_labels(y, None, rng)) == sorted(y)


@pytest.mark.slow
def test_shuffled_label_baseline_sits_at_chance(tmp_path, tiny_dataset):
    out = run(cfg_for(tiny_dataset, output={"root": str(tmp_path)}, labels={"column": "stim"},
                      baselines={"permutations": 20}))
    m = json.loads((out / "metrics.json").read_text())
    sh = m["baselines"]["shuffled_labels"]
    assert sh["n"] == 20 and 0.3 < sh["balanced_accuracy"] < 0.7
    assert 0 < sh["p_value_model"] <= 1 and sh["shuffled_by"] == "trial"
    assert any("shuffled labels" in line for line in m["summary_lines"])
    assert scipy.io.loadmat(out / "results.mat")["shuffled_null"].size == 20


@pytest.mark.slow
def test_mae_then_finetune(tmp_path, tiny_dataset):
    mae = run(cfg_for(tiny_dataset, task="mae", output={"root": str(tmp_path)}))
    mat = scipy.io.loadmat(mae / "results.mat")
    assert mat["r2_by_pair"].shape[1] == 12 and "example_original" in mat
    ft = run(cfg_for(tiny_dataset, output={"root": str(tmp_path)}, model={"init_from": str(mae)}))
    m = json.loads((ft / "metrics.json").read_text())
    assert m["flags"]["inherited_from"] == str(mae)


@pytest.mark.slow
def test_poisoned_test_trials_do_not_change_training(tmp_path, fast_dataset):
    """Setting every test trial to garbage must leave the training history identical."""
    from prismt.data.synthetic import make_synthetic
    from prismt.io import write_dataset

    c = cfg_for(fast_dataset, output={"root": str(tmp_path)})
    a = run(c)
    test = [int(r["trial_index"]) - 1 for r in csv.DictReader(open(a / "splits.csv")) if r["split"] == "test"
            and r["fold"] == "1"]
    d = make_synthetic("fast", "medium", 0)
    d["X"][test] = 50.0
    X, trials = d.pop("X"), d.pop("trials")
    bad = write_dataset(tmp_path / "poisoned.mat", X, trials, **d)
    b = run(cfg_for(bad, output={"root": str(tmp_path)}))
    ha = [r for r in csv.DictReader(open(a / "history.csv")) if r["fold"] == "1"]
    hb = [r for r in csv.DictReader(open(b / "history.csv")) if r["fold"] == "1"]
    assert [r["train_loss"] for r in ha] == [r["train_loss"] for r in hb]
    assert [r["val_loss"] for r in ha] == [r["val_loss"] for r in hb]
