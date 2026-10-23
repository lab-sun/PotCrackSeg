import inspect
import math
import os
import sys
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import jaccard_score, precision_recall_fscore_support

from util.util import compute_results, label2onehot

# CI / default: 200 examples. HYPOTHESIS_PROFILE=stress: max_examples >= 2000.
_COMMON = {"deadline": None, "suppress_health_check": [HealthCheck.too_slow]}
settings.register_profile("ci", max_examples=200, **_COMMON)
settings.register_profile("default", max_examples=200, **_COMMON)
settings.register_profile("stress", max_examples=2000, **_COMMON)
_PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "ci")
if _PROFILE not in ("ci", "default", "stress"):
    _PROFILE = "ci"
settings.load_profile(_PROFILE)

# @settings(max_examples=2000, deadline=None) when HYPOTHESIS_PROFILE=stress.
if _PROFILE == "stress":
    property_settings = settings(max_examples=2000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
else:
    property_settings = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def reference_compute_results(conf_total):
    """Independent per-class precision, recall, IoU and F1.

    Rows are ground truth and columns are predictions. F1 is NaN when either
    input ratio is NaN or exactly zero, matching compute_results.
    """
    conf = np.asarray(conf_total, dtype=np.float64)
    n = conf.shape[0]
    precision = np.empty(n, dtype=np.float64)
    recall = np.empty(n, dtype=np.float64)
    iou = np.empty(n, dtype=np.float64)
    f1 = np.empty(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for cid in range(n):
            tp = float(conf[cid, cid])
            predicted = float(conf[:, cid].sum())
            actual = float(conf[cid, :].sum())
            # Stable union TP + FP + FN, so a huge perfect diagonal stays finite.
            union = tp + (predicted - tp) + (actual - tp)
            precision[cid] = float("nan") if predicted == 0.0 else tp / predicted
            recall[cid] = float("nan") if actual == 0.0 else tp / actual
            iou[cid] = float("nan") if union == 0.0 else tp / union
            p = float(precision[cid])
            r = float(recall[cid])
            if math.isnan(p) or math.isnan(r) or p == 0.0 or r == 0.0:
                f1[cid] = float("nan")
            else:
                f1[cid] = 2.0 / (1.0 / p + 1.0 / r)
    return precision, recall, iou, f1


def _finite_in_unit_interval(arr):
    finite = np.isfinite(arr)
    assert not np.any(np.isinf(arr))
    assert np.all((arr[finite] >= 0.0) & (arr[finite] <= 1.0))


def _assert_metric_invariants(conf, compare_sklearn):
    precision, recall, iou, f1 = compute_results(conf)
    ref_p, ref_r, ref_i, ref_f = reference_compute_results(conf)
    n = conf.shape[0]
    assert precision.shape == (n,)
    assert recall.shape == (n,)
    assert iou.shape == (n,)
    assert f1.shape == (n,)
    np.testing.assert_allclose(precision, ref_p, equal_nan=True, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(recall, ref_r, equal_nan=True, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(iou, ref_i, equal_nan=True, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(f1, ref_f, equal_nan=True, rtol=1e-6, atol=1e-8)

    _finite_in_unit_interval(precision)
    _finite_in_unit_interval(recall)
    _finite_in_unit_interval(iou)
    _finite_in_unit_interval(f1)

    both = np.isfinite(precision) & np.isfinite(recall) & (precision > 0) & (recall > 0)
    if np.any(both):
        harmonic = 2.0 * precision[both] * recall[both] / (precision[both] + recall[both])
        np.testing.assert_allclose(f1[both], harmonic, rtol=1e-5, atol=1e-8)
    forced_nan = np.isnan(precision) | np.isnan(recall) | (precision == 0) | (recall == 0)
    assert np.all(np.isnan(f1[forced_nan]))

    defined = np.isfinite(iou) & np.isfinite(precision) & np.isfinite(recall)
    assert np.all(iou[defined] <= precision[defined] + 1e-8)
    assert np.all(iou[defined] <= recall[defined] + 1e-8)

    if n > 0 and conf.shape[0] == conf.shape[1]:
        swapped_p, swapped_r, swapped_i, swapped_f = compute_results(np.asarray(conf).T)
        np.testing.assert_allclose(precision, swapped_r, equal_nan=True, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(recall, swapped_p, equal_nan=True, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(iou, swapped_i, equal_nan=True, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(f1, swapped_f, equal_nan=True, rtol=1e-6, atol=1e-8)

    off = np.array(conf, dtype=np.float64, copy=True)
    if n:
        np.fill_diagonal(off, 0.0)
        if np.all(np.isfinite(conf)) and np.all(off == 0) and np.all(np.diag(np.asarray(conf, dtype=np.float64)) > 0):
            np.testing.assert_allclose(precision, 1.0)
            np.testing.assert_allclose(recall, 1.0)
            np.testing.assert_allclose(iou, 1.0)
            np.testing.assert_allclose(f1, 1.0)

    if compare_sklearn and n > 0 and float(np.asarray(conf).sum()) > 0:
        _assert_sklearn_agreement(np.asarray(conf, dtype=np.int64), precision, recall, iou, f1)


def _assert_sklearn_agreement(conf, precision, recall, iou, f1):
    parts_true = []
    parts_pred = []
    n = conf.shape[0]
    for i in range(n):
        for j in range(n):
            count = int(conf[i, j])
            if count:
                parts_true.append(np.full(count, i, dtype=np.int64))
                parts_pred.append(np.full(count, j, dtype=np.int64))
    y_true = np.concatenate(parts_true)
    y_pred = np.concatenate(parts_pred)
    labels = list(range(n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk_p, sk_r, sk_f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0,
        )
        if "zero_division" in inspect.signature(jaccard_score).parameters:
            sk_iou = jaccard_score(
                y_true, y_pred, labels=labels, average=None, zero_division=0,
            )
        else:
            sk_iou = jaccard_score(y_true, y_pred, labels=labels, average=None)
    col = conf.sum(axis=0)
    row = conf.sum(axis=1)
    tp = np.diag(conf)
    union = row + col - tp
    np.testing.assert_allclose(precision[col > 0], sk_p[col > 0], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(recall[row > 0], sk_r[row > 0], rtol=1e-6, atol=1e-8)
    both_pos = (col > 0) & (row > 0) & (tp > 0)
    np.testing.assert_allclose(f1[both_pos], sk_f[both_pos], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(iou[union > 0], sk_iou[union > 0], rtol=1e-6, atol=1e-8)
    absent_pred = col == 0
    absent_true = row == 0
    assert np.all(np.isnan(precision[absent_pred]))
    assert np.all(np.isnan(recall[absent_true]))
    assert np.all(np.isnan(iou[union == 0]))


def _square_counts(min_n, max_n, max_count):
    return st.integers(min_n, max_n).flatmap(
        lambda n: arrays(
            dtype=np.int64,
            shape=(n, n),
            elements=st.integers(min_value=0, max_value=max_count),
        )
    )


_float_elements = st.one_of(
    st.just(0.0),
    st.just(1.0),
    st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False, width=64),
    st.just(1e308),
    st.just(np.inf),
    st.just(np.nan),
)


def _square_floats():
    return st.integers(1, 3).flatmap(
        lambda n: arrays(dtype=np.float64, shape=(n, n), elements=_float_elements)
    )


@property_settings
@given(conf=_square_counts(0, 4, 25))
@example(conf=np.zeros((0, 0), dtype=np.int64))
@example(conf=np.zeros((1, 1), dtype=np.int64))
@example(conf=np.zeros((3, 3), dtype=np.int64))
@example(conf=np.array([[5]], dtype=np.int64))
@example(conf=np.array([[12, 0, 0], [0, 7, 0], [0, 0, 9]], dtype=np.int64))
@example(conf=np.array([[50, 10, 5], [8, 40, 2], [4, 6, 30]], dtype=np.int64))
@example(conf=np.array([[10, 0, 0], [0, 5, 0], [0, 0, 0]], dtype=np.int64))
@example(conf=np.array([[10, 2, 0], [3, 0, 1], [0, 1, 8]], dtype=np.int64))
@example(conf=np.array([[np.int64(2**62), np.int64(2**62)], [np.int64(0), np.int64(1)]]))
@example(conf=np.array([[2**24, 1], [0, 1]], dtype=np.int64))
def test_compute_results_integer_confusion(conf):
    # Huge counts are checked against the reference only. Expanding them into
    # sklearn label vectors is not bounded.
    total = float(np.asarray(conf, dtype=np.float64).sum()) if conf.size else 0.0
    _assert_metric_invariants(conf, compare_sklearn=0 < total <= 500)


@property_settings
@given(conf=_square_floats())
@example(conf=np.zeros((1, 1), dtype=np.float64))
@example(conf=np.zeros((3, 3), dtype=np.float64))
@example(conf=np.full((2, 2), np.nan))
@example(conf=np.full((2, 2), np.inf))
@example(conf=np.array([[1e308, 1e308], [1e308, 1e308]], dtype=np.float64))
@example(conf=np.array([[1e308, 0.0], [0.0, 1.0]], dtype=np.float64))
@example(conf=np.array([[1e-320, 0.0], [1e308, 1.0]], dtype=np.float64))
@example(conf=np.array([[1e-320, 1e308], [0.0, 1.0]], dtype=np.float64))
@example(conf=np.array([[np.nan, 0.0], [1.0, 2.0]], dtype=np.float64))
@example(conf=np.array([[np.inf, 1.0], [0.0, np.nan]], dtype=np.float64))
def test_compute_results_nonfinite_and_huge(conf):
    _assert_metric_invariants(conf, compare_sklearn=False)


def _label_cases():
    return st.integers(1, 5).flatmap(
        lambda n_class: st.integers(1, 6).flatmap(
            lambda height: st.integers(1, 6).flatmap(
                lambda width: arrays(
                    dtype=np.uint8,
                    shape=(height, width),
                    elements=st.integers(min_value=0, max_value=n_class - 1),
                ).map(lambda labels: (n_class, labels))
            )
        )
    )


@property_settings
@given(case=_label_cases())
@example(case=(1, np.zeros((1, 1), dtype=np.uint8)))
@example(case=(3, np.zeros((2, 2), dtype=np.uint8)))
@example(case=(3, np.array([[0, 1], [2, 0]], dtype=np.uint8)))
@example(case=(1, np.array([[0, 0, 0]], dtype=np.uint8)))
def test_label2onehot_is_a_float_partition(case):
    n_class, labels = case
    height, width = labels.shape
    out = label2onehot(labels, n_class, width, height)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert tuple(out.shape) == (n_class, height, width)
    assert torch.all((out == 0) | (out == 1))
    assert torch.equal(out.sum(dim=0), torch.ones(height, width))
    assert torch.equal(out.argmax(dim=0).to(torch.int64), torch.as_tensor(labels).long())


def test_property_profile_budgets():
    assert settings.get_profile("ci").max_examples == 200
    assert settings.get_profile("default").max_examples == 200
    assert settings.get_profile("stress").max_examples >= 2000
    assert settings.get_profile("stress").deadline is None
    assert settings.get_profile("ci").deadline is None
