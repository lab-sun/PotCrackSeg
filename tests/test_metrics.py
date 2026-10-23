import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Repository root, so `util` imports when pytest collects this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings

import numpy as np
import pytest
import torch

from util.util import compute_results, get_palette, label2onehot


@pytest.fixture(autouse=True)
def fail_on_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def test_get_palette_shape():
    palette = get_palette()
    assert palette.shape == (5, 3)


def test_perfect_huge_diagonal_stays_one():
    conf = np.array([
        [1e308, 0.0],
        [0.0, 1.0],
    ])
    precision, recall, iou, f1 = compute_results(conf)
    np.testing.assert_allclose(precision, [1.0, 1.0])
    np.testing.assert_allclose(recall, [1.0, 1.0])
    np.testing.assert_allclose(iou, [1.0, 1.0])
    np.testing.assert_allclose(f1, [1.0, 1.0])


def test_perfect_prediction():
    conf = np.array([
        [12, 0, 0],
        [0, 7, 0],
        [0, 0, 9],
    ])
    precision, recall, iou, f1 = compute_results(conf)
    np.testing.assert_allclose(precision, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(recall, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(iou, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(f1, [1.0, 1.0, 1.0])


def test_hand_computed_confusion_matrix():
    # Rows are ground truth, columns are prediction.
    conf = np.array([
        [50, 10, 5],
        [8, 40, 2],
        [4, 6, 30],
    ])
    precision, recall, iou, f1 = compute_results(conf)

    # class 0: TP 50, column sum 62, row sum 65, union 77
    # class 1: TP 40, column sum 56, row sum 50, union 66
    # class 2: TP 30, column sum 37, row sum 40, union 47
    expected_precision = np.array([50 / 62, 40 / 56, 30 / 37])
    expected_recall = np.array([50 / 65, 40 / 50, 30 / 40])
    expected_iou = np.array([50 / 77, 40 / 66, 30 / 47])
    expected_f1 = np.array([
        2 / (1 / (50 / 62) + 1 / (50 / 65)),
        2 / (1 / (40 / 56) + 1 / (40 / 50)),
        2 / (1 / (30 / 37) + 1 / (30 / 40)),
    ])
    np.testing.assert_allclose(precision, expected_precision)
    np.testing.assert_allclose(recall, expected_recall)
    np.testing.assert_allclose(iou, expected_iou)
    np.testing.assert_allclose(f1, expected_f1)


def test_class_with_no_samples_and_no_predictions_is_nan():
    conf = np.array([
        [10, 0, 0],
        [0, 5, 0],
        [0, 0, 0],
    ])
    precision, recall, iou, f1 = compute_results(conf)
    assert np.isnan(precision[2])
    assert np.isnan(recall[2])
    assert np.isnan(iou[2])
    assert np.isnan(f1[2])


def test_zero_true_positives_f1_is_nan():
    conf = np.array([
        [10, 2, 0],
        [3, 0, 1],
        [0, 1, 8],
    ])
    precision, recall, iou, f1 = compute_results(conf)
    assert precision[1] == 0.0
    assert recall[1] == 0.0
    assert iou[1] == 0.0
    assert np.isnan(f1[1])


def test_int64_counts_do_not_overflow_metrics():
    # 2**62 + 2**62 overflows int64 to a negative row sum.
    conf = np.array([
        [np.int64(2**62), np.int64(2**62)],
        [np.int64(0), np.int64(1)],
    ], dtype=np.int64)
    precision, recall, iou, f1 = compute_results(conf)
    assert precision[0] == 1.0
    assert recall[0] == 0.5
    assert iou[0] == 0.5
    assert f1[0] == pytest.approx(2 / (1 / 1.0 + 1 / 0.5))
    assert recall[1] == 1.0
    assert 0.0 < precision[1] < 1.0
    assert np.isfinite(f1[1])

    # 2**24 + 1 is not exact in float32, so the reduction must be float64.
    narrow = np.array([[2**24, 1], [0, 1]], dtype=np.int64)
    precision, recall, iou, f1 = compute_results(narrow)
    # Column sum is the exact power of two, so precision stays 1.
    # Row sum is 2**24 + 1, which float32 rounds away and float64 keeps.
    assert precision[0] == 1.0
    assert recall[0] == pytest.approx((2**24) / (2**24 + 1))
    assert recall[0] < 1.0
    assert iou[0] == pytest.approx((2**24) / (2**24 + 1))
    assert f1[0] == pytest.approx(2 / (1 / precision[0] + 1 / recall[0]))


def test_f1_is_nan_when_only_one_ratio_underflows_to_zero():
    # Column sum overflows the mantissa of the tiny true positive: precision is 0, recall is 1.
    conf = np.array([
        [1e-320, 0.0],
        [1e308, 1.0],
    ])
    precision, recall, iou, f1 = compute_results(conf)
    assert precision[0] == 0.0
    assert recall[0] == 1.0
    assert iou[0] == 0.0
    assert np.isnan(f1[0])

    # Row sum does the same to recall while precision stays 1.
    conf = np.array([
        [1e-320, 1e308],
        [0.0, 1.0],
    ])
    precision, recall, iou, f1 = compute_results(conf)
    assert precision[0] == 1.0
    assert recall[0] == 0.0
    assert iou[0] == 0.0
    assert np.isnan(f1[0])


def test_class_count_follows_confusion_rows():
    conf = np.array([
        [5, 1, 4],
        [2, 3, 0],
    ])
    precision, recall, iou, f1 = compute_results(conf)
    assert precision.shape == (2,)
    assert precision[0] == pytest.approx(5 / 7)
    assert recall[0] == pytest.approx(5 / 10)
    assert iou[0] == pytest.approx(5 / 12)
    assert f1[0] == pytest.approx(2 / (1 / (5 / 7) + 1 / 0.5))
    assert precision[1] == pytest.approx(3 / 4)
    assert recall[1] == pytest.approx(3 / 5)
    assert iou[1] == pytest.approx(0.5)
    assert f1[1] == pytest.approx(2 / (1 / (3 / 4) + 1 / (3 / 5)))


def test_label2onehot_returns_float_mask_for_uint8_ids():
    label = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    out = label2onehot(label, n_class=3, input_w=2, input_h=2)
    assert out.dtype == torch.float32
    assert out.shape == (3, 2, 2)
    assert torch.equal(out.sum(dim=0), torch.ones(2, 2))
    expected = torch.tensor([
        [[1, 0], [0, 1]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]],
    ], dtype=torch.float32)
    assert torch.equal(out, expected)
