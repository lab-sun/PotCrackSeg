import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Repository root, so `util` imports when pytest collects this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings

import numpy as np
import pytest

from util.util import compute_results, get_palette


@pytest.fixture(autouse=True)
def fail_on_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def test_get_palette_shape():
    palette = get_palette()
    assert palette.shape == (5, 3)


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
    assert np.isnan(f1[1])
