import numpy as np
import pytest
from src.evaluate import find_optimal_threshold, evaluate_model


def test_find_optimal_threshold():
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    threshold = find_optimal_threshold(y_true, scores)
    assert 0 < threshold < 1


def test_evaluate_model_returns_f1():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0])

    result = evaluate_model(y_true, y_pred)
    assert 'f1' in result
    assert 0 <= result['f1'] <= 1
