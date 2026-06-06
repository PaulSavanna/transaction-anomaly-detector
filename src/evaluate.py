from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
    f1_score
)
import numpy as np


def find_optimal_threshold(y_true, scores):
    """Поиск оптимального порога по F1"""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    # избегаем деления на ноль
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]


def evaluate_model(y_true, y_pred, scores=None):
    """Полная оценка модели"""
    print(classification_report(y_true, y_pred))

    if scores is not None:
        pr_auc = average_precision_score(y_true, scores)
        print(f"PR-AUC: {pr_auc:.4f}")

    f1 = f1_score(y_true, y_pred)
    print(f"F1: {f1:.4f}")

    return {'f1': f1}
