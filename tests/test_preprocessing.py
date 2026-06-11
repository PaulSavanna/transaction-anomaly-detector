import pandas as pd
import numpy as np
import pytest
from src.preprocessing import add_features, get_features, scale_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Time': [0, 3600, 7200],
        'Amount': [100.0, 50.0, 200.0],
        'V1': [-1.0, 0.5, 1.0],
        'V2': [0.5, -0.3, 0.8],
        'Class': [0, 0, 1]
    })


def test_add_features_creates_new_columns(sample_df):
    result = add_features(sample_df)
    assert 'Amount_log' in result.columns
    assert 'Hour' in result.columns


def test_add_features_preserves_original(sample_df):
    original_cols = set(sample_df.columns)
    _ = add_features(sample_df)
    assert set(sample_df.columns) == original_cols


def test_get_features_returns_list():
    features = get_features()
    assert isinstance(features, list)
    assert 'Amount_log' in features
    assert 'Hour' in features
    assert 'V1' in features


def test_scale_features():
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    X_test = np.array([[7, 8]])

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert np.abs(X_train_scaled.mean(axis=0)).max() < 0.1
