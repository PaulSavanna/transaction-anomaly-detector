import numpy as np
import pytest
from src.models import Autoencoder, get_isolation_forest


def test_isolation_forest_creation():
    model = get_isolation_forest()
    assert model.contamination == 0.002


def test_isolation_forest_fit_predict():
    X = np.random.randn(1000, 30)
    model = get_isolation_forest()
    model.fit(X)
    preds = model.predict(X)
    assert len(preds) == 1000
    assert set(np.unique(preds)).issubset({-1, 1})


def test_autoencoder_output_shape():
    model = Autoencoder(input_dim=30)
    x = torch.randn(10, 30)
    output = model(x)
    assert output.shape == (10, 30)


import torch


def test_autoencoder_encoding():
    model = Autoencoder(input_dim=30)
    x = torch.randn(10, 30)
    encoded = model.encoder(x)
    assert encoded.shape == (10, 8)
