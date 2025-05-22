import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics

def test_train_model_returns_classifier():
    """
    Test that train_model returns a RandomForestClassifier instance.
    """
    X = np.array([[1, 0], [0, 1], [1, 1]])
    y = np.array([0, 1, 0])
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_inference_output_matches_input():
    """
    Test that inference returns a prediction array with the correct length.
    """
    X = np.array([[1, 0], [0, 1]])
    y = np.array([0, 1])
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]

def test_compute_model_metrics_on_known_data():
    """
    Test compute_model_metrics for a known set of true/predicted labels.
    """
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert round(precision, 2) == 1.00
    assert round(recall, 2) == 0.67
    assert round(fbeta, 2) == 0.80
