import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import load_data, add_features, get_features, scale_features
from src.models import get_isolation_forest
from src.evaluate import evaluate_model, find_optimal_threshold


def train_and_save(data_path: str, model_dir: str = "models"):
    """Обучение и сохранение модели"""
    Path(model_dir).mkdir(exist_ok=True)

    df = load_data(data_path)
    df = add_features(df)
    features = get_features()

    X = df[features]
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = get_isolation_forest()
    model.fit(X_train_scaled)

    scores = -model.decision_function(X_test_scaled)
    threshold = find_optimal_threshold(y_test, scores)

    preds = (scores > threshold).astype(int)
    evaluate_model(y_test, preds, scores)

    with open(f"{model_dir}/isolation_forest.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open(f"{model_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(f"{model_dir}/threshold.pkl", 'wb') as f:
        pickle.dump(threshold, f)

    print(f"\nModel saved to {model_dir}/")
    print(f"Threshold: {threshold:.6f}")


if __name__ == "__main__":
    train_and_save("data/raw/creditcard.csv")
