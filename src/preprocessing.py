import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Загрузка данных"""
    return pd.read_csv(path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавление производных признаков"""
    df = df.copy()
    df['Amount_log'] = np.log1p(df['Amount'])
    df['Hour'] = (df['Time'] / 3600).astype(int)
    return df


def get_features() -> list[str]:
    """Список признаков для модели"""
    return [
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
        'Amount', 'Amount_log', 'Hour'
    ]


def scale_features(X_train, X_test):
    """Масштабирование признаков"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
