"""Генерация сэмпла данных для демонстрации"""
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

data = {
    'Time': np.random.uniform(0, 172800, n_samples),
    'Amount': np.random.exponential(100, n_samples),
    'Class': np.zeros(n_samples, dtype=int)
}

for i in range(29):
    data[f'V{i+1}'] = np.random.randn(n_samples)

anomaly_idx = np.random.choice(n_samples, 3, replace=False)
data['Class'][anomaly_idx] = 1
data['Amount'][anomaly_idx] *= 10

df = pd.DataFrame(data)
df.to_csv('data/sample.csv', index=False)
