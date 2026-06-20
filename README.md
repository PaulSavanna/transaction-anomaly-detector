# Transaction Anomaly Detector

Система детекции аномалий в финансовых транзакциях. Использует Isolation Forest и Autoencoder для поиска мошеннических транзакций в датасете кредитных карт.

## Результаты

| Модель | PR-AUC | F1 |
|--------|--------|-----|
| Isolation Forest (baseline) | 0.32 | 0.38 |
| Isolation Forest (tuned) | 0.45 | 0.42 |
| Autoencoder | 0.51 | 0.44 |

**Финальный выбор:** Isolation Forest с contamination=0.003 — лучший баланс между precision и recall, плюс проще в продакшене.

## Быстрый старт

```bash
# Установка
pip install -r requirements.txt

# Обучение модели
python -m src.train

# Запуск API
uvicorn src.api:app --reload
```

## API

```
POST /predict
{
    "V1": -1.359, "V2": -0.072, ...,
    "Amount": 149.62
}

Response:
{
    "anomaly_score": 0.234,
    "is_anomaly": false,
    "threshold": 0.156
}
```

## Approach & Trade-offs

### Почему Isolation Forest
- Работает быстро на полном датасете (~280k строк)
- Не требует обучения на "нормальных" данных (в отличие от автоэнкодера)
- Простой в продакшене — один pickle файл

### Почему не One-Class SVM
- O(n^2) по памяти — на 280k строках не реалистично
- На подвыборке (50k) показал худшие результаты

### Почему не автоэнкодер как основной
- Лучший PR-AUC (0.51 vs 0.45), но:
  - Требует PyTorch в продакшене
  - Сложнее в отладке
  - Менее интерпретируемый

## What Didn't Work

1. **Rolling average features** — ухудшили метрики, вероятно из-за шума
2. **One-Class SVM** — слишком медленно на полном датасете
3. **Accuracy как метрика** — бесполезна при 0.17% аномалий

## Структура проекта

```
├── data/
│   ├── README.md              # инструкции по скачиванию данных
│   ├── sample.csv             # сэмпл для демонстрации
│   └── generate_sample.py     # скрипт генерации сэмпла
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_comparison.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluate.py
│   ├── train.py
│   └── api.py
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluate.py
├── models/                    # сгенерированные модели (не в git)
├── requirements.txt
└── .github/workflows/ci.yml
```