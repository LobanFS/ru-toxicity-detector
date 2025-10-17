# RU Toxicity Detector

Классификация токсичных сообщений на русском: baseline (TF-IDF + Logistic Regression) и современная модель (RuBERT/DistilBERT).

## Задача
Бинарная классификация: `toxic` vs `non-toxic`. Цель — показать:
- понятный baseline,
- улучшение с трансформером,
- переносимость на другой домен (кросс-доменная проверка).

## Датасеты (ссылки и инструкции)
> Датасеты в репозиторий **не коммитятся**. Скачайте локально в `data/`.

- **OK.ru Russian Toxic Comments** — основной train/val/test.
- **2ch + Pikabu Russian Toxic Comments** — внешняя проверка (domain shift).

В `data/README.md` будут шаги загрузки/распаковки.

## Быстрый старт
```bash
# 1) окружение
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2) baseline
python src/train_baseline.py --data_dir data/okru --out models/baseline.joblib

# 3) проверка baseline на внешнем наборе
python src/eval.py --model models/baseline.joblib --data_dir data/pikabu2ch

# 4) инференс одной строки
python src/infer.py --text "ты дурак" --model models/baseline.joblib
