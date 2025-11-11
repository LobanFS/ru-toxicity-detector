# RU Toxicity Detector

Классификация токсичных сообщений на русском: baseline (TF-IDF + Logistic Regression) и современная модель (RuBERT-Tiny).

## Задача
Бинарная классификация: toxic vs non-toxic.
Цель проекта: показать понятный baseline, улучшить качество трансформером и проверить переносимость на другой домен.

## Датасеты
Датасеты не хранятся в репозитории — скачайте их вручную и положите в папку data/.

OK.ru Russian Toxic Comments — основной train/val/test (https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments)
Pikabu / 2ch Russian Toxic Comments — внешний тест (https://www.kaggle.com/datasets/aybatov/toxic-russian-comments-from-pikabu-and-2ch)

Файлы должны лежать так:
data/okru/okru_raw.txt
data/pikabu2ch/pikabu2ch_raw.csv

## Подготовка данных
OK.ru:
python src/data/okru/prepare_okru.py
python src/data/okru/split_okru.py
Pikabu:
python src/data/pikabu/prepare_pikabu.py
python src/data/pikabu/split_pikabu.py

После этого появятся CSV:
data/okru/train.csv, val.csv, test.csv
data/pikabu2ch/val.csv, test.csv

## Быстрый старт
## 1) Установите зависимости
pip install -r requirements.txt

## 1.5) Используйте готовые модели ( опционально )
python src/main.py 
если модель с трансформером не лежит в локальных файлах, она устанавливается из huggingface_hub.

## 2) Обучите baseline
python src/baseline/train_baseline.py

## 3) Запустите инференс через меню
python src/predict_cli/predict_baseline_cli.py

## 4) Модели и хранение
После обучения baseline сохраняется в:
models/baseline.joblib
models/baseline.threshold.json

## 5) Трансформер (RuBERT-Tiny) обучается в ноутбуке src/transformer/training_transformer.ipynb (в Google Colab).
После обучения скачайте модель в формате Hugging Face и поместите в models/rubert-tiny/.
В папке должны быть стандартные файлы config.json, model.safetensors, tokenizer.json, tokenizer_config.json, vocab.txt.
Можно добавить файл inference_config.json для инференса с параметрами threshold_global, temperature и max_length.

## 6) Инференс
Универсальное меню (выбор baseline или rubert-tiny):
python src/main.py
При старте выбрать:
1 — baseline (TF-IDF + LogisticRegression)
2 — rubert-tiny (трансформер)
Далее вводите текст построчно. Команда stop завершает работу.

## Оценка
Обучение — OK.ru (train + val)
Тест — Pikabu (test)
Метрики — F1 (основная), ROC-AUC, PR-AUC.
Цель — максимизация F1 для токсичного класса (1).

## Основные файлы проекта
Подготовка данных: prepare_okru.py, split_okru.py, prepare_pikabu.py, split_pikabu.py
Обучение baseline: train_baseline.py, tune_baseline.py
Обучение трансформера: training_transformer.ipynb
Инференс: predict_cli.py, script_for_help.py, predict_rubert_cli.py

## Примечания
Baseline использует TF-IDF по словам (1–2-граммы) и символам (3–5-граммы) + LogisticRegression с class_weight=balanced.
Порог классификации подбирается по OK.ru val (максимизация F1).
Трансформер основан на cointegrated/rubert-tiny2 c temperature_scaling.
Все модели и артефакты хранятся в папке models/.