import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from src.data.pathing import BASE, DATA
from src.utils import iter_clean

OKRU_TRAIN = DATA / "okru" / "train.csv"
PIKABU_VAL = DATA / "pikabu2ch" / "val.csv"
PIKABU_TEST = DATA / "pikabu2ch" / "test.csv"

OUT_DIR = DATA / "models" /" baseline_tuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "model.joblib"
CFG_PATH = OUT_DIR / "best_config.json"


def load_xy(path: Path):
    df = pd.read_csv(path)
    X = list(iter_clean(df["text"].tolist()))
    y = df["label"].astype(int).values
    return X, y


def build_pipeline():
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_features=200_000,
        sublinear_tf=True
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=300_000,
        sublinear_tf=True
    )
    fe = FeatureUnion([("word", word), ("char", char)])
    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        n_jobs=1
    )
    pipe = Pipeline([("featurizer", fe), ("clf", clf)])
    return pipe


def fit_grid_search(Xtr, ytr):
    pipe = build_pipeline()

    param_grid = {
        "featurizer__word__ngram_range": [(1,1), (1,2)],
        "featurizer__char__ngram_range": [(3,4), (3,5)],
        "featurizer__word__min_df": [2, 3],
        "clf__C": [0.5, 2.0, 3.0],
    }
    grid = GridSearchCV(
        pipe, param_grid=param_grid,
        scoring="f1",
        cv=3, n_jobs=-1, verbose=2, refit=True
    )
    grid.fit(Xtr, ytr)
    return grid


def choose_best_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1s = (2 * precision * recall) / (precision + recall + 1e-12)
    # последний элемент precision/recall без threshold отбрасываем
    thr = thresholds[np.argmax(f1s[:-1])] if len(thresholds) else 0.5
    return float(thr)


def evaluate(name, y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    print(f"\n== {name} (thr={thr:.3f}) ==")
    print(classification_report(y_true, pred, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_true, proba):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_true, proba):.4f}")
    print(f"F1(1):   {f1_score(y_true, pred):.4f}")


def main():
    Xtr, ytr = load_xy(OKRU_TRAIN)
    Xva, yva = load_xy(PIKABU_VAL)
    Xte, yte = load_xy(PIKABU_TEST)

    grid = fit_grid_search(Xtr, ytr)
    best = grid.best_estimator_
    print("\nBest params:", grid.best_params_)

    joblib.dump(best, MODEL_PATH)
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_params": grid.best_params_}, f, ensure_ascii=False, indent=2)
    print(f"saved model to {MODEL_PATH}")

    proba_val = best.predict_proba(Xva)[:, 1]
    thr = choose_best_threshold(yva, proba_val)
    print(f"\nChosen threshold on VAL (max F1 for class=1): {thr:.3f}")

    evaluate("VAL", yva, proba_val, thr=thr)
    proba_test = best.predict_proba(Xte)[:, 1]
    evaluate("TEST", yte, proba_test, thr=thr)

    with open(OUT_DIR / "threshold.txt", "w", encoding="utf-8") as f:
        f.write(str(thr))
    print(f"saved threshold to {OUT_DIR / 'threshold.txt'}")


if __name__ == "__main__":
    main()
