import json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, precision_recall_curve)
from src.data.pathing import BASE, DATA
from src.utils import iter_clean

# тест + валидация на okru, тест ( переход на новый домен ) - на pikabu.

TRAIN = DATA / "okru" / "train.csv"
VAL   = DATA / "okru" / "val.csv"
TEST  = DATA / "pikabu2ch" / "pikabu2ch_normalised.csv"

OUT   = BASE / "models" / "baseline.joblib"
OUT.parent.mkdir(parents=True, exist_ok=True)
THR_PATH = OUT.with_suffix(".threshold.json")

def load_dataset(path: Path):
    df = pd.read_csv(path)
    X = list(iter_clean(df["text"].astype(str).tolist()))
    y = df["label"].astype(int).values
    return X, y

def best_f1_threshold(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[f1[:-1].argmax()]) if len(t) else 0.5

def evaluate(name, y, proba, thr):
    pred = (proba >= thr).astype(int)
    print(f"\n== {name} (thr={thr:.3f}) ==")
    print(classification_report(y, pred, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y, proba):.4f}")
    print(f"PR-AUC:  {average_precision_score(y, proba):.4f}")

def main():
    Xtr, ytr = load_dataset(TRAIN)
    Xva, yva = load_dataset(VAL)
    Xte, yte = load_dataset(TEST)

    word = TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                           min_df=3, max_features=200_000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5),
                           min_df=3, max_features=300_000, sublinear_tf=True)
    fe = FeatureUnion([("word", word), ("char", char)])

    clf = LogisticRegression(
        solver="liblinear", class_weight="balanced",
        max_iter=1000, n_jobs=1, random_state=42
    )

    model = Pipeline([("featurizer", fe), ("clf", clf)])
    model.fit(Xtr, ytr)

    proba_val  = model.predict_proba(Xva)[:, 1]
    thr = best_f1_threshold(yva, proba_val)
    evaluate("VAL(okru)", yva, proba_val, thr)

    proba_test = model.predict_proba(Xte)[:, 1]
    evaluate("TEST(pikabu)", yte, proba_test, thr)

    joblib.dump(model, OUT)
    with open(THR_PATH, "w", encoding="utf-8") as f:
        json.dump({"threshold": thr}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved model to {OUT}\nSaved threshold to {THR_PATH}")

if __name__ == "__main__":
    main()
