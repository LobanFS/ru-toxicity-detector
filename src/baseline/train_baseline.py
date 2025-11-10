import pandas as pd
import joblib
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from src.utils import iter_clean
from src.data.pathing import DATA, BASE

# Бейзлайн - TD-IDF + логичстическая регрессия.

TRAIN = DATA / "okru" / "train.csv"
VAL = DATA / "pikabu2ch/val.csv"
TEST = DATA / "pikabu2ch" / "test.csv"
OUT = BASE / "models" / "baseline.joblib"
OUT.parent.mkdir(parents=True, exist_ok=True)

def load_dataset(path):
    df = pd.read_csv(path)
    texts = list(iter_clean(df["text"].tolist()))
    labels = df["label"].astype(int).values
    return texts, labels

def main():
    X_train, y_train = load_dataset(TRAIN)
    X_val, y_val = load_dataset(VAL)
    X_test, y_test = load_dataset(TEST)

    word_vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1,2),
        min_df=3, max_features=200_000, sublinear_tf=True
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3,5),
        min_df=3, max_features=300_000, sublinear_tf=True
    )
    featurizer = FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])

    clf = LogisticRegression(
        solver="liblinear", class_weight="balanced",
        max_iter=1000, n_jobs=1
    )

    model = Pipeline([("featurizer", featurizer), ("clf", clf)])
    model.fit(X_train, y_train)

    def evaluate(name, X, y):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        print(f"\n== {name} ==")
        print(classification_report(y, pred, digits=4))
        print(f"ROC-AUC: {roc_auc_score(y, proba):.4f}")
        print(f"PR-AUC:  {average_precision_score(y, proba):.4f}")

    evaluate("VAL", X_val, y_val)
    evaluate("TEST", X_test, y_test)

    joblib.dump(model, OUT)

if __name__ == "__main__":
    main()
