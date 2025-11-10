import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data.pathing import DATA
from src.utils import iter_clean

OKRU = DATA / "okru" / "okru_normalised.csv"
PIKABU = DATA / "pikabu2ch" / "pikabu2ch_normalised.csv"
OUT_DIR = DATA / "merged"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RND = 42

def main():
    # читаем только нужные колонки и приводим типы (чуть быстрее и стабильнее)
    usecols = ["text", "label"]
    dtype = {"label": "int8"}

    okru = pd.read_csv(OKRU, usecols=usecols, dtype=dtype).assign(source="okru")
    pikabu = pd.read_csv(PIKABU, usecols=usecols, dtype=dtype).assign(source="pikabu")

    merged = pd.concat([okru, pikabu], ignore_index=True)

    # КЛЮЧЕВОЕ: чистим так же, как в обучении, и убираем точные дубликаты
    merged["text_clean"] = list(iter_clean(merged["text"].astype(str)))
    merged = merged.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)

    # если большой файл, можно не писать общий merged.csv, чтобы не тратить время
    # merged.drop(columns=["text_clean"]).to_csv(OUT_DIR / "merged.csv", index=False)

    # 80/10/10 со стратификацией по метке
    train_df, tmp_df = train_test_split(
        merged, test_size=0.20, random_state=RND, stratify=merged["label"]
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, random_state=RND, stratify=tmp_df["label"]
    )

    # сохраняем без служебной колонки
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.drop(columns=["text_clean"]).to_csv(OUT_DIR / f"{name}.csv", index=False)

    # sanity-check размеров и баланса
    total = len(train_df) + len(val_df) + len(test_df)
    print("sizes:", len(train_df), len(val_df), len(test_df))
    print("fractions:", round(len(train_df)/total,3),
                      round(len(val_df)/total,3),
                      round(len(test_df)/total,3))
    print("train balance:", train_df["label"].value_counts(normalize=True).round(3).to_dict())

if __name__ == "__main__":
    main()
