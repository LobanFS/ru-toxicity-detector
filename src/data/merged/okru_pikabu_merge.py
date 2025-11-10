import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.pathing import DATA
OKRU = DATA / "okru" / "okru_normalised.csv"
PIKABU = DATA / "pikabu2ch" / "pikabu2ch_normalised.csv"
OUT_DIR = DATA / "merged"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    okru = pd.read_csv(OKRU)
    pikabu = pd.read_csv(PIKABU)

    okru = okru.assign(source="okru")
    pikabu = pikabu.assign(source="pikabu")

    merged = pd.concat([okru, pikabu], ignore_index=True)
    merged.to_csv(OUT_DIR / "merged.csv", index=False)

    train_df, tmp_df = train_test_split(
        merged, test_size=0.20, random_state=42, stratify=merged["label"]
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, random_state=42, stratify=tmp_df["label"]
    )

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)

    print(train_df['label'].value_counts())

if __name__ == "__main__":
    main()
