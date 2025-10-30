from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SRC = Path("data/okru/okru_normalized.csv")
OUT = Path("data/okru")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(SRC)

    train_df, tmp_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, random_state=42, stratify=tmp_df["label"]
    )

    train_df.to_csv(OUT / "train.csv", index=False)
    val_df.to_csv(OUT / "val.csv", index=False)
    test_df.to_csv(OUT / "test.csv", index=False)

    print("✅ saved:",
          OUT / "train.csv", train_df.shape,
          OUT / "val.csv",   val_df.shape,
          OUT / "test.csv",  test_df.shape, sep="\n")

if __name__ == "__main__":
    main()
