import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.pathing import DATA

SRC = DATA / "pikabu2ch" / "pikabu2ch_normalised.csv"
OUT = DATA / "pikabu2ch"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(SRC)
    val_df, test_df = train_test_split(
        df, test_size=0.50, random_state=42, stratify=df["label"]
    )
    val_df.to_csv(OUT / "val.csv", index=False)
    test_df.to_csv(OUT / "test.csv", index=False)
    print("✅ saved:",
          OUT / "val.csv",   val_df.shape,
          OUT / "test.csv",  test_df.shape, sep="\n")

if __name__ == "__main__":
    main()