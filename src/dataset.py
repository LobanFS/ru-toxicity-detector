import pandas as pd
import joblib
import json
from src.data.pathing import BASE
from src.utils import iter_clean
from src.data.pathing import DATA

def main():
    data = pd.read_csv(DATA / "okru" / "train.csv")
    print(data["label"].value_counts())
if __name__ == "__main__":
    main()
