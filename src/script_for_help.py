import pandas as pd
from src.data.pathing import BASE, DATA

def main():
    src = DATA / "okru" / "train.csv"
    data = pd.read_csv(src)
    print(data['label'].value_counts())
if __name__ == "__main__":
    main()