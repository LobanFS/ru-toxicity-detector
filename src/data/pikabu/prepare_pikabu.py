import pandas as pd
from pathlib import Path

SRC = Path("data/pikabu2ch/pikabu2ch_raw.csv")
DST = Path("data/pikabu2ch/pikabu2ch_normalized.csv")
DST.parent.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(SRC, encoding="utf-8")
    out = pd.DataFrame({
        "text": df["comment"].astype(str).str.strip(),
        "label": (df["toxic"].astype(float) > 0).astype(int)
    })
    out = out[out["text"].str.len() > 0]
    out.to_csv(DST, index=False)
    pos = int(out["label"].sum())
    print(f"✅ saved: {DST} rows={len(out)}  pos={pos}  neg={len(out)-pos}")

if __name__ == "__main__":
    main()