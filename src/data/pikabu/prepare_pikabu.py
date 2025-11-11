import re
import pandas as pd
from src.data.pathing import DATA

SRC = DATA / "pikabu2ch" / "pikabu2ch_raw.csv"
DST = DATA / "pikabu2ch" / "pikabu2ch_normalised.csv"
DST.parent.mkdir(parents=True, exist_ok=True)

def normalise_ru(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    t = text.strip()
    t = t.replace("Ё", "Е").replace("ё", "е")
    t = t.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+|u/\w+", " ", t)
    t = re.sub(r"&[a-z]+;", " ", t)
    t = re.sub(r"\+?\d[\d\-\s]{7,}\d", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def main():
    df = pd.read_csv(SRC, encoding="utf-8")
    need_cols = {"comment", "toxic"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"В {SRC} нет колонок: {missing}. Ожидаю {need_cols}")
    out = pd.DataFrame({
        "text": df["comment"].astype(str).apply(normalise_ru),
        "label": (pd.to_numeric(df["toxic"], errors="coerce").fillna(0) > 0).astype(int)
    })
    out = out.dropna(subset=["text"])
    out = out[out["text"].str.len() > 0]
    out = out.drop_duplicates()
    out.to_csv(DST, index=False)
    pos = int(out["label"].sum())
    neg = int(len(out) - pos)
    print(f"saved: {DST}  rows={len(out)}  pos={pos}  neg={neg}")

if __name__ == "__main__":
    main()
