import re
import csv
from src.data.pathing import DATA

def normalize_ru(text: str) -> str:
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
    src = DATA / "okru" / "okru_raw.txt"
    dst_dir = DATA / "okru"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "okru_normalised.csv"

    toxic_set = {"INSULT", "THREAT", "OBSCENITY"}
    rows = []
    legend_started = False

    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip().startswith("---"):
                legend_started = True
                continue
            if legend_started or not line.strip():
                continue

            m = re.match(r"(?P<labels>(?:__label__\w+(?:,)?)+)\s+(?P<text>.+)", line)
            if not m:
                continue

            labels_raw = [x.replace("__label__", "") for x in m.group("labels").split(",")]
            y = int(any(l in toxic_set for l in labels_raw))

            text = normalize_ru(m.group("text"))
            if not text:
                continue

            rows.append((text, y))

    seen = set()
    unique_rows = []
    for text, y in rows:
        if text not in seen:
            seen.add(text)
            unique_rows.append((text, y))

    with dst.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(unique_rows)

    print(f"✅ saved: {dst}  rows={len(unique_rows)}  pos={sum(y for _, y in unique_rows)}")

if __name__ == "__main__":
    main()
