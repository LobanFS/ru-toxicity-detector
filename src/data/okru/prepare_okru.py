import re
import csv
from src.data.pathing import DATA

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
            text = m.group("text").strip()
            rows.append((text, y))

    with dst.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)

    print(f"✅ saved: {dst}  rows={len(rows)}")

if __name__ == "__main__":
    main()