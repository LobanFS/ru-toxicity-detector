import joblib
import json
from src.data.pathing import BASE
from src.utils import iter_clean
MODEL_PATH = BASE / "models" / "baseline.joblib"
THR_PATH = BASE / "models" / "baseline.threshold.json"

def main():
    model = joblib.load(MODEL_PATH)
    try:
        with open(THR_PATH, "r", encoding="utf-8") as f:
            thr = float(json.load(f).get("threshold", 0.5))
    except FileNotFoundError:
        thr = 0.5
    while True:
        s = input("> ").strip()
        if s.lower() == "stop":
            print("Выход.")
            break
        if not s:
            continue

        clean = next(iter_clean([s]))
        proba = model.predict_proba([clean])[0, 1]
        label = int(proba >= thr)
        print(f"proba={proba:.4f} → {'toxic' if label else 'not toxic'}")

if __name__ == "__main__":
    main()
