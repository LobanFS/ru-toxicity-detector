import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "rubert-tiny"

def softmax_with_temperature(logits, T):
    logits = logits / T
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, 1]

def main():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    with open(MODEL_DIR / "inference_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    T = float(cfg.get("temperature", 1.0))
    thr = float(cfg.get("threshold_global", 0.5))
    max_len = int(cfg.get("max_length", 256))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    print(f"Model loaded (T={T:.3f}, thr={thr:.3f})")
    print("Введите текст (stop — выход):")

    while True:
        s = input("> ").strip()
        if s.lower() == "stop":
            print("Выход.")
            break
        if not s:
            continue

        batch = tok([s], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**batch).logits.cpu().numpy()

        p = softmax_with_temperature(logits, T)[0]
        label = int(p >= thr)
        print(f"proba={p:.4f} → {'toxic' if label else 'not toxic'}")

if __name__ == "__main__":
    main()
