import json
import numpy as np
from huggingface_hub import snapshot_download
from src.data.pathing import BASE
from src.utils import iter_clean

BASELINE_PATH = BASE / "models" / "baseline.joblib"
BASELINE_THR  = BASE / "models" / "baseline.threshold.json"
HF_DIR        = BASE / "models" / "rubert-tiny"
HF_REPO_ID = "LobanFS/ru-toxicity-detector"
HF_LOCAL_DIR = BASE / "models" / "rubert-tiny"

def ensure_hf_model_local():
    if HF_LOCAL_DIR.exists() and any(HF_LOCAL_DIR.iterdir()):
        return HF_LOCAL_DIR
    print(f"[HF] local model not found, downloading {HF_REPO_ID} ...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(HF_LOCAL_DIR),
        local_dir_use_symlinks=False,
        revision="main"
    )
    print(f"[HF] downloaded to {HF_LOCAL_DIR}")
    return HF_LOCAL_DIR

def load_baseline():
    import joblib
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Не найден baseline: {BASELINE_PATH}")
    pipe = joblib.load(BASELINE_PATH)
    thr = 0.5
    if BASELINE_THR.exists():
        with open(BASELINE_THR, "r", encoding="utf-8") as f:
            thr = float(json.load(f).get("threshold", 0.5))
    print(f"[BASELINE] loaded: {BASELINE_PATH.name} | thr={thr:.3f}")

    def predict(texts):
        X = list(iter_clean(texts))
        proba = pipe.predict_proba(X)[:, 1]
        return proba, (proba >= thr).astype(int)
    return predict

def _softmax_T(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / max(T, 1e-8)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]

def load_hf():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_dir = HF_DIR
    if not model_dir.exists():
        raise FileNotFoundError(f"Не найдена HF-модель: {model_dir}")

    # дефолты + inference_config.json
    T, thr, max_len = 1.0, 0.5, 256
    cfg_path = model_dir / "inference_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        T       = float(cfg.get("temperature", T))
        thr     = float(cfg.get("threshold_global", thr))
        max_len = int(cfg.get("max_length", max_len))

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"[HF] loaded: {model_dir.name} | T={T:.3f} thr={thr:.3f} max_len={max_len}")

    def predict(texts):
        batch = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**batch).logits.cpu().numpy()
        proba = _softmax_T(logits, T)
        return proba, (proba >= thr).astype(int)
    return predict

def main():
    ensure_hf_model_local()
    print("Выбери модель:\n  1) baseline  2) rubert-tiny (HF)")
    choice = input("Ваш выбор [1/2]: ").strip()
    if choice == "1":
        predict = load_baseline()
    elif choice == "2":
        predict = load_hf()
    else:
        print("Неизвестный выбор."); return

    print("Введите текст (stop — выход):")
    while True:
        s = input("> ").strip()
        if not s:
            continue
        if s.lower() == "stop":
            print("Выход."); break
        proba, pred = predict([s])
        print(f"proba={proba[0]:.4f} → {'toxic' if pred[0] else 'not toxic'}")

if __name__ == "__main__":
    main()
