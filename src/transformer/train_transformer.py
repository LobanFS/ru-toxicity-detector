import argparse, numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from transformers.trainer_utils import set_seed
from src.data.pathing import DATA, BASE

set_seed(42)

TRAIN = DATA / "okru" / "train.csv"
VAL = DATA / "okru" / "val.csv"
TEST = DATA / "pikabu2ch" / "pikabu2ch_normalised.csv"
OUT = BASE / "models" / "rubert-tiny2-tox"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="cointegrated/rubert-tiny2")
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--val",   default=VAL)
    ap.add_argument("--test",  default=TEST)
    ap.add_argument("--out",   default=OUT)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch",  type=int, default=32)
    ap.add_argument("--max_len",type=int, default=128)
    args = ap.parse_args()

    ds = load_dataset("csv", data_files={"train": args.train, "val": args.val, "test": args.test})

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    def tokenize(b): return tok(b["text"], truncation=True, padding="max_length", max_length=args.max_len)
    ds = ds.map(tokenize, batched=True).rename_column("label","labels")
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        p = (logits - logits.max(axis=1, keepdims=True))
        p = np.exp(p); p = p / p.sum(axis=1, keepdims=True)
        yhat = p.argmax(axis=1)
        return {
            "f1": f1_score(y, yhat),
            "roc_auc": roc_auc_score(y, p[:,1]),
            "pr_auc":  average_precision_score(y, p[:,1])
        }

    args_tr = TrainingArguments(
        output_dir=args.out,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        logging_dir="logs",
        report_to="none",
        logging_strategy="steps"
    )

    trainer = Trainer(
        model=model, args=args_tr,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        tokenizer=tok, compute_metrics=compute_metrics
    )

    trainer.train()
    print("test on pikabu")
    print(trainer.evaluate(ds["test"]))
    trainer.save_model(args.out); tok.save_pretrained(args.out)

if __name__ == "__main__":
    main()
