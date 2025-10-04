import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError as e:
    raise SystemExit(
        "peft not installed. Run: pip install -U peft"
    )

MODEL_NAME = "bert-base-uncased"
OUT_DIR = Path("outputs/bert_agnews")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def device_info():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_split(csv_path: str):
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{csv_path} must have 'text' and 'label' columns")
    df["label"] = df["label"].astype(int)
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

def tokenize_function(examples, tokenizer, max_length=128):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length,
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def plot_losses(trainer, out_path):
    logs = trainer.state.log_history
    steps, train_loss, eval_steps, eval_loss = [], [], [], []
    seen_train = set()
    for entry in logs:
        if "loss" in entry and "epoch" in entry and "step" in entry:
            if entry["step"] not in seen_train:
                steps.append(entry["step"])
                train_loss.append(entry["loss"])
                seen_train.add(entry["step"])
        if "eval_loss" in entry and "epoch" in entry:
            eval_steps.append(entry.get("step", len(eval_steps)))
            eval_loss.append(entry["eval_loss"])

    plt.figure()
    if steps:
        plt.plot(steps, train_loss, label="train loss")
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label="eval loss")
    plt.title("Training & Evaluation Loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    print(f"Detected device: {device_info()}")
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    train_ds = load_split("data/train.csv")
    val_ds   = load_split("data/val.csv")
    test_ds  = load_split("data/test.csv")

    num_labels = len(set(train_ds["label"]))
    print(f"num_labels = {num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    tokenize = lambda batch: tokenize_function(batch, tokenizer)
    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_tok   = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_tok  = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.10,
        target_modules=["query", "key", "value", "dense"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_metrics = trainer.evaluate(test_tok, metric_key_prefix="test")
    print("Test metrics:", test_metrics)

    trainer.save_model(str(OUT_DIR / "best_full_model"))
    tokenizer.save_pretrained(str(OUT_DIR / "best_full_model"))

    model.save_pretrained(str(OUT_DIR / "lora_adapter"))
    tokenizer.save_pretrained(str(OUT_DIR / "lora_adapter"))

    plot_losses(trainer, str(OUT_DIR / "loss_plot.png"))
    print(f"Saved loss plot to {OUT_DIR / 'loss_plot.png'}")

if __name__ == "__main__":
    main()
