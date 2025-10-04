from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from peft import PeftModel

OUT_DIR      = Path("outputs/bert_agnews")
MODEL_DIR    = OUT_DIR / "best_full_model"
ADAPTER_DIR  = OUT_DIR / "lora_adapter"
TEST_CSV     = Path("data/test.csv")
BATCH_SIZE   = 32
SAVE_PREDS   = False
MAX_LENGTH   = 128

def device_str():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_test_dataset(csv_path: Path) -> Dataset:
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("test.csv must contain 'text' and 'label' columns")
    keep_cols = ["text", "label"] + (["label_text"] if "label_text" in df.columns else [])
    df = df[keep_cols].copy()
    df["label"] = df["label"].astype(int)
    return Dataset.from_pandas(df, preserve_index=False)

def infer_label_names(test_df: pd.DataFrame):
    if "label_text" in test_df.columns:
        mapping = (
            test_df.groupby("label")["label_text"]
                  .agg(lambda s: s.mode().iat[0] if not s.mode().empty else str(s.iloc[0]))
                  .to_dict()
        )
        return [mapping[i] if i in mapping else str(i) for i in sorted(test_df["label"].unique())]
    return [str(i) for i in sorted(test_df["label"].unique())]

def tokenize_function(batch, tokenizer, max_length: int):
    return tokenizer(batch["text"], truncation=True, padding=False, max_length=max_length)

def compute_metrics_fn(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def ensure_model(model_dir: Path, adapter_dir: Path, num_labels: int):
    """
    Try to load a full Transformers model first; if not found,
    load base BERT and attach LoRA adapter.
    """
    if (model_dir / "config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        used = str(model_dir)
    elif (adapter_dir / "adapter_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        base = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        try:
            model = model.merge_and_unload()
        except Exception:
            pass
        used = str(adapter_dir)
    else:
        raise FileNotFoundError(
            "No model found. Expected either:\n"
            f" - {model_dir}/config.json  (full model)\n"
            f" - {adapter_dir}/adapter_config.json  (LoRA adapter)"
        )
    return model, tokenizer, used

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device_str()}")

    test_ds = load_test_dataset(TEST_CSV)
    num_labels = len(set(test_ds["label"]))
    print(f"num_labels: {num_labels}")

    test_df = test_ds.to_pandas()
    class_names = infer_label_names(test_df)

    model, tokenizer, used_path = ensure_model(MODEL_DIR, ADAPTER_DIR, num_labels)
    print(f"Loaded model from: {used_path}")

    tok = lambda b: tokenize_function(b, tokenizer, MAX_LENGTH)
    test_tok = test_ds.map(tok, batched=True, remove_columns=[c for c in test_ds.column_names if c != "label"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    eval_args = TrainingArguments(
        output_dir=str(OUT_DIR / "eval_tmp"),
        per_device_eval_batch_size=BATCH_SIZE,
        dataloader_drop_last=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
    )

    pred_output = trainer.predict(test_tok)
    metrics = {k: float(v) for k, v in pred_output.metrics.items()}
    logits = pred_output.predictions
    labels = pred_output.label_ids
    preds = np.argmax(logits, axis=-1)

    print("Test metrics:", metrics)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure()
    disp.plot(include_values=True, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix - Test")
    plt.tight_layout()
    cm_path = OUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")

    if SAVE_PREDS:
        out_pred = test_df.copy()
        out_pred["pred_label"] = preds
        (OUT_DIR / "predictions.csv").write_text(out_pred.to_csv(index=False))
        print(f"Saved predictions to: {OUT_DIR / 'predictions.csv'}")

if __name__ == "__main__":
    main()
