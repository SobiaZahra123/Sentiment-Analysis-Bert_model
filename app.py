import os
import json
import warnings
import random
import logging

# ── Suppress All Warnings ─────────────────────────────────────
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ══════════════════════════════════════════════════════════════
#  FIX: Use parse_known_args so Colab's -f argument is ignored
# ══════════════════════════════════════════════════════════════
import argparse
parser = argparse.ArgumentParser(description="Train BERT on your own CSV dataset")
parser.add_argument("--dataset",    type=str, default=None)
parser.add_argument("--text_col",   type=str, default=None)
parser.add_argument("--label_col",  type=str, default=None)
parser.add_argument("--model_dir",  type=str, default="saved_bert_model")
parser.add_argument("--graphs_dir", type=str, default="graphs")
parser.add_argument("--epochs",     type=int, default=5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_len",    type=int, default=64)
parser.add_argument("--lr",         type=float, default=3e-4)
parser.add_argument("--test_size",  type=float, default=0.2)

args, _ = parser.parse_known_args()

MODEL_SAVE_DIR = args.model_dir
GRAPHS_DIR     = args.graphs_dir
MAX_LEN        = args.max_len
BATCH_SIZE     = args.batch_size
EPOCHS         = args.epochs
LR             = args.lr
TEST_SIZE      = args.test_size
RANDOM_SEED    = 42
MODEL_NAME     = 'bert-base-uncased'

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,     exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print(f"  Device  : {device}")
print(f"  Model   : {MODEL_NAME}")
print(f"  Epochs  : {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
#  STEP 1 — Upload your dataset
# ══════════════════════════════════════════════════════════════
print("\n[1] Loading dataset...")

# ── Detect environment (Colab vs local) ───────────────────────
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

dataset_path = args.dataset

if dataset_path is None:
    if is_colab():
        # In Colab: use file upload widget
        print("\n  Running in Google Colab.")
        print("  A file picker will appear — select your CSV file.\n")
        from google.colab import files
        uploaded = files.upload()
        if not uploaded:
            raise ValueError("No file uploaded. Please upload a CSV file.")
        dataset_path = list(uploaded.keys())[0]
        print(f"\n  File uploaded: {dataset_path}")
    else:
        # Local: ask the user to type the path
        print("\n  No --dataset argument provided.")
        print("  Enter the full path to your CSV file:")
        dataset_path = input("  Path: ").strip().strip('"').strip("'")

if not os.path.isfile(dataset_path):
    raise FileNotFoundError(
        f"\n  File not found: '{dataset_path}'\n"
        f"  Check the path and try again.\n"
    )

df = pd.read_csv(dataset_path)
print(f"\n  File    : {dataset_path}")
print(f"  Shape   : {df.shape}")
print(f"  Columns : {list(df.columns)}")
print(f"\n  Preview (first 3 rows):")
print(df.head(3).to_string())

# ── Auto-detect column names ──────────────────────────────────
def pick_column(df, arg_val, common_names, label):
    if arg_val and arg_val in df.columns:
        print(f"\n  Using '{arg_val}' as the {label} column.")
        return arg_val
    for name in common_names:
        for col in df.columns:
            if col.lower() == name.lower():
                print(f"\n  Auto-detected '{col}' as the {label} column.")
                return col
    print(f"\n  Could not auto-detect the {label} column.")
    print(f"  Available columns: {list(df.columns)}")
    chosen = input(f"  Type the column name for {label}: ").strip()
    if chosen not in df.columns:
        raise ValueError(f"Column '{chosen}' not found.")
    return chosen

text_col  = pick_column(df, args.text_col,
                        ["tweet","text","sentence","content","review","comment","message"],
                        "tweet/text")
label_col = pick_column(df, args.label_col,
                        ["sentiment","label","target","class","category","polarity","emotion"],
                        "sentiment/label")

# ── Clean data ────────────────────────────────────────────────
df = df[[text_col, label_col]].copy()
df.columns = ["tweet", "sentiment"]
before = len(df)
df.dropna(subset=["tweet","sentiment"], inplace=True)
df["tweet"]     = df["tweet"].astype(str).str.strip()
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
df = df[df["tweet"] != ""]
after = len(df)
if before != after:
    print(f"\n  Dropped {before-after} empty/missing rows. Remaining: {after}")

# ── Label distribution ────────────────────────────────────────
label_counts  = df["sentiment"].value_counts()
unique_labels = sorted(df["sentiment"].unique())
label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
id2label  = {i: lbl for lbl, i in label2id.items()}
df["label"] = df["sentiment"].map(label2id)

print(f"\n  Label distribution:")
for lbl, cnt in label_counts.items():
    pct = cnt / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"    {lbl:15s}: {cnt:5d} ({pct:.1f}%) {bar}")
print(f"\n  Encoding : {label2id}")
print(f"  Total rows: {len(df)}")

if len(df) < 10:
    raise ValueError(f"Dataset too small ({len(df)} rows). Need at least 10 rows.")

# ── Class distribution graph ──────────────────────────────────
plt.figure(figsize=(8, 4))
colors = ["#EF4444","#F59E0B","#22C55E","#3B82F6","#8B5CF6"][:len(unique_labels)]
bars = plt.bar(
    [l.capitalize() for l in unique_labels],
    [label_counts.get(l, 0) for l in unique_labels],
    color=colors, edgecolor="white", linewidth=1.5
)
for bar in bars:
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             str(int(bar.get_height())), ha="center", fontweight="bold", fontsize=12)
plt.title("Class Distribution of Uploaded Dataset", fontsize=13, fontweight="bold")
plt.xlabel("Sentiment Class"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/class_distribution.png", dpi=150)
plt.close()
print(f"\n  Class distribution graph saved.")

# ══════════════════════════════════════════════════════════════
#  STEP 2 — Tokenizer
# ══════════════════════════════════════════════════════════════
print(f"\n[2] Loading tokenizer ({MODEL_NAME})...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
print(f"    Vocab size: {tokenizer.vocab_size}")

# ══════════════════════════════════════════════════════════════
#  STEP 3 — Dataset class
# ══════════════════════════════════════════════════════════════
class TwitterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ══════════════════════════════════════════════════════════════
#  STEP 4 — Train / validation split
# ══════════════════════════════════════════════════════════════
can_stratify = all(label_counts.get(l, 0) >= 2 for l in unique_labels)
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_SEED,
    stratify=df["label"] if can_stratify else None
)
print(f"\n[3] Split — Train: {len(train_df)} | Validation: {len(test_df)}")

train_ds     = TwitterDataset(train_df["tweet"], train_df["label"], tokenizer, MAX_LEN)
test_ds      = TwitterDataset(test_df["tweet"],  test_df["label"],  tokenizer, MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ══════════════════════════════════════════════════════════════
#  STEP 5 — Model
# ══════════════════════════════════════════════════════════════
print(f"\n[4] Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(unique_labels),
    ignore_mismatched_sizes=True
)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"    Parameters : {total_params:,}")
print(f"    Num labels : {len(unique_labels)} → {unique_labels}")

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ══════════════════════════════════════════════════════════════
#  STEP 6 — Training loop
# ══════════════════════════════════════════════════════════════
print(f"\n[5] Training for {EPOCHS} epochs...\n")
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    t_loss, t_correct, t_total = 0, 0, 0
    for batch in train_loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        optimizer.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, labels=lbls)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss    += out.loss.item()
        preds      = out.logits.argmax(dim=1)
        t_correct += (preds == lbls).sum().item()
        t_total   += lbls.size(0)

    scheduler.step()
    avg_tl = t_loss / len(train_loader)
    t_acc  = t_correct / t_total

    model.eval()
    v_loss, v_correct, v_total = 0, 0, 0
    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
            v_loss    += out.loss.item()
            preds      = out.logits.argmax(dim=1)
            v_correct += (preds == lbls).sum().item()
            v_total   += lbls.size(0)

    avg_vl = v_loss / len(test_loader)
    v_acc  = v_correct / v_total

    train_losses.append(avg_tl); val_losses.append(avg_vl)
    train_accs.append(t_acc);    val_accs.append(v_acc)

    is_best = v_acc > best_val_acc
    if is_best:
        best_val_acc = v_acc
        model.save_pretrained(MODEL_SAVE_DIR)
        tokenizer.save_pretrained(MODEL_SAVE_DIR)

    print(f"  Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {avg_tl:.4f}  Acc: {t_acc*100:.1f}% | "
          f"Val Loss: {avg_vl:.4f}  Acc: {v_acc*100:.1f}%"
          + ("  ← best" if is_best else ""))

# ══════════════════════════════════════════════════════════════
#  STEP 7 — Training graphs
# ══════════════════════════════════════════════════════════════
ep = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(ep, train_losses, "b-o", label="Train Loss",      linewidth=2, markersize=7)
plt.plot(ep, val_losses,   "r-o", label="Validation Loss", linewidth=2, markersize=7)
plt.fill_between(ep, train_losses, val_losses, alpha=0.07, color="purple")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training vs Validation Loss", fontsize=13, fontweight="bold")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/loss_curve.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(ep, [a*100 for a in train_accs], "b-o", label="Train Accuracy",      linewidth=2, markersize=7)
plt.plot(ep, [a*100 for a in val_accs],   "r-o", label="Validation Accuracy", linewidth=2, markersize=7)
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy", fontsize=13, fontweight="bold")
plt.legend(); plt.grid(alpha=0.3); plt.ylim(0, 105); plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/accuracy_curve.png", dpi=150)
plt.close()
print(f"\n    Graphs saved → {GRAPHS_DIR}/")

# ══════════════════════════════════════════════════════════════
#  STEP 8 — Final evaluation
# ══════════════════════════════════════════════════════════════
print("\n[6] Evaluating best saved model...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_SAVE_DIR, num_labels=len(unique_labels), ignore_mismatched_sizes=True
)
model.to(device); model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        out  = model(input_ids=ids, attention_mask=mask)
        preds = out.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

print(f"\n    Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
print(f"    Precision : {prec:.4f}")
print(f"    Recall    : {rec:.4f}")
print(f"    F1-Score  : {f1:.4f}")
print(f"\n{classification_report(all_labels, all_preds, target_names=[l.capitalize() for l in unique_labels], zero_division=0)}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[l.capitalize() for l in unique_labels],
            yticklabels=[l.capitalize() for l in unique_labels],
            linewidths=0.5, annot_kws={"size": 14})
plt.title("Confusion Matrix", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Label"); plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/confusion_matrix.png", dpi=150)
plt.close()
print(f"    Confusion matrix saved.")

# ══════════════════════════════════════════════════════════════
#  STEP 9 — Save metadata
# ══════════════════════════════════════════════════════════════
meta = {
    "model_name": MODEL_NAME,
    "dataset":    dataset_path,
    "text_col":   text_col,
    "label_col":  label_col,
    "num_labels": len(unique_labels),
    "label2id":   label2id,
    "id2label":   {str(k): v for k, v in id2label.items()},
    "accuracy":   round(acc,  4),
    "precision":  round(prec, 4),
    "recall":     round(rec,  4),
    "f1":         round(f1,   4),
    "epochs":     EPOCHS,
    "batch_size": BATCH_SIZE,
    "max_len":    MAX_LEN,
    "lr":         LR,
    "train_size": len(train_df),
    "test_size":  len(test_df),
    "total_rows": len(df),
}
with open(f"{MODEL_SAVE_DIR}/training_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n    Metadata saved → {MODEL_SAVE_DIR}/training_meta.json")
print("\n" + "=" * 60)
print(f"  TRAINING COMPLETE")
print(f"  Dataset           : {dataset_path}")
print(f"  Best Val Accuracy : {best_val_acc*100:.2f}%")
print(f"  F1-Score          : {f1:.4f}")
print(f"  Model saved to    : {MODEL_SAVE_DIR}/")
print(f"  Graphs saved to   : {GRAPHS_DIR}/")
print("=" * 60)
