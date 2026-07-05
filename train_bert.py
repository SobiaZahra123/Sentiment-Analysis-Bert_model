import os
import json
import warnings
import random
import logging
import argparse
from datetime import datetime

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
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ══════════════════════════════════════════════════════════════
#  PARSE ARGUMENTS
# ══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Train BERT on your own CSV dataset")
parser.add_argument("--dataset",    type=str, default=None, help="Path to CSV dataset")
parser.add_argument("--text_col",   type=str, default=None, help="Column name for text")
parser.add_argument("--label_col",  type=str, default=None, help="Column name for labels")
parser.add_argument("--model_dir",  type=str, default="saved_bert_model", help="Model save directory")
parser.add_argument("--graphs_dir", type=str, default="graphs", help="Graphs save directory")
parser.add_argument("--epochs",     type=int, default=50, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size (reduced for memory)")
parser.add_argument("--max_len",    type=int, default=128, help="Max sequence length")
parser.add_argument("--lr",         type=float, default=2e-5, help="Learning rate")
parser.add_argument("--test_size",  type=float, default=0.2, help="Test split ratio")
parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")
args, _ = parser.parse_known_args()

# ── Configuration ─────────────────────────────────────────────
MODEL_SAVE_DIR = args.model_dir
GRAPHS_DIR     = args.graphs_dir
MAX_LEN        = args.max_len
BATCH_SIZE     = args.batch_size
EPOCHS         = args.epochs
LR             = args.lr
TEST_SIZE      = args.test_size
WARMUP_STEPS   = args.warmup_steps
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
EARLY_STOPPING_PATIENCE = args.early_stopping_patience
NUM_WORKERS    = args.num_workers
RANDOM_SEED    = 42
MODEL_NAME     = 'bert-base-uncased'

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,     exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 70)
print("  BERT SENTIMENT ANALYSIS TRAINING")
print("=" * 70)
print(f"  Device                     : {device}")
print(f"  Model                      : {MODEL_NAME}")
print(f"  Epochs                     : {EPOCHS}")
print(f"  Batch Size                 : {BATCH_SIZE}")
print(f"  Gradient Accumulation      : {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective Batch Size       : {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  Learning Rate              : {LR}")
print(f"  Max Sequence Length        : {MAX_LEN}")
print(f"  Warmup Steps               : {WARMUP_STEPS}")
print(f"  Early Stopping Patience    : {EARLY_STOPPING_PATIENCE}")
print("=" * 70)

# ══════════════════════════════════════════════════════════════
#  DATASET PATH DETECTION
# ══════════════════════════════════════════════════════════════
def detect_platform():
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    try:
        import google.colab
        return "colab"
    except ImportError:
        pass
    return "local"

PLATFORM = detect_platform()
print(f"\n  Platform : {PLATFORM}")

# ── Dataset path ──────────────────────────────────────────────
DATASET_PATH = "/kaggle/input/datasets/sobiamatthal/sentiment-analysis-tweets/twitter_training.csv"

print("\n[1] Loading dataset...")

dataset_path = args.dataset

if dataset_path is None:
    if PLATFORM == "kaggle":
        dataset_path = DATASET_PATH
        if not os.path.isfile(dataset_path):
            print(f"\n  ⚠ Hardcoded path not found: {dataset_path}")
            print("     Scanning /kaggle/input/ for any CSV file...")
            found = []
            for root, dirs, files in os.walk("/kaggle/input"):
                for f in files:
                    if f.lower().endswith(".csv"):
                        found.append(os.path.join(root, f))
            if found:
                dataset_path = found[0]
                print(f"     Auto-selected: {dataset_path}")
            else:
                raise FileNotFoundError("No CSV file found under /kaggle/input/")
    
    elif PLATFORM == "colab":
        from google.colab import files
        uploaded = files.upload()
        if not uploaded:
            raise ValueError("No file uploaded.")
        dataset_path = list(uploaded.keys())[0]
    
    else:
        print("\n  Enter the full path to your CSV file:")
        dataset_path = input("  Path: ").strip().strip('"').strip("'")

if not os.path.isfile(dataset_path):
    raise FileNotFoundError(f"File not found: '{dataset_path}'")

# ── Load CSV ──────────────────────────────────────────────────
df = pd.read_csv(dataset_path, encoding='utf-8', on_bad_lines='skip')
print(f"\n  File    : {dataset_path}")
print(f"  Shape   : {df.shape}")
print(f"  Columns : {list(df.columns)}")

# ── Clean and rename columns ──────────────────────────────────
df.columns = ['id', 'platform', 'sentiment', 'tweet']
print(f"\n  Renamed columns: {list(df.columns)}")

# ── Clean data ──────────────────────────────────────────────
before = len(df)
df.dropna(subset=['tweet', 'sentiment'], inplace=True)
df['tweet'] = df['tweet'].astype(str).str.strip()
df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
df = df[df['tweet'] != '']
after = len(df)

if before != after:
    print(f"\n  Dropped {before - after} empty rows. Remaining: {after}")

# ── Label encoding ──────────────────────────────────────────
label_counts = df['sentiment'].value_counts()
unique_labels = sorted(df['sentiment'].unique())

print(f"\n  Unique labels: {unique_labels}")

label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
id2label = {i: lbl for lbl, i in label2id.items()}
df['label'] = df['sentiment'].map(label2id)

print(f"\n  Label distribution:")
for lbl, cnt in label_counts.items():
    pct = cnt / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"    {lbl:20s}: {cnt:6d}  ({pct:.1f}%)  {bar}")
print(f"\n  Encoding: {label2id}")
print(f"  Total rows: {len(df)}")

# ── Check dataset validity ──────────────────────────────────
if len(df) < 10:
    raise ValueError(f"Dataset too small ({len(df)} rows). Need at least 10 rows.")

if len(unique_labels) < 2:
    raise ValueError(f"Dataset has only one label: {unique_labels}. Need at least 2 classes.")

# ── Class distribution graph ────────────────────────────────
plt.figure(figsize=(8, 4))
colors = ['#EF4444','#F59E0B','#22C55E','#3B82F6','#8B5CF6'][:len(unique_labels)]
bars = plt.bar(
    [l.capitalize() for l in unique_labels],
    [label_counts.get(l, 0) for l in unique_labels],
    color=colors, edgecolor='white', linewidth=1.5
)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             str(int(bar.get_height())), ha='center', fontweight='bold', fontsize=12)
plt.title('Class Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Sentiment Class'); plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/class_distribution.png", dpi=150)
plt.close()
print(f"\n  Graph saved → {GRAPHS_DIR}/class_distribution.png")

# ══════════════════════════════════════════════════════════════
#  TOKENIZER
# ══════════════════════════════════════════════════════════════
print(f"\n[2] Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
print(f"    Vocab size: {tokenizer.vocab_size}")

# ══════════════════════════════════════════════════════════════
#  DATASET CLASS
# ══════════════════════════════════════════════════════════════
class TwitterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            enc = self.tokenizer(
                self.texts[idx],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            return {
                'input_ids': torch.zeros(self.max_len, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_len, dtype=torch.long),
                'labels': torch.tensor(0, dtype=torch.long)
            }

# ══════════════════════════════════════════════════════════════
#  TRAIN/VALIDATION SPLIT
# ══════════════════════════════════════════════════════════════
can_stratify = all(label_counts.get(l, 0) >= 2 for l in unique_labels)
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_SEED,
    stratify=df['label'] if can_stratify else None
)
print(f"\n[3] Split — Train: {len(train_df)} | Validation: {len(test_df)}")

train_ds = TwitterDataset(train_df['tweet'], train_df['label'], tokenizer, MAX_LEN)
test_ds = TwitterDataset(test_df['tweet'], test_df['label'], tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

print(f"    Train batches: {len(train_loader)}")
print(f"    Test batches: {len(test_loader)}")

# ══════════════════════════════════════════════════════════════
#  BUILD MODEL
# ══════════════════════════════════════════════════════════════
print(f"\n[4] Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(unique_labels),
    ignore_mismatched_sizes=True
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    Total Parameters    : {total_params:,}")
print(f"    Trainable Parameters: {trainable_params:,}")

# ── Optimizer ─────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01, eps=1e-8)

# ── Scheduler ─────────────────────────────────────────────────
total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps
)

# ── Mixed Precision ────────────────────────────────────────────
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

# ══════════════════════════════════════════════════════════════
#  TRAINING LOOP - 50 EPOCHS (FORCED EXECUTION)
# ══════════════════════════════════════════════════════════════
print(f"\n[5] Starting Training for {EPOCHS} epochs...")
print("-" * 70)
print("  Training will begin now...")
print("-" * 70)

train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0
patience_counter = 0
epoch_times = []

# ── FORCE TORCH TO EXECUTE ──────────────────────────────────
torch.cuda.empty_cache() if device.type == 'cuda' else None

try:
    for epoch in range(EPOCHS):
        print(f"\n  ⏳ Epoch {epoch+1}/{EPOCHS} starting...")
        epoch_start = datetime.now()
        
        # ── Train ─────────────────────────────────────────────────
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                lbls = batch['labels'].to(device)
                
                # Forward pass with mixed precision
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=ids, attention_mask=mask, labels=lbls)
                        loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                    scaler.scale(loss).backward()
                else:
                    outputs = model(input_ids=ids, attention_mask=mask, labels=lbls)
                    loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                    loss.backward()
                
                total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                
                # Gradient accumulation
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or batch_idx == len(train_loader) - 1:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Track accuracy
                preds = outputs.logits.argmax(dim=1)
                train_correct += (preds == lbls).sum().item()
                train_total += lbls.size(0)
                
                # Print progress every 50 batches
                if (batch_idx + 1) % 50 == 0:
                    print(f"    Batch {batch_idx+1}/{len(train_loader)} | "
                          f"Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")
                
            except Exception as e:
                print(f"    ⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        if train_total == 0:
            print("  ⚠️ No valid training samples. Skipping epoch.")
            continue
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # ── Validation ────────────────────────────────────────────
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                try:
                    ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    lbls = batch['labels'].to(device)
                    
                    outputs = model(input_ids=ids, attention_mask=mask, labels=lbls)
                    total_val_loss += outputs.loss.item()
                    
                    preds = outputs.logits.argmax(dim=1)
                    val_correct += (preds == lbls).sum().item()
                    val_total += lbls.size(0)
                except Exception as e:
                    continue
        
        if val_total == 0:
            print(f"  ⚠️ No valid validation samples in epoch {epoch+1}")
            continue
        
        avg_val_loss = total_val_loss / len(test_loader)
        val_accuracy = val_correct / val_total
        
        # ── Store metrics ─────────────────────────────────────────
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        
        # ── Save best model ──────────────────────────────────────
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            model.save_pretrained(MODEL_SAVE_DIR)
            tokenizer.save_pretrained(MODEL_SAVE_DIR)
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1
            is_best = False
        
        # ── Early Stopping ───────────────────────────────────────
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  🛑 Early stopping triggered at epoch {epoch+1}")
            break
        
        # ── Logging ──────────────────────────────────────────────
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        epoch_times.append(epoch_time)
        
        lr = scheduler.get_last_lr()[0]
        print(f"  ✅ Epoch {epoch+1:2d}/{EPOCHS} Complete | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy*100:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_accuracy*100:.2f}% | "
              f"LR: {lr:.2e} | "
              f"Time: {epoch_time:.1f}s"
              + ("  ★ BEST" if is_best else ""))

except KeyboardInterrupt:
    print("\n\n  ⚠️ Training interrupted by user.")
except Exception as e:
    print(f"\n\n  ❌ Training error: {e}")
    import traceback
    traceback.print_exc()

# ── Final Summary ────────────────────────────────────────────
if len(train_losses) == 0:
    print("\n  No training completed. Please check your dataset.")
else:
    print("-" * 70)
    print(f"\n  ✅ Training Complete!")
    print(f"  Total Training Time: {sum(epoch_times)/60:.2f} minutes")
    print(f"  Best Validation Accuracy: {best_val_acc*100:.2f}%")

    # ── Training Graphs ──────────────────────────────────────────
    print("\n[6] Generating training graphs...")
    ep = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(ep, train_losses, "b-o", label="Train Loss", linewidth=2, markersize=6)
    axes[0].plot(ep, val_losses, "r-o", label="Validation Loss", linewidth=2, markersize=6)
    axes[0].fill_between(ep, train_losses, val_losses, alpha=0.07, color="purple")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(ep, [a*100 for a in train_accs], "b-o", label="Train Accuracy", linewidth=2, markersize=6)
    axes[1].plot(ep, [a*100 for a in val_accs], "r-o", label="Validation Accuracy", linewidth=2, markersize=6)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training vs Validation Accuracy", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/training_curves.png", dpi=150)
    plt.close()
    print(f"    Graphs saved → {GRAPHS_DIR}/training_curves.png")

    # ── Final Evaluation ──────────────────────────────────────────
    print("\n[7] Evaluating best model on validation set...")

    model = BertForSequenceClassification.from_pretrained(
        MODEL_SAVE_DIR, num_labels=len(unique_labels), ignore_mismatched_sizes=True
    )
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            try:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                lbls = batch['labels'].to(device)
                outputs = model(input_ids=ids, attention_mask=mask)
                preds = outputs.logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())
            except:
                continue

    if len(all_labels) > 0:
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        print(f"\n  Final Metrics:")
        print(f"    Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
        print(f"    Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}")
        print(f"    F1-Score  : {f1:.4f}")

        print(f"\n{classification_report(all_labels, all_preds, target_names=[l.capitalize() for l in unique_labels], zero_division=0)}")

        # ── Confusion Matrix ────────────────────────────────────
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[l.capitalize() for l in unique_labels],
                    yticklabels=[l.capitalize() for l in unique_labels],
                    linewidths=0.5, annot_kws={"size": 14})
        plt.title("Confusion Matrix", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(f"{GRAPHS_DIR}/confusion_matrix.png", dpi=150)
        plt.close()
        print(f"    Confusion matrix saved → {GRAPHS_DIR}/confusion_matrix.png")

    # ── Save Metadata ────────────────────────────────────────────
    meta = {
        "model_name": MODEL_NAME,
        "dataset": dataset_path,
        "num_labels": len(unique_labels),
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "epochs_trained": len(train_losses),
        "total_epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "max_len": MAX_LEN,
        "lr": LR,
        "warmup_steps": WARMUP_STEPS,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "total_rows": len(df),
        "best_val_accuracy": round(best_val_acc, 4),
        "training_time_minutes": round(sum(epoch_times)/60, 2),
    }
    with open(f"{MODEL_SAVE_DIR}/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Metadata saved → {MODEL_SAVE_DIR}/training_meta.json")

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"  Platform            : {PLATFORM}")
    print(f"  Dataset             : {dataset_path}")
    print(f"  Total Rows          : {len(df)}")
    print(f"  Epochs Trained      : {len(train_losses)}/{EPOCHS}")
    print(f"  Best Val Accuracy   : {best_val_acc*100:.2f}%")
    if len(all_labels) > 0:
        print(f"  Final Test Accuracy : {acc*100:.2f}%")
        print(f"  F1-Score            : {f1:.4f}")
    print(f"  Training Time       : {sum(epoch_times)/60:.2f} minutes")
    print(f"  Model Saved To      : {MODEL_SAVE_DIR}/")
    print(f"  Graphs Saved To     : {GRAPHS_DIR}/")
    print("=" * 70)
