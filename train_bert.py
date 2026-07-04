import os, json, warnings, random
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertConfig, BertForSequenceClassification, BertTokenizer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── Config ────────────────────────────────────────────────────
DATASET_PATH   = 'dataset/twitter_sentiment.csv'
MODEL_SAVE_DIR = 'saved_bert_model'
GRAPHS_DIR     = 'graphs'
MAX_LEN        = 64
BATCH_SIZE     = 32
EPOCHS         = 5
LR             = 3e-4
TEST_SIZE      = 0.2
RANDOM_SEED    = 42
MODEL_NAME     = 'bert-base-uncased-local (4-layer, 256-hidden)'

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

# ── 1. Dataset ────────────────────────────────────────────────
print("\n[1] Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"    Rows: {len(df)} | Columns: {list(df.columns)}")

label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
id2label  = {0: 'negative', 1: 'neutral', 2: 'positive'}
df['label'] = df['sentiment'].map(label2id)
counts = df['sentiment'].value_counts()
print(f"    Labels: {counts.to_dict()}")

# class distribution graph
plt.figure(figsize=(7, 4))
colors = ['#EF4444', '#F59E0B', '#22C55E']
bars = plt.bar(['Negative','Neutral','Positive'],
               [counts.get('negative',0), counts.get('neutral',0), counts.get('positive',0)],
               color=colors, edgecolor='white', linewidth=1.5)
for bar in bars:
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
             str(int(bar.get_height())), ha='center', fontweight='bold', fontsize=12)
plt.title('Class Distribution — Twitter Sentiment Dataset', fontsize=13, fontweight='bold')
plt.xlabel('Sentiment'); plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'{GRAPHS_DIR}/class_distribution.png', dpi=150)
plt.close()
print("    Class distribution graph saved.")

# ── 2. Tokenizer (character-level with simple word splitting) ─
print("\n[2] Setting up tokenizer...")
tokenizer = BertTokenizer(vocab_file=f'{MODEL_SAVE_DIR}/vocab.txt',
                          do_lower_case=True)
print(f"    Vocab size: {tokenizer.vocab_size}")

# ── 3. Dataset class ──────────────────────────────────────────
class TwitterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = list(texts), list(labels)
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── 4. Split ──────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df['label']
)
print(f"\n[3] Split — Train: {len(train_df)} | Test: {len(test_df)}")

train_ds = TwitterDataset(train_df['tweet'], train_df['label'], tokenizer, MAX_LEN)
test_ds  = TwitterDataset(test_df['tweet'],  test_df['label'],  tokenizer, MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 5. Model ──────────────────────────────────────────────────
print("\n[4] Loading model from saved_bert_model/...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_SAVE_DIR,
    num_labels=3,
    ignore_mismatched_sizes=True
)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"    Parameters: {total_params:,}")

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── 6. Training ───────────────────────────────────────────────
print(f"\n[5] Training for {EPOCHS} epochs...\n")
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

best_val_acc = 0
for epoch in range(EPOCHS):
    # Train
    model.train()
    t_loss, t_correct, t_total = 0, 0, 0
    for batch in train_loader:
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbls = batch['labels'].to(device)
        optimizer.zero_grad()
        out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
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

    # Validate
    model.eval()
    v_loss, v_correct, v_total = 0, 0, 0
    with torch.no_grad():
        for batch in test_loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)
            out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
            v_loss    += out.loss.item()
            preds      = out.logits.argmax(dim=1)
            v_correct += (preds == lbls).sum().item()
            v_total   += lbls.size(0)

    avg_vl = v_loss / len(test_loader)
    v_acc  = v_correct / v_total

    train_losses.append(avg_tl); val_losses.append(avg_vl)
    train_accs.append(t_acc);    val_accs.append(v_acc)

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        model.save_pretrained(MODEL_SAVE_DIR)

    print(f"  Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss:{avg_tl:.4f} Acc:{t_acc*100:.1f}% | "
          f"Val Loss:{avg_vl:.4f} Acc:{v_acc*100:.1f}%"
          + (" ← best" if v_acc == best_val_acc else ""))

# ── 7. Graphs ─────────────────────────────────────────────────
ep = range(1, EPOCHS+1)

plt.figure(figsize=(8,5))
plt.plot(ep, train_losses, 'b-o', label='Train Loss',      linewidth=2, markersize=7)
plt.plot(ep, val_losses,   'r-o', label='Validation Loss', linewidth=2, markersize=7)
plt.fill_between(ep, train_losses, val_losses, alpha=0.08, color='purple')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Training vs Validation Loss', fontsize=13, fontweight='bold')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f'{GRAPHS_DIR}/loss_curve.png', dpi=150)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(ep, [a*100 for a in train_accs], 'b-o', label='Train Accuracy',      linewidth=2, markersize=7)
plt.plot(ep, [a*100 for a in val_accs],   'r-o', label='Validation Accuracy', linewidth=2, markersize=7)
plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
plt.title('Training vs Validation Accuracy', fontsize=13, fontweight='bold')
plt.legend(); plt.grid(alpha=0.3); plt.ylim(0,105); plt.tight_layout()
plt.savefig(f'{GRAPHS_DIR}/accuracy_curve.png', dpi=150)
plt.close()
print("\n    Loss & Accuracy graphs saved.")

# ── 8. Evaluate best model ────────────────────────────────────
print("\n[6] Evaluating best model on test set...")
model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_DIR, num_labels=3)
model.to(device); model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbls = batch['labels'].to(device)
        out  = model(input_ids=ids, attention_mask=mask)
        preds = out.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
rec  = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1   = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"\n    Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
print(f"    Precision : {prec:.4f}")
print(f"    Recall    : {rec:.4f}")
print(f"    F1-Score  : {f1:.4f}")
print(f"\n{classification_report(all_labels, all_preds, target_names=['Negative','Neutral','Positive'], zero_division=0)}")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative','Neutral','Positive'],
            yticklabels=['Negative','Neutral','Positive'],
            linewidths=0.5, annot_kws={'size':14})
plt.title('Confusion Matrix', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Label'); plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(f'{GRAPHS_DIR}/confusion_matrix.png', dpi=150)
plt.close()
print("    Confusion matrix saved.")

# ── 9. Save tokenizer + metadata ─────────────────────────────
tokenizer.save_pretrained(MODEL_SAVE_DIR)

meta = {
    'model_name': MODEL_NAME, 'num_labels': 3,
    'max_len': MAX_LEN, 'label2id': label2id,
    'id2label': {str(k): v for k,v in id2label.items()},
    'accuracy': round(acc,4), 'precision': round(prec,4),
    'recall': round(rec,4), 'f1': round(f1,4),
    'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
    'train_size': len(train_df), 'test_size': len(test_df)
}
with open(f'{MODEL_SAVE_DIR}/training_meta.json','w') as f:
    json.dump(meta, f, indent=2)

print(f"\n    Model + tokenizer + metadata saved to {MODEL_SAVE_DIR}/")
print("\n" + "=" * 60)
print(f"  TRAINING COMPLETE")
print(f"  Best Val Accuracy : {best_val_acc*100:.2f}%")
print(f"  F1-Score          : {f1:.4f}")
print("=" * 60)
