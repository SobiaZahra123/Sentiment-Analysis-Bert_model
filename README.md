# BERT Twitter Sentiment Analysis — PyQt6 Desktop App

## Project Overview
A desktop application that trains a BERT model on a Twitter sentiment dataset and provides a PyQt6 GUI for real-time sentiment prediction.

## Dataset
- **Source**: Custom Twitter Sentiment Dataset (Sentiment140-style)
- **File**: `dataset/twitter_sentiment.csv`
- **Columns**: `tweet` (text), `sentiment` (positive/negative/neutral)
- **Size**: 900 balanced samples (300 per class)

## Model
- **Model**: `bert-base-uncased` (HuggingFace Transformers)
- **Task**: 3-class sentiment classification
- **Backend**: PyTorch
- **Saved to**: `saved_bert_model/`

## Training Configuration
| Parameter | Value |
|---|---|
| Model | bert-base-uncased |
| Max Length | 128 tokens |
| Batch Size | 16 |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Train/Test Split | 80/20 |
| Optimizer | AdamW |

## Project Structure
```
bert_sentiment/
├── train_bert.py          ← Training script
├── app.py                 ← PyQt6 GUI application
├── requirements.txt
├── README.md
├── dataset/
│   └── twitter_sentiment.csv
├── saved_bert_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── training_meta.json
└── graphs/
    ├── class_distribution.png
    ├── loss_curve.png
    ├── accuracy_curve.png
    └── confusion_matrix.png
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_bert.py
```
This will:
- Load `dataset/twitter_sentiment.csv`
- Train BERT for 3 epochs
- Save model to `saved_bert_model/`
- Generate all evaluation graphs in `graphs/`

### 3. Launch the GUI
```bash
python app.py
```

### 4. Using the GUI
1. Click **Load BERT Model** → select the `saved_bert_model/` folder
2. Click **Load Dataset CSV** → select `dataset/twitter_sentiment.csv`
3. Click any tweet in the table to predict its sentiment
4. Or type a custom sentence in the manual input box and click **Predict**

## GUI Features
- Load Dataset CSV button
- Load BERT Model button
- Scrollable tweet table with actual labels
- Click-to-predict on any tweet
- Manual text input for custom sentences
- Sentiment prediction result (POSITIVE / NEGATIVE / NEUTRAL)
- Confidence percentage + progress bar
- Per-class probability breakdown

## Paul's Critical Thinking Standards
| Standard | Application |
|---|---|
| Clarity | Dataset columns (tweet, sentiment), GUI workflow clearly documented |
| Accuracy | Real evaluation metrics reported, no results hidden |
| Precision | Exact model name, split 80/20, 3 epochs, batch 16, LR 2e-5 |
| Relevance | All 4 graphs directly support model performance |
| Depth | Performance analysis across sentiment classes |
| Logic | Training → saving → loading → GUI prediction connected end-to-end |
| Fairness | Dataset balanced (300/class); slang and irony are known BERT limitations |

## GUI
![Uploading Screenshot 2026-07-05 191227.png…]()
