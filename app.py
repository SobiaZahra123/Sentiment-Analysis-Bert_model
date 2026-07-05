import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import torch
import numpy as np
import math

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QTextEdit,
    QFileDialog, QFrame, QHeaderView, QSplitter,
    QProgressBar, QStatusBar, QGroupBox, QGridLayout, QMessageBox,
    QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette

# ── Colours ───────────────────────────────────────────────────
C_BG       = "#F0F4FF"
C_PANEL    = "#FFFFFF"
C_NAVY     = "#1B2A6B"
C_BLUE     = "#2563EB"
C_GREEN    = "#16A34A"
C_RED      = "#DC2626"
C_AMBER    = "#D97706"
C_GRAY     = "#6B7280"
C_BORDER   = "#D1D5DB"
C_ROW_ALT  = "#F8FAFF"
C_SEL      = "#DBEAFE"
C_TEXT     = "#1F2937"

LABEL_COLORS = {
    'positive': ('#DCFCE7', '#15803D'),
    'negative': ('#FEE2E2', '#991B1B'),
    'neutral':  ('#FEF9C3', '#92400E'),
}

# ── Worker thread for model inference ─────────────────────────
class PredictWorker(QThread):
    result_ready = pyqtSignal(str, float, list)
    error        = pyqtSignal(str)

    def __init__(self, model, tokenizer, text, max_len, id2label):
        super().__init__()
        self.model     = model
        self.tokenizer = tokenizer
        self.text      = text
        self.max_len   = max_len
        self.id2label  = id2label

    def run(self):
        try:
            enc = self.tokenizer(
                self.text, max_length=self.max_len,
                padding='max_length', truncation=True,
                return_tensors='pt'
            )
            with torch.no_grad():
                out    = self.model(**enc)
                probs  = torch.softmax(out.logits, dim=1).squeeze().tolist()
                pred   = int(torch.argmax(out.logits, dim=1).item())
                label  = self.id2label[str(pred)]
                conf   = max(probs)
            self.result_ready.emit(label, conf, probs)
        except Exception as e:
            self.error.emit(str(e))

# ── Worker thread for loading model ───────────────────────────
class LoadModelWorker(QThread):
    loaded  = pyqtSignal(object, object, dict)
    error   = pyqtSignal(str)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        try:
            from transformers import BertTokenizer, BertForSequenceClassification
            meta_path = os.path.join(self.folder, 'training_meta.json')
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            tokenizer = BertTokenizer.from_pretrained(self.folder)
            model     = BertForSequenceClassification.from_pretrained(self.folder)
            model.eval()
            self.loaded.emit(model, tokenizer, meta)
        except Exception as e:
            self.error.emit(str(e))

# ── Styled button helper ───────────────────────────────────────
def make_btn(text, color, hover, min_w=120, h=36):
    btn = QPushButton(text)
    btn.setMinimumSize(min_w, h)
    btn.setFont(QFont("Arial", 9, QFont.Weight.Bold))
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {color};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 5px 14px;
        }}
        QPushButton:hover  {{ background-color: {hover}; }}
        QPushButton:pressed{{ background-color: {color}; }}
        QPushButton:disabled{{ background-color: #9CA3AF; }}
    """)
    return btn

# ── Main Window ───────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model     = None
        self.tokenizer = None
        self.meta      = {}
        self.df        = None
        self.display_rows = 200
        self.current_page = 0
        self.page_count   = 0
        self.max_len   = 128
        self.id2label  = {'0':'negative','1':'neutral','2':'positive'}
        self.worker    = None
        self._init_ui()

    # ── UI Setup ─────────────────────────────────────────────
    def _init_ui(self):
        self.setWindowTitle("Twitter Sentiment Analysis — BERT PyQt6 GUI")
        self.setMinimumSize(1200, 750)
        self.setStyleSheet(f"QMainWindow {{ background-color: {C_BG}; }}")

        central = QWidget()
        central.setStyleSheet(f"background-color: {C_BG};")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # ── Header ──────────────────────────────────────────
        hdr = QFrame()
        hdr.setStyleSheet(f"background-color:{C_NAVY}; border-radius:12px;")
        hdr.setFixedHeight(65)
        hdr_lay = QVBoxLayout(hdr)
        hdr_lay.setContentsMargins(20, 0, 20, 0)
        title_lbl = QLabel("Twitter Sentiment Analysis — BERT PyQt GUI")
        title_lbl.setFont(QFont("Arial", 15, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color:white;")
        sub_lbl = QLabel("Load dataset · Load trained model · Click a tweet · View predicted sentiment")
        sub_lbl.setFont(QFont("Arial", 9))
        sub_lbl.setStyleSheet("color:#93C5FD;")
        hdr_lay.addWidget(title_lbl)
        hdr_lay.addWidget(sub_lbl)
        root.addWidget(hdr)

        # ── Toolbar ─────────────────────────────────────────
        tb = QFrame()
        tb.setStyleSheet(f"background-color:{C_PANEL}; border-radius:10px; border:1px solid {C_BORDER};")
        tb.setFixedHeight(55)
        tb_lay = QHBoxLayout(tb)
        tb_lay.setContentsMargins(14, 0, 14, 0)
        tb_lay.setSpacing(10)

        self.btn_load_csv   = make_btn("📂 Load Dataset CSV", C_BLUE, "#1D4ED8")
        self.btn_load_model = make_btn("🤖 Load BERT Model",  "#7C3AED", "#6D28D9")
        self.btn_predict    = make_btn("⚡ Predict",           C_GREEN, "#15803D", min_w=100)
        self.btn_predict.setEnabled(False)

        self.status_lbl = QLabel("Model Status: Not loaded")
        self.status_lbl.setFont(QFont("Arial", 10))
        self.status_lbl.setStyleSheet(f"color:{C_GRAY}; padding:4px 12px; "
            f"background:#F3F4F6; border-radius:8px; border:1px solid {C_BORDER};")
        self.status_lbl.setMinimumWidth(260)

        self.btn_load_csv.clicked.connect(self._load_dataset)
        self.btn_load_model.clicked.connect(self._load_model)
        self.btn_predict.clicked.connect(self._predict_manual)

        tb_lay.addWidget(self.btn_load_csv)
        tb_lay.addWidget(self.btn_load_model)
        tb_lay.addWidget(self.btn_predict)
        tb_lay.addStretch()
        tb_lay.addWidget(self.status_lbl)
        root.addWidget(tb)

        # ── Progress bar ──────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(3)
        self.progress.setStyleSheet(f"QProgressBar{{border:none;background:{C_BG};}} "
            f"QProgressBar::chunk{{background:{C_BLUE};}}")
        self.progress.hide()
        root.addWidget(self.progress)

        # ── Main splitter ────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle{background:#E5E7EB;}")

        # ── LEFT — tweet table ──────────────────────────────
        left = QFrame()
        left.setStyleSheet(f"background:{C_PANEL}; border-radius:12px; border:1px solid {C_BORDER};")
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(12, 10, 12, 10)
        left_lay.setSpacing(6)

        tweets_hdr = QLabel("📋 Loaded Tweets")
        tweets_hdr.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        tweets_hdr.setStyleSheet(f"color:{C_NAVY};")

        self.tweet_count_lbl = QLabel("No dataset loaded")
        self.tweet_count_lbl.setFont(QFont("Arial", 8))
        self.tweet_count_lbl.setStyleSheet(f"color:{C_GRAY};")

        page_nav = QHBoxLayout()
        page_nav.setContentsMargins(0, 0, 0, 0)
        page_nav.setSpacing(8)
        self.btn_prev_page = make_btn("◀ Prev", C_GRAY, "#9CA3AF", min_w=80)
        self.btn_next_page = make_btn("Next ▶", C_GRAY, "#9CA3AF", min_w=80)
        self.page_lbl = QLabel("Page 0 of 0")
        self.page_lbl.setFont(QFont("Arial", 9))
        self.page_lbl.setStyleSheet(f"color:{C_GRAY};")
        self.btn_prev_page.clicked.connect(self._prev_page)
        self.btn_next_page.clicked.connect(self._next_page)
        self.btn_prev_page.setEnabled(False)
        self.btn_next_page.setEnabled(False)
        page_nav.addWidget(self.btn_prev_page)
        page_nav.addWidget(self.page_lbl)
        page_nav.addWidget(self.btn_next_page)
        page_nav.addStretch()

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["#", "Tweet Text", "Actual"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 35)
        self.table.setColumnWidth(2, 85)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                border:none; gridline-color:{C_BORDER};
                font-size:11px; background:{C_PANEL};
            }}
            QTableWidget::item:selected {{
                background:{C_SEL}; color:{C_NAVY};
            }}
            QHeaderView::section {{
                background:{C_NAVY}; color:white;
                font-weight:bold; font-size:11px;
                padding:4px; border:none;
            }}
            QTableWidget::item:alternate {{ background:{C_ROW_ALT}; }}
        """)
        self.table.itemClicked.connect(self._on_tweet_clicked)
        self.table.verticalHeader().hide()
        self.table.setWordWrap(True)

        left_lay.addWidget(tweets_hdr)
        left_lay.addWidget(self.tweet_count_lbl)
        left_lay.addLayout(page_nav)
        left_lay.addWidget(self.table)
        splitter.addWidget(left)

        # ── RIGHT — prediction panel ──────────────────────────
        right = QFrame()
        right.setStyleSheet(f"background:{C_PANEL}; border-radius:12px; border:1px solid {C_BORDER};")
        right.setMinimumWidth(350)
        right.setMaximumWidth(420)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(14, 12, 14, 12)
        right_lay.setSpacing(10)

        pred_hdr = QLabel("🎯 Prediction Panel")
        pred_hdr.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        pred_hdr.setStyleSheet(f"color:{C_NAVY};")
        right_lay.addWidget(pred_hdr)

        # ── Selected Sentence ──────────────────────────────────
        sel_grp = QGroupBox("Selected Sentence")
        sel_grp.setStyleSheet(f"""
            QGroupBox {{ font-weight:bold; font-size:10px; color:{C_NAVY};
                border:2px solid {C_BORDER}; border-radius:8px; margin-top:8px; padding:6px; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 4px; }}
        """)
        sel_lay = QVBoxLayout(sel_grp)
        sel_lay.setContentsMargins(8, 6, 8, 6)
        self.selected_lbl = QLabel("Click a tweet from the table\nor type below and click Predict.")
        self.selected_lbl.setWordWrap(True)
        self.selected_lbl.setFont(QFont("Arial", 10))
        self.selected_lbl.setStyleSheet(f"color:{C_GRAY}; padding:4px; background:#FAFBFC; border-radius:4px;")
        self.selected_lbl.setMinimumHeight(50)
        sel_lay.addWidget(self.selected_lbl)
        right_lay.addWidget(sel_grp)

        # ── Predicted Sentiment ──────────────────────────────
        res_grp = QGroupBox("Predicted Sentiment")
        res_grp.setStyleSheet(f"""
            QGroupBox {{ font-weight:bold; font-size:10px; color:{C_NAVY};
                border:2px solid {C_BORDER}; border-radius:8px; margin-top:8px; padding:6px; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 4px; }}
        """)
        res_lay = QVBoxLayout(res_grp)
        res_lay.setContentsMargins(8, 6, 8, 6)
        self.pred_result_lbl = QLabel("—")
        self.pred_result_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pred_result_lbl.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.pred_result_lbl.setFixedHeight(55)
        self.pred_result_lbl.setStyleSheet(f"background:#F3F4F6; border-radius:8px; color:{C_GRAY};")
        res_lay.addWidget(self.pred_result_lbl)
        right_lay.addWidget(res_grp)

        # ── Confidence ──────────────────────────────────────────
        conf_grp = QGroupBox("Confidence")
        conf_grp.setStyleSheet(f"""
            QGroupBox {{ font-weight:bold; font-size:10px; color:{C_NAVY};
                border:2px solid {C_BORDER}; border-radius:8px; margin-top:8px; padding:6px; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 4px; }}
        """)
        conf_lay = QVBoxLayout(conf_grp)
        conf_lay.setContentsMargins(8, 6, 8, 6)
        
        self.conf_lbl = QLabel("0%")
        self.conf_lbl.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.conf_lbl.setStyleSheet(f"color:{C_NAVY};")
        self.conf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        self.conf_bar.setFixedHeight(12)
        self.conf_bar.setStyleSheet(f"""
            QProgressBar {{ border:none; background:#E5E7EB; border-radius:6px; }}
            QProgressBar::chunk {{ background:{C_BLUE}; border-radius:6px; }}
        """)
        conf_lay.addWidget(self.conf_lbl)
        conf_lay.addWidget(self.conf_bar)
        right_lay.addWidget(conf_grp)

        # ── Class Probabilities ───────────────────────────────
        prob_grp = QGroupBox("Class Probabilities")
        prob_grp.setStyleSheet(f"""
            QGroupBox {{ font-weight:bold; font-size:10px; color:{C_NAVY};
                border:2px solid {C_BORDER}; border-radius:8px; margin-top:8px; padding:6px; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 4px; }}
        """)
        prob_lay = QGridLayout(prob_grp)
        prob_lay.setContentsMargins(8, 6, 8, 6)
        prob_lay.setVerticalSpacing(4)
        
        self.prob_labels = {}
        self.prob_bars = {}
        for row, (cls, (bg, fg)) in enumerate(LABEL_COLORS.items()):
            lbl = QLabel(cls.capitalize())
            lbl.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            lbl.setStyleSheet(f"color:{fg};")
            
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(10)
            bar.setStyleSheet(f"""
                QProgressBar{{ border:none; background:#E5E7EB; border-radius:4px; }}
                QProgressBar::chunk{{ background:{fg}; border-radius:4px; }}
            """)
            
            pct = QLabel("0%")
            pct.setFont(QFont("Arial", 10))
            pct.setStyleSheet(f"color:{C_GRAY};")
            pct.setMinimumWidth(45)
            pct.setAlignment(Qt.AlignmentFlag.AlignRight)
            
            prob_lay.addWidget(lbl, row, 0)
            prob_lay.addWidget(bar, row, 1)
            prob_lay.addWidget(pct, row, 2)
            self.prob_labels[cls] = pct
            self.prob_bars[cls] = bar
        
        right_lay.addWidget(prob_grp)

        # ── Manual Input ──────────────────────────────────────
        manual_grp = QGroupBox("✏️ Manual Input")
        manual_grp.setStyleSheet(f"""
            QGroupBox {{ font-weight:bold; font-size:10px; color:{C_NAVY};
                border:2px solid {C_BORDER}; border-radius:8px; margin-top:8px; padding:6px; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 4px; }}
        """)
        manual_lay = QVBoxLayout(manual_grp)
        manual_lay.setContentsMargins(8, 6, 8, 6)
        
        self.manual_input = QTextEdit()
        self.manual_input.setPlaceholderText("Type a tweet here and click Predict…")
        self.manual_input.setFixedHeight(60)
        self.manual_input.setFont(QFont("Arial", 10))
        self.manual_input.setStyleSheet(f"""
            QTextEdit {{
                border:2px solid {C_BORDER};
                border-radius:8px;
                padding:6px;
                background:white;
                font-size:11px;
            }}
            QTextEdit:focus {{ border-color:{C_BLUE}; }}
        """)
        manual_lay.addWidget(self.manual_input)
        right_lay.addWidget(manual_grp)
        right_lay.addStretch()

        splitter.addWidget(right)
        splitter.setSizes([700, 400])
        root.addWidget(splitter)

        # ── Status bar ───────────────────────────────────────
        self.statusBar().setStyleSheet(f"QStatusBar{{ background:{C_PANEL}; color:{C_GRAY}; font-size:10px; }}")
        self.statusBar().showMessage("Ready — Load a BERT model and dataset to begin.")

    # ── Load Dataset ──────────────────────────────────────────
    def _load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Twitter Dataset CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            self.df = pd.read_csv(path, encoding='utf-8', dtype=str, low_memory=False)
            # Find tweet and label columns
            text_col = next((c for c in self.df.columns if 'tweet' in c.lower() or 'text' in c.lower()), self.df.columns[0])
            label_col = next((c for c in self.df.columns if 'sentiment' in c.lower() or 'label' in c.lower()), self.df.columns[-1])
            self.df = self.df.rename(columns={text_col:'tweet', label_col:'sentiment'})
            self.df = self.df.reset_index(drop=True)
            self.current_page = 0
            self.page_count = math.ceil(len(self.df) / self.display_rows) if len(self.df) else 0
            self._populate_table()
            self.statusBar().showMessage(f"Dataset loaded: {len(self.df)} tweets from {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset:\n{e}")

    def _populate_table(self):
        if self.df is None:
            return
        self.table.setRowCount(0)
        start = self.current_page * self.display_rows
        end = min(start + self.display_rows, len(self.df))
        display = self.df.iloc[start:end]
        self.page_lbl.setText(f"Page {self.current_page + 1} of {self.page_count}")
        self.btn_prev_page.setEnabled(self.current_page > 0)
        self.btn_next_page.setEnabled(self.current_page < self.page_count - 1)
        self.table.setRowCount(len(display))
        for i, (_, row) in enumerate(display.iterrows()):
            num = QTableWidgetItem(str(i+1 + start))
            num.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            tweet = QTableWidgetItem(str(row['tweet'])[:100] + ("..." if len(str(row['tweet'])) > 100 else ""))
            sent = str(row.get('sentiment','')).lower()
            label = QTableWidgetItem(sent.capitalize())
            label.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if 'positive' in sent:
                label.setForeground(QColor("#15803D"))
            elif 'negative' in sent:
                label.setForeground(QColor("#DC2626"))
            elif 'neutral' in sent:
                label.setForeground(QColor("#D97706"))
            else:
                label.setForeground(QColor("#6B7280"))
            self.table.setItem(i, 0, num)
            self.table.setItem(i, 1, tweet)
            self.table.setItem(i, 2, label)
        self.table.resizeRowsToContents()
        total = len(self.df)
        self.tweet_count_lbl.setText(f"Showing {start + 1}-{end} of {total} tweets")

    # ── Load Model ────────────────────────────────────────────
    def _load_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Saved BERT Model Folder")
        if not folder:
            return
        self.progress.show()
        self.btn_load_model.setEnabled(False)
        self.status_lbl.setText("Model Status: Loading…")
        self.statusBar().showMessage("Loading BERT model, please wait…")
        self._load_worker = LoadModelWorker(folder)
        self._load_worker.loaded.connect(self._on_model_loaded)
        self._load_worker.error.connect(self._on_model_error)
        self._load_worker.start()

    def _on_model_loaded(self, model, tokenizer, meta):
        self.model = model
        self.tokenizer = tokenizer
        self.meta = meta
        self.max_len = meta.get('max_len', 128)
        id2l = meta.get('id2label', {'0':'negative','1':'neutral','2':'positive'})
        self.id2label = {str(k): v for k, v in id2l.items()}
        self.progress.hide()
        self.btn_load_model.setEnabled(True)
        self.btn_predict.setEnabled(True)
        folder_name = os.path.basename(self._load_worker.folder)
        self.status_lbl.setText(f"Model Status: ✅ {folder_name} loaded")
        self.status_lbl.setStyleSheet(
            "color:#15803D; padding:4px 12px; background:#DCFCE7; "
            "border-radius:8px; border:1px solid #86EFAC; font-weight:bold;")
        acc = meta.get('accuracy','N/A')
        f1 = meta.get('f1','N/A')
        self.statusBar().showMessage(f"Model loaded | Accuracy: {acc} | F1: {f1} | Ready to predict")

    def _on_model_error(self, err):
        self.progress.hide()
        self.btn_load_model.setEnabled(True)
        QMessageBox.critical(self, "Model Load Error", f"Failed to load model:\n{err}")
        self.statusBar().showMessage("Model load failed.")

    # ── Tweet click ───────────────────────────────────────────
    def _on_tweet_clicked(self, item):
        row = item.row()
        tweet_item = self.table.item(row, 1)
        if tweet_item:
            actual_index = self.current_page * self.display_rows + row
            full_text = self.df.iloc[actual_index]['tweet']
            self.selected_lbl.setText(full_text)
            self.manual_input.setPlainText(full_text)
            if self.model:
                self._run_prediction(full_text)

    def _prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self._populate_table()

    def _next_page(self):
        if self.current_page < self.page_count - 1:
            self.current_page += 1
            self._populate_table()

    # ── Manual predict ────────────────────────────────────────
    def _predict_manual(self):
        text = self.manual_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Empty Input", "Please type a sentence first.")
            return
        if not self.model:
            QMessageBox.warning(self, "No Model", "Please load a BERT model first.")
            return
        self.selected_lbl.setText(text)
        self._run_prediction(text)

    def _run_prediction(self, text):
        self.pred_result_lbl.setText("…")
        self.pred_result_lbl.setStyleSheet(f"background:#F3F4F6; border-radius:8px; color:{C_GRAY};")
        self.statusBar().showMessage("Running prediction…")
        self.worker = PredictWorker(self.model, self.tokenizer, text, self.max_len, self.id2label)
        self.worker.result_ready.connect(self._show_result)
        self.worker.error.connect(lambda e: self.statusBar().showMessage(f"Error: {e}"))
        self.worker.start()

    def _show_result(self, label, confidence, probs):
        bg, fg = LABEL_COLORS.get(label, ('#F3F4F6','#374151'))
        self.pred_result_lbl.setText(label.upper())
        self.pred_result_lbl.setStyleSheet(
            f"background:{bg}; border-radius:8px; color:{fg}; "
            f"border:2px solid {fg}; font-size:24px; font-weight:bold;")
        pct = int(confidence * 100)
        self.conf_lbl.setText(f"{pct}%")
        self.conf_bar.setValue(pct)
        cls_order = ['negative','neutral','positive']
        for i, cls in enumerate(cls_order):
            p = probs[i] if i < len(probs) else 0
            val = int(p * 100)
            self.prob_bars[cls].setValue(val)
            self.prob_labels[cls].setText(f"{val}%")
        self.statusBar().showMessage(f"Predicted: {label.upper()} | Confidence: {pct}%")

# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
