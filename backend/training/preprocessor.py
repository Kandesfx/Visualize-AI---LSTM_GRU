"""
Preprocessor cho tiếng Việt.
Bao gồm: text cleaning, tokenization, vocabulary building.
"""
import re
import json
import unicodedata
from collections import Counter

import pandas as pd

# Thử import underthesea, fallback sang split đơn giản nếu không có
try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False
    print("[WARNING] underthesea không được cài. Sử dụng tokenizer đơn giản.")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PAD_TOKEN, UNK_TOKEN, PAD_IDX, UNK_IDX, MIN_FREQ, VOCAB_PATH


class Preprocessor:
    """
    Xử lý text tiếng Việt và quản lý vocabulary.
    """
    
    def __init__(self, vocab=None):
        self.vocab = vocab  # word -> index
        self.idx_to_word = None  # index -> word
        if vocab:
            self.idx_to_word = {v: k for k, v in vocab.items()}
    
    @staticmethod
    def clean_text(text):
        """Làm sạch text tiếng Việt."""
        # Chuẩn hóa Unicode
        text = unicodedata.normalize('NFC', text)
        # Lowercase
        text = text.lower().strip()
        # Xóa ký tự đặc biệt nhưng giữ dấu tiếng Việt và dấu câu cơ bản
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ,.]', '', text)
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def tokenize(text):
        """Tách từ tiếng Việt."""
        if HAS_UNDERTHESEA:
            tokens = word_tokenize(text, format="list")
        else:
            tokens = text.split()
        # Lọc bỏ token rỗng và dấu câu đơn lẻ
        tokens = [t.strip() for t in tokens if t.strip() and t.strip() not in [',', '.']]
        return tokens
    
    @staticmethod
    def process_text(text):
        """Pipeline hoàn chỉnh: clean → tokenize."""
        cleaned = Preprocessor.clean_text(text)
        tokens = Preprocessor.tokenize(cleaned)
        return tokens
    
    def build_vocab(self, texts, min_freq=MIN_FREQ):
        """
        Xây dựng vocabulary từ danh sách texts.
        
        Args:
            texts: list[str] - danh sách câu
            min_freq: int - tần suất tối thiểu để vào vocab
        """
        counter = Counter()
        for text in texts:
            tokens = self.process_text(text)
            counter.update(tokens)
        
        # Bắt đầu với special tokens
        self.vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        idx = 2
        
        for word, freq in counter.most_common():
            if freq >= min_freq:
                self.vocab[word] = idx
                idx += 1
        
        self.idx_to_word = {v: k for k, v in self.vocab.items()}
        print(f"[Vocab] Xây dựng xong: {len(self.vocab)} từ (min_freq={min_freq})")
        return self.vocab
    
    def save_vocab(self, path=VOCAB_PATH):
        """Lưu vocabulary ra file JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        print(f"[Vocab] Đã lưu tại: {path}")
    
    @staticmethod
    def load_vocab(path=VOCAB_PATH):
        """Tải vocabulary từ file JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        print(f"[Vocab] Đã tải: {len(vocab)} từ từ {path}")
        return vocab
    
    def encode(self, text, max_len=50):
        """
        Chuyển text thành danh sách indices.
        
        Args:
            text: str - câu input
            max_len: int - độ dài tối đa
            
        Returns:
            list[int] - token indices (đã pad/truncate)
        """
        tokens = self.process_text(text)
        indices = []
        for token in tokens[:max_len]:
            idx = self.vocab.get(token, UNK_IDX)
            indices.append(idx)
        
        # Padding
        while len(indices) < max_len:
            indices.append(PAD_IDX)
        
        return indices, tokens
    
    def encode_tokens_only(self, text):
        """
        Chuyển text thành indices KHÔNG padding (cho demo).
        Trả về cả tokens và indices.
        """
        tokens = self.process_text(text)
        indices = []
        for token in tokens:
            idx = self.vocab.get(token, UNK_IDX)
            indices.append(idx)
        return indices, tokens
