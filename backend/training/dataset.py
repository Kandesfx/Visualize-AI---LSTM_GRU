"""
PyTorch Dataset cho sentiment analysis.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_SEQ_LEN, PAD_IDX


class SentimentDataset(Dataset):
    """
    Dataset cho phân tích cảm xúc feedback sinh viên.
    """
    
    def __init__(self, csv_path, preprocessor, max_len=MAX_SEQ_LEN):
        """
        Args:
            csv_path: đường dẫn file CSV
            preprocessor: Preprocessor instance (đã có vocab)
            max_len: độ dài tối đa sequence
        """
        self.preprocessor = preprocessor
        self.max_len = max_len
        
        # Đọc CSV
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        self.sentences = df['sentence'].tolist()
        self.labels = df['label'].tolist()
        
        print(f"[Dataset] Đã tải {len(self.sentences)} mẫu từ {os.path.basename(csv_path)}")
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        text = str(self.sentences[idx])
        label = int(self.labels[idx])
        
        # Encode text
        indices, _ = self.preprocessor.encode(text, self.max_len)
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)
