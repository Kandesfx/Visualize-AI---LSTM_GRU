"""
Training script cho LSTM và GRU models.
Chạy: python -m training.train --model lstm --epochs 20
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_CSV, TEST_CSV, VAL_CSV, VOCAB_PATH,
    LSTM_MODEL_PATH, GRU_MODEL_PATH,
    EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT,
    LEARNING_RATE, BATCH_SIZE, EPOCHS, MAX_SEQ_LEN, MIN_FREQ, PAD_IDX
)
from training.preprocessor import Preprocessor
from training.dataset import SentimentDataset
from models.lstm_model import LSTMSentiment
from models.gru_model import GRUSentiment


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train 1 epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (texts, labels) in enumerate(dataloader):
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred_labels = predictions.argmax(dim=1)
        correct += (pred_labels == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            pred_labels = predictions.argmax(dim=1)
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train LSTM/GRU Sentiment Model')
    parser.add_argument('--model', type=str, choices=['lstm', 'gru'], required=True,
                        help='Loại model: lstm hoặc gru')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Số epochs (default: {EPOCHS})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    args = parser.parse_args()
    
    device = torch.device('cpu')
    print(f"\n{'='*60}")
    print(f"  Training {args.model.upper()} Sentiment Model")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"{'='*60}\n")
    
    # 1. Build vocabulary từ training data
    print("[1/5] Xây dựng vocabulary...")
    train_df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    preprocessor = Preprocessor()
    vocab = preprocessor.build_vocab(train_df['sentence'].tolist(), min_freq=MIN_FREQ)
    preprocessor.save_vocab(VOCAB_PATH)
    vocab_size = len(vocab)
    print(f"       Vocab size: {vocab_size}\n")
    
    # 2. Tạo datasets
    print("[2/5] Tạo datasets...")
    train_dataset = SentimentDataset(TRAIN_CSV, preprocessor, MAX_SEQ_LEN)
    val_dataset = SentimentDataset(VAL_CSV, preprocessor, MAX_SEQ_LEN)
    test_dataset = SentimentDataset(TEST_CSV, preprocessor, MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"       Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}\n")
    
    # 3. Tạo model
    print("[3/5] Khởi tạo model...")
    if args.model == 'lstm':
        model = LSTMSentiment(
            vocab_size=vocab_size,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            n_layers=N_LAYERS,
            dropout=DROPOUT,
            pad_idx=PAD_IDX
        )
        save_path = LSTM_MODEL_PATH
    else:
        model = GRUSentiment(
            vocab_size=vocab_size,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            n_layers=N_LAYERS,
            dropout=DROPOUT,
            pad_idx=PAD_IDX
        )
        save_path = GRU_MODEL_PATH
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Model: {args.model.upper()}")
    print(f"       Total params: {total_params:,}")
    print(f"       Trainable params: {trainable_params:,}\n")
    
    # 4. Training
    print("[4/5] Bắt đầu training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_acc = 0
    best_epoch = 0
    
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Train Acc':>10} | {'Val Loss':>10} | {'Val Acc':>9} | {'Time':>6}")
    print("-" * 70)
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        elapsed = time.time() - start
        
        # Lưu model tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'embed_dim': EMBED_DIM,
                'hidden_dim': HIDDEN_DIM,
                'output_dim': OUTPUT_DIM,
                'n_layers': N_LAYERS,
                'dropout': DROPOUT,
                'model_type': args.model
            }, save_path)
            marker = " ★"
        else:
            marker = ""
        
        print(f"{epoch:>6} | {train_loss:>11.4f} | {train_acc:>9.2%} | {val_loss:>10.4f} | {val_acc:>8.2%} | {elapsed:>5.1f}s{marker}")
    
    print(f"\n       Best model tại epoch {best_epoch} (Val Acc: {best_val_acc:.2%})")
    print(f"       Đã lưu: {save_path}")
    
    # 5. Test
    print(f"\n[5/5] Đánh giá trên tập test...")
    # Load best model
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"       Test Loss: {test_loss:.4f}")
    print(f"       Test Acc:  {test_acc:.2%}")
    print(f"\n{'='*60}")
    print(f"  ✅ Training {args.model.upper()} hoàn tất!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
