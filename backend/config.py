"""
Cấu hình hyperparameters và đường dẫn cho project.
"""
import os

# === Đường dẫn ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

TRAIN_CSV = os.path.join(DATA_DIR, "sentiment_data.csv")
TEST_CSV = os.path.join(DATA_DIR, "sentiment_test.csv")
VAL_CSV = os.path.join(DATA_DIR, "sentiment_validation.csv")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")

LSTM_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "lstm_model.pth")
GRU_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "gru_model.pth")

# === Model Hyperparameters ===
EMBED_DIM = 128       # Embedding dimension
HIDDEN_DIM = 64       # Hidden state dimension
OUTPUT_DIM = 2        # Binary classification (pos/neg)
N_LAYERS = 1          # Single layer for demo clarity
DROPOUT = 0.3         # Dropout rate

# === Training Hyperparameters ===
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 20
MAX_SEQ_LEN = 50      # Max tokens per sentence
MIN_FREQ = 2          # Min word frequency for vocab

# === Special Tokens ===
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

# === Server ===
HOST = "0.0.0.0"
PORT = 8000

# Tạo thư mục checkpoints nếu chưa có
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
