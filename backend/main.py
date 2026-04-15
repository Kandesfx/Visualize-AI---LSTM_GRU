"""
FastAPI Server - Entry point.
Serve cả API endpoints và frontend static files.
"""
import os
import sys
import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from config import (
    LSTM_MODEL_PATH, GRU_MODEL_PATH, VOCAB_PATH,
    EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, PAD_IDX,
    FRONTEND_DIR, HOST, PORT
)
from models.lstm_model import LSTMSentiment
from models.gru_model import GRUSentiment
from models.state_extractor import LSTMStateExtractor, GRUStateExtractor
from training.preprocessor import Preprocessor
from api.routes import router, setup


app = FastAPI(
    title="LSTM & GRU Demo API",
    description="API cho web demo trực quan LSTM & GRU - Phân tích cảm xúc feedback sinh viên",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_models():
    """Tải models và preprocessor khi server khởi động."""
    device = torch.device('cpu')
    
    # 1. Load vocabulary
    if not os.path.exists(VOCAB_PATH):
        print(f"[ERROR] Không tìm thấy vocab tại {VOCAB_PATH}")
        print("        Hãy chạy training trước: python -m training.train --model lstm --epochs 20")
        return None, None, None, None, None
    
    vocab = Preprocessor.load_vocab(VOCAB_PATH)
    preprocessor = Preprocessor(vocab)
    vocab_size = len(vocab)
    
    # 2. Load LSTM model
    lstm_model = None
    lstm_extractor = None
    if os.path.exists(LSTM_MODEL_PATH):
        print(f"[Model] Đang tải LSTM từ {LSTM_MODEL_PATH}...")
        checkpoint = torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=False)
        lstm_model = LSTMSentiment(
            vocab_size=vocab_size,
            embed_dim=checkpoint.get('embed_dim', EMBED_DIM),
            hidden_dim=checkpoint.get('hidden_dim', HIDDEN_DIM),
            output_dim=checkpoint.get('output_dim', OUTPUT_DIM),
            n_layers=checkpoint.get('n_layers', N_LAYERS),
            dropout=0,  # No dropout ở inference
            pad_idx=PAD_IDX
        )
        lstm_model.load_state_dict(checkpoint['model_state_dict'])
        lstm_model.eval()
        lstm_extractor = LSTMStateExtractor(lstm_model)
        print(f"[Model] ✅ LSTM đã tải thành công")
    else:
        print(f"[Model] ⚠️  Không tìm thấy LSTM model tại {LSTM_MODEL_PATH}")
    
    # 3. Load GRU model
    gru_model = None
    gru_extractor = None
    if os.path.exists(GRU_MODEL_PATH):
        print(f"[Model] Đang tải GRU từ {GRU_MODEL_PATH}...")
        checkpoint = torch.load(GRU_MODEL_PATH, map_location=device, weights_only=False)
        gru_model = GRUSentiment(
            vocab_size=vocab_size,
            embed_dim=checkpoint.get('embed_dim', EMBED_DIM),
            hidden_dim=checkpoint.get('hidden_dim', HIDDEN_DIM),
            output_dim=checkpoint.get('output_dim', OUTPUT_DIM),
            n_layers=checkpoint.get('n_layers', N_LAYERS),
            dropout=0,
            pad_idx=PAD_IDX
        )
        gru_model.load_state_dict(checkpoint['model_state_dict'])
        gru_model.eval()
        gru_extractor = GRUStateExtractor(gru_model)
        print(f"[Model] ✅ GRU đã tải thành công")
    else:
        print(f"[Model] ⚠️  Không tìm thấy GRU model tại {GRU_MODEL_PATH}")
    
    return lstm_model, gru_model, lstm_extractor, gru_extractor, preprocessor


# Load models khi startup
@app.on_event("startup")
async def startup_event():
    print("\n" + "="*50)
    print("  🧠 LSTM & GRU Demo Server")
    print("="*50 + "\n")
    
    lstm_m, gru_m, lstm_ext, gru_ext, prep = load_models()
    setup(lstm_m, gru_m, lstm_ext, gru_ext, prep)
    
    print(f"\n[Server] Frontend: {FRONTEND_DIR}")
    print(f"[Server] Đang chạy tại http://localhost:{PORT}")
    print(f"[Server] API docs: http://localhost:{PORT}/docs\n")


# Register API routes
app.include_router(router)

# Serve frontend static files
if os.path.exists(FRONTEND_DIR):
    app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/lstm")
    async def serve_lstm():
        return FileResponse(os.path.join(FRONTEND_DIR, "lstm.html"))

    @app.get("/gru")
    async def serve_gru():
        return FileResponse(os.path.join(FRONTEND_DIR, "gru.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
