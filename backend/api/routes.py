"""
API Routes cho prediction endpoints.
"""
import torch
from fastapi import APIRouter, HTTPException

from api.schemas import PredictRequest, PredictResponse, HealthResponse

router = APIRouter()

# Globals - sẽ được set bởi main.py khi khởi động
lstm_model = None
gru_model = None
lstm_extractor = None
gru_extractor = None
preprocessor = None


def setup(lstm_m, gru_m, lstm_ext, gru_ext, prep):
    """Inject dependencies từ main.py."""
    global lstm_model, gru_model, lstm_extractor, gru_extractor, preprocessor
    lstm_model = lstm_m
    gru_model = gru_m
    lstm_extractor = lstm_ext
    gru_extractor = gru_ext
    preprocessor = prep


@router.post("/api/predict/lstm", response_model=PredictResponse)
async def predict_lstm(request: PredictRequest):
    """
    Dự đoán sentiment bằng LSTM + trích xuất tất cả intermediate states.
    """
    if lstm_model is None:
        raise HTTPException(status_code=503, detail="LSTM model chưa được tải")
    
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text không được rỗng")
    
    # Encode (không padding cho demo)
    indices, tokens = preprocessor.encode_tokens_only(text)
    
    if not tokens:
        raise HTTPException(status_code=400, detail="Không tách được từ nào từ câu input")
    
    # Tạo tensor
    input_ids = torch.tensor([indices], dtype=torch.long)
    
    # Trích xuất states
    result = lstm_extractor.extract(input_ids, tokens)
    
    return PredictResponse(
        input_text=text,
        **result
    )


@router.post("/api/predict/gru", response_model=PredictResponse)
async def predict_gru(request: PredictRequest):
    """
    Dự đoán sentiment bằng GRU + trích xuất tất cả intermediate states.
    """
    if gru_model is None:
        raise HTTPException(status_code=503, detail="GRU model chưa được tải")
    
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text không được rỗng")
    
    # Encode (không padding cho demo)
    indices, tokens = preprocessor.encode_tokens_only(text)
    
    if not tokens:
        raise HTTPException(status_code=400, detail="Không tách được từ nào từ câu input")
    
    # Tạo tensor
    input_ids = torch.tensor([indices], dtype=torch.long)
    
    # Trích xuất states
    result = gru_extractor.extract(input_ids, tokens)
    
    return PredictResponse(
        input_text=text,
        **result
    )


@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Kiểm tra trạng thái server và models."""
    return HealthResponse(
        status="ok",
        lstm_loaded=lstm_model is not None,
        gru_loaded=gru_model is not None,
        vocab_size=len(preprocessor.vocab) if preprocessor and preprocessor.vocab else 0
    )
