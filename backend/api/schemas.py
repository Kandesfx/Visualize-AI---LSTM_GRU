"""
Pydantic schemas cho API request/response.
"""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class PredictRequest(BaseModel):
    text: str


class GateInfo(BaseModel):
    mean: float
    values: List[float]


class FormulaInfo(BaseModel):
    forget: Optional[str] = None
    input: Optional[str] = None
    candidate: str
    cell_update: Optional[str] = None
    output: Optional[str] = None
    hidden: str
    reset: Optional[str] = None
    update: Optional[str] = None


class CellState(BaseModel):
    timestep: int
    token: str
    embedding: List[float]
    embedding_summary: List[float]
    forget_gate: Optional[GateInfo] = None
    input_gate: Optional[GateInfo] = None
    candidate: GateInfo
    output_gate: Optional[GateInfo] = None
    cell_state: Optional[GateInfo] = None
    hidden_state: GateInfo
    reset_gate: Optional[GateInfo] = None
    update_gate: Optional[GateInfo] = None
    formulas: Dict[str, str]


class EmbeddingInfo(BaseModel):
    token: str
    vector: List[float]
    summary: List[float]
    dim: int
    mean: float
    min: float
    max: float


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]


class FinalLayer(BaseModel):
    logits: List[float]
    softmax: List[float]


class PredictResponse(BaseModel):
    input_text: str
    tokens: List[str]
    prediction: Prediction
    embeddings: List[EmbeddingInfo]
    cell_states: List[CellState]
    final_layer: FinalLayer


class HealthResponse(BaseModel):
    status: str
    lstm_loaded: bool
    gru_loaded: bool
    vocab_size: int
