"""
Schemas Pydantic para a API FastAPI.
Definem o formato das requisicoes e respostas.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """Requisicao de previsao com dados fornecidos pelo usuario."""
    close_prices: list[float] = Field(
        ...,
        description="Lista de precos de fechamento (minimo 60 valores)",
        min_length=60
    )

    class Config:
        json_schema_extra = {
            "example": {
                "close_prices": [36.5, 36.8, 37.1, 36.9, 37.3] + [37.0] * 55
            }
        }


class PredictionResponse(BaseModel):
    """Resposta com a previsao do modelo."""
    symbol: Optional[str] = None
    predicted_price: float = Field(..., description="Preco previsto (BRL)")
    last_close: Optional[float] = Field(None, description="Ultimo preco real (BRL)")
    last_date: Optional[str] = Field(None, description="Data do ultimo preco")
    change: float = Field(..., description="Variacao em BRL")
    change_pct: float = Field(..., description="Variacao em %")
    model_metrics: Optional[dict] = None
    sequence_length: Optional[int] = None


class HealthResponse(BaseModel):
    """Resposta do healthcheck."""
    status: str
    model_loaded: bool
    device: str
    symbol: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Metricas de performance do modelo e da API."""
    model_metrics: dict
    api_metrics: dict
