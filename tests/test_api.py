"""
Testes basicos para a API.
Executar: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def client():
    """Cria um client de teste para a API."""
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    """Testa se o endpoint /health retorna status 200."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_metrics(client):
    """Testa se o endpoint /metrics retorna dados."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "model_metrics" in data
    assert "api_metrics" in data


def test_predict_symbol(client):
    """Testa previsao com ticker padrao."""
    response = client.get("/predict/PETR4.SA")
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 0


def test_predict_from_data(client):
    """Testa previsao com dados fornecidos."""
    # 60 precos ficticios
    prices = [35.0 + i * 0.1 for i in range(60)]
    response = client.post("/predict", json={"close_prices": prices})
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data


def test_predict_insufficient_data(client):
    """Testa erro quando dados insuficientes."""
    response = client.post("/predict", json={"close_prices": [35.0] * 10})
    assert response.status_code == 422  # Validation error
