---
license: mit
tags:
  - pytorch
  - lstm
  - time-series
  - stock-prediction
  - finance
  - brazilian-stocks
language:
  - pt
pipeline_tag: time-series-forecasting
---

# LSTM Stock Price Predictor - PETR4.SA (Petrobras)

Modelo LSTM (Long Short-Term Memory) treinado em PyTorch para prever o preco de fechamento das acoes da Petrobras (PETR4.SA) na bolsa brasileira (B3).

## Descricao

Este modelo utiliza dados historicos de precos para capturar padroes temporais e gerar previsoes do proximo preco de fechamento. Desenvolvido como projeto de estudo em Deep Learning aplicado a series temporais financeiras.

## Arquitetura

- **Tipo:** LSTM (2 camadas empilhadas)
- **Hidden size:** 128
- **Dropout:** 0.2
- **Janela temporal:** 60 dias
- **Feature:** Preco de fechamento (Close)
- **Framework:** PyTorch

## Metricas (Conjunto de Teste)

| Metrica | Valor |
|---------|-------|
| MAE | Calculado apos treino |
| RMSE | Calculado apos treino |
| MAPE | Calculado apos treino |

## Como Usar

```python
import torch
import joblib
import json
import numpy as np

# Carregar artefatos
config = json.load(open("config.json"))
scaler = joblib.load("scaler.joblib")

# Reconstruir modelo
from model import LSTMModel

model = LSTMModel(
    input_size=config["model"]["input_size"],
    hidden_size=config["model"]["hidden_size"],
    num_layers=config["model"]["num_layers"],
    dropout=config["model"]["dropout"]
)
model.load_state_dict(torch.load("lstm_model.pth", map_location="cpu", weights_only=True))
model.eval()

# Fazer previsao (com 60 precos de fechamento)
prices = np.array([...]).reshape(-1, 1)  # 60 precos
prices_scaled = scaler.transform(prices)
input_tensor = torch.FloatTensor(prices_scaled.reshape(1, 60, 1))

with torch.no_grad():
    prediction = model(input_tensor).item()

predicted_price = scaler.inverse_transform([[prediction]])[0][0]
print(f"Preco previsto: R$ {predicted_price:.2f}")
```

## Arquivos

- `lstm_model.pth` - Pesos do modelo treinado (state_dict)
- `scaler.joblib` - MinMaxScaler ajustado nos dados de treino
- `config.json` - Hiperparametros, metricas e metadados
- `model.py` - Definicao da classe LSTMModel

## Dados de Treinamento

- **Fonte:** Yahoo Finance via biblioteca yfinance
- **Acao:** PETR4.SA (Petrobras)
- **Periodo:** Janeiro 2018 ate a data de treino
- **Split:** 70% treino / 15% validacao / 15% teste (cronologico)

## Limitacoes

- Este modelo e **educacional** e nao deve ser usado para decisoes financeiras reais
- Treinado apenas com dados de preco de fechamento -- nao considera fatores fundamentalistas, noticias ou sentimento de mercado
- Performance pode degradar em periodos de alta volatilidade ou eventos atipicos (ex: crises)
- Previsao limitada a 1 dia a frente

## Autor

**Guilherme de Mauro Favaron**

## Licenca

MIT
