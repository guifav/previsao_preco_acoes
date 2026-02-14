# Predicao de Precos de Acoes com LSTM - PETR4.SA

**Projeto de Deep Learning aplicado a Series Temporais Financeiras**

Modelo preditivo de redes neurais LSTM (Long Short-Term Memory) para prever o preco de fechamento das acoes da Petrobras (PETR4.SA), com pipeline completa desde a coleta de dados ate o deploy em producao.

## Links do Projeto

| Recurso | Link |
|---------|------|
| Dashboard Live | [HuggingFace Spaces](https://huggingface.co/spaces/guifav/lstm-petr4-stock-prediction) |
| Modelo Treinado | [HuggingFace Hub](https://huggingface.co/guifav/lstm-petr4-stock-prediction) |
| Notebook | [Ver notebook](notebooks/lstm_petr4_stock_prediction.ipynb) |
| Video Explicativo | [YouTube](https://youtube.com) |

## Arquitetura

```
                     +-------------------+
                     |   Yahoo Finance   |
                     |   (yfinance)      |
                     +--------+----------+
                              |
                              v
                     +--------+----------+
                     |   Pre-processamento|
                     |   MinMaxScaler     |
                     |   Sliding Windows  |
                     +--------+----------+
                              |
                              v
                     +--------+----------+
                     |   Modelo LSTM      |
                     |   PyTorch          |
                     |   2 camadas, 128h  |
                     +--------+----------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
    +---------+---------+           +---------+---------+
    |  HuggingFace Hub  |           |  HuggingFace      |
    |  (modelo salvo)   |           |  Spaces           |
    +-------------------+           |  FastAPI + Gradio  |
                                    +-------------------+
```

## Estrutura do Projeto

```
previsao_preco_acoes/
├── README.md                          # Este arquivo
├── Dockerfile                         # Container para deploy
├── docker-compose.yml                 # Orquestracao local
├── requirements.txt                   # Dependencias Python
├── .gitignore
├── notebooks/
│   └── lstm_petr4_stock_prediction.ipynb  # Notebook didatico (Colab-ready)
├── src/
│   ├── __init__.py
│   ├── model.py                       # Definicao da classe LSTM
│   ├── data.py                        # Coleta e pre-processamento
│   ├── train.py                       # Script de treinamento
│   └── predict.py                     # Logica de inferencia
├── api/
│   ├── __init__.py
│   ├── main.py                        # FastAPI + Gradio dashboard
│   └── schemas.py                     # Schemas Pydantic
├── models/                            # Artefatos do modelo treinado
│   ├── lstm_model.pth
│   ├── scaler.joblib
│   └── config.json
├── huggingface/
│   ├── hub/README.md                  # Model card para HF Hub
│   └── spaces/README.md               # Config do HF Spaces
└── tests/
    └── test_api.py                    # Testes da API
```

## Como Usar

### 1. Notebook (recomendado para aprendizado)

Abra o notebook em Jupyter (local ou Colab) e execute celula por celula. Ele contem explicacoes detalhadas de cada etapa.

### 2. Treinamento via script

```bash
# Instalar dependencias
pip install -r requirements.txt

# Treinar o modelo
python -m src.train --symbol PETR4.SA --epochs 100

# Os artefatos serao salvos em models/
```

### 3. Rodar a API localmente

```bash
# Com Python
uvicorn api.main:app --host 0.0.0.0 --port 7860

# Com Docker
docker-compose up --build

# Acessar:
# Dashboard: http://localhost:7860
# API Docs:  http://localhost:7860/docs
```

### 4. Endpoints da API

| Metodo | Endpoint | Descricao |
|--------|----------|-----------|
| GET | `/health` | Status da API e modelo |
| GET | `/predict/{symbol}` | Previsao automatica (baixa dados recentes) |
| POST | `/predict` | Previsao com dados fornecidos pelo usuario |
| GET | `/metrics` | Metricas do modelo e da API |

Exemplo de requisicao:
```bash
curl http://localhost:7860/predict/PETR4.SA
```

## Tecnologias

- **Python 3.11**
- **PyTorch** - Framework de deep learning
- **FastAPI** - API REST
- **Gradio** - Dashboard interativo
- **yfinance** - Coleta de dados financeiros
- **scikit-learn** - Pre-processamento (MinMaxScaler)
- **Plotly** - Graficos interativos
- **Docker** - Containerizacao

## Metricas do Modelo

As metricas sao calculadas no conjunto de teste (15% dos dados, os mais recentes):

| Metrica | Descricao |
|---------|-----------|
| MAE | Erro absoluto medio em BRL |
| RMSE | Raiz do erro quadratico medio em BRL |
| MAPE | Erro percentual absoluto medio |

Valores especificos sao gerados apos o treinamento e ficam registrados em `models/config.json`.

## Limitacoes

- Modelo educacional, **nao use para decisoes financeiras reais**
- Treinado apenas com preco de fechamento historico
- Nao considera fatores externos (noticias, macroeconomia, sentimento)
- Previsao limitada a 1 dia a frente
- Performance pode degradar em cenarios atipicos

## Autor

**Guilherme de Mauro Favaron**

## Licenca

MIT
