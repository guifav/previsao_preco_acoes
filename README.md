# Predicao de Precos de Acoes com LSTM - PETR4.SA

**Projeto de Deep Learning aplicado a Series Temporais Financeiras**

Modelo preditivo de redes neurais LSTM (Long Short-Term Memory) para prever o preco de fechamento das acoes da Petrobras (PETR4.SA), com pipeline completa desde a coleta de dados ate o deploy em producao.

## Links do Projeto

| Recurso | Link |
|---------|------|
| Dashboard Live | [HuggingFace Spaces](https://huggingface.co/spaces/guifav/lstm-petr4-stock-prediction) |
| Modelo Treinado | [HuggingFace Hub](https://huggingface.co/guifav/lstm-petr4-stock-prediction) |
| Notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guifav/previsao_preco_acoes/blob/main/notebooks/lstm_petr4_stock_prediction.ipynb) |
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
├── ROTEIRO_VIDEO.md                   # Roteiro do video explicativo
├── Dockerfile                         # Container para deploy
├── docker-compose.yml                 # Orquestracao local
├── requirements.txt                   # Dependencias Python
├── .gitignore
├── .github/
│   └── workflows/
│       └── retrain.yml                # GitHub Action: retreino semanal
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

## Quer usar com outra acao? (VALE3, ITUB4, BBDC4...)

1. Faca um **fork** deste repositorio (botao no canto superior direito)
2. Clique no badge **"Open in Colab"** acima
3. No Colab, va em **Runtime > Change runtime type > T4 GPU**
4. Troque o ticker de `PETR4.SA` para a acao desejada (ex: `VALE3.SA`)
5. Execute todas as celulas -- o notebook e autocontido e didatico
6. Crie um token no [HuggingFace](https://huggingface.co/settings/tokens) com permissao Write
7. Rode as celulas de upload para publicar no HuggingFace Hub e criar o Space
8. Teste sua API: `curl https://SEU_USUARIO-lstm-petr4-stock-prediction.hf.space/predict/VALE3.SA`

Nao precisa instalar nada. Tudo roda no navegador, de graca. Veja o passo a passo detalhado no [ROTEIRO_VIDEO.md](ROTEIRO_VIDEO.md#passo-a-passo-como-usar-este-projeto-com-outra-acao).

## Como Usar (desenvolvimento)

### 1. Notebook (recomendado para aprendizado)

Abra o notebook no Google Colab (clique no badge acima) e execute celula por celula. Ele contem explicacoes detalhadas de cada etapa. Tambem funciona localmente em Jupyter com Python 3.11+.

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
| GET | `/model/info` | Arquitetura, treinamento e limitacoes |

#### Exemplos com curl (API em producao)

```bash
# Health check
curl https://guifav-lstm-petr4-stock-prediction.hf.space/health

# Previsao automatica para PETR4
curl https://guifav-lstm-petr4-stock-prediction.hf.space/predict/PETR4.SA

# Metricas do modelo e da API
curl https://guifav-lstm-petr4-stock-prediction.hf.space/metrics

# Informacoes detalhadas do modelo
curl https://guifav-lstm-petr4-stock-prediction.hf.space/model/info

# Previsao com dados customizados (POST)
curl -X POST https://guifav-lstm-petr4-stock-prediction.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"close_prices": [36.5, 36.8, 37.1, 36.9, 37.3, 37.0, 36.7, 37.2, 37.5, 37.8, 38.0, 37.6, 37.9, 38.2, 38.5, 38.1, 37.8, 38.3, 38.6, 38.9, 39.1, 38.7, 39.0, 39.3, 39.6, 39.2, 38.9, 39.4, 39.7, 40.0, 39.6, 39.3, 39.8, 40.1, 40.4, 40.0, 39.7, 40.2, 40.5, 40.8, 41.0, 40.6, 40.3, 40.8, 41.1, 41.4, 41.0, 40.7, 41.2, 41.5, 41.8, 41.4, 41.1, 41.6, 41.9, 42.2, 41.8, 41.5, 42.0, 42.3]}'
```

#### Exemplos locais

```bash
# Mesmos endpoints, mas usando localhost
curl http://localhost:7860/predict/PETR4.SA
```

### 5. Retreino automatico (GitHub Actions)

O modelo e retreinado automaticamente toda segunda-feira via GitHub Actions, incorporando os dados mais recentes. O workflow:

1. Baixa dados atualizados via yfinance
2. Retreina o modelo LSTM
3. Publica os novos artefatos no HuggingFace Hub
4. Atualiza o Space com o modelo atualizado

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

**Guilherme Favaron**

## Licenca

MIT
