"""
API FastAPI + Dashboard Gradio para predicao de precos de acoes com LSTM.

Este modulo cria:
1. Uma API RESTful com endpoints /predict, /health, /metrics
2. Um dashboard interativo Gradio com graficos e previsao em tempo real

Uso local:
    uvicorn api.main:app --host 0.0.0.0 --port 7860

HuggingFace Spaces:
    O Spaces executa este arquivo automaticamente.
"""

import os
import sys
import time
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from collections import deque

import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Adicionar o diretorio raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import StockPredictor
from src.data import fetch_stock_data
from api.schemas import (
    PredictionRequest, PredictionResponse,
    HealthResponse, MetricsResponse
)

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# === ESTADO GLOBAL ===
predictor = None
start_time = None
request_log = deque(maxlen=1000)  # Ultimas 1000 requisicoes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo na inicializacao."""
    global predictor, start_time
    start_time = time.time()

    models_dir = os.environ.get("MODELS_DIR", "models")
    logger.info(f"Carregando modelo de {models_dir}...")

    try:
        predictor = StockPredictor(models_dir)
        logger.info(f"Modelo carregado com sucesso! Device: {predictor.device}")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise

    yield
    logger.info("Encerrando aplicacao...")


# === FASTAPI ===
app = FastAPI(
    title="LSTM Stock Predictor - PETR4.SA",
    description=(
        "API para predicao do preco de fechamento de acoes "
        "usando modelo LSTM em PyTorch. "
        "Projeto de Deep Learning aplicado a Series Temporais Financeiras."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


def log_request(endpoint: str, duration: float, success: bool):
    """Registra metricas de cada requisicao."""
    request_log.append({
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "duration_ms": round(duration * 1000, 2),
        "success": success
    })


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health():
    """Verifica se a API e o modelo estao funcionando."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        device=str(predictor.device) if predictor else "unknown",
        symbol=predictor.config["training"]["symbol"] if predictor else "unknown",
        uptime_seconds=round(time.time() - start_time, 1)
    )


@app.get("/predict/{symbol}", response_model=PredictionResponse)
async def predict_symbol(symbol: str = "PETR4.SA"):
    """
    Preve o proximo preco de fechamento para o ticker informado.
    Baixa automaticamente os dados mais recentes.
    """
    t0 = time.time()
    try:
        result = predictor.predict(symbol)
        log_request(f"/predict/{symbol}", time.time() - t0, True)
        return PredictionResponse(**result)
    except Exception as e:
        log_request(f"/predict/{symbol}", time.time() - t0, False)
        logger.error(f"Erro na previsao para {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict_from_data(request: PredictionRequest):
    """
    Preve o proximo preco a partir de uma lista de precos fornecida.
    Util para testar com dados customizados.
    """
    t0 = time.time()
    try:
        result = predictor.predict_from_data(request.close_prices)
        log_request("/predict", time.time() - t0, True)
        return PredictionResponse(**result)
    except Exception as e:
        log_request("/predict", time.time() - t0, False)
        logger.error(f"Erro na previsao com dados fornecidos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Retorna metricas do modelo e da API."""
    recent = list(request_log)
    successful = [r for r in recent if r["success"]]
    failed = [r for r in recent if not r["success"]]

    avg_latency = (
        np.mean([r["duration_ms"] for r in successful])
        if successful else 0
    )

    return MetricsResponse(
        model_metrics=predictor.config.get("metrics", {}),
        api_metrics={
            "total_requests": len(recent),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "avg_latency_ms": round(avg_latency, 2),
            "uptime_seconds": round(time.time() - start_time, 1)
        }
    )


@app.get("/model/info")
async def model_info():
    """Retorna detalhes sobre arquitetura, treinamento e limitacoes do modelo."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo nao carregado.")

    cfg = predictor.config
    m = cfg.get("metrics", {}).get("test", {})
    t = cfg.get("training", {})
    d = cfg.get("data_info", {})

    return {
        "symbol": t.get("symbol"),
        "architecture": {
            "type": "LSTM (Long Short-Term Memory)",
            "num_layers": cfg["model"]["num_layers"],
            "hidden_size": cfg["model"]["hidden_size"],
            "dropout": cfg["model"]["dropout"],
            "sequence_length": cfg["model"]["sequence_length"],
            "framework": "PyTorch"
        },
        "training": {
            "period": f"{t.get('start_date')} a {t.get('end_date')}",
            "total_samples": d.get("total_samples"),
            "optimizer": f"Adam (lr={t.get('learning_rate')})",
            "batch_size": t.get("batch_size"),
            "epochs_run": t.get("num_epochs_run"),
            "split": {
                "train": t.get("train_ratio"),
                "val": t.get("val_ratio"),
                "test": t.get("test_ratio")
            }
        },
        "test_metrics": {
            "MAE": m.get("MAE"),
            "RMSE": m.get("RMSE"),
            "MAPE": m.get("MAPE")
        },
        "metrics_explanation": {
            "MAE": "Erro absoluto medio em R$. Exemplo: MAE=1.50 significa que o modelo erra em media R$1.50.",
            "RMSE": "Raiz do erro quadratico medio. Penaliza erros grandes. Se RMSE >> MAE, ha outliers.",
            "MAPE": "Erro percentual medio. MAPE=3% significa que a previsao erra em media 3% do valor real."
        },
        "limitations": "Modelo treinado exclusivamente com PETR4.SA. Previsoes para outros tickers usam o mesmo scaler e pesos, portanto nao sao confiaveis.",
        "exported_at": cfg.get("exported_at")
    }


# === GRADIO DASHBOARD ===

def create_prediction_plot(symbol: str):
    """Gera grafico de previsao para o dashboard."""
    try:
        # Dados historicos
        df = fetch_stock_data(symbol)

        # Previsao
        result = predictor.predict(symbol)

        # Grafico
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                f"Historico de Precos - {symbol}",
                "Volume de Negociacao"
            )
        )

        # Ultimos 6 meses para visualizacao
        df_recent = df.tail(180)

        fig.add_trace(
            go.Scatter(
                x=df_recent.index, y=df_recent["Close"],
                mode="lines", name="Preco Fechamento",
                line=dict(color="#1f77b4", width=2)
            ),
            row=1, col=1
        )

        # Media movel
        ma20 = df_recent["Close"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=df_recent.index, y=ma20,
                mode="lines", name="MM 20 dias",
                line=dict(color="orange", width=1, dash="dash")
            ),
            row=1, col=1
        )

        # Ponto da previsao
        next_date = pd.Timestamp(result["last_date"]) + pd.Timedelta(days=1)
        # Ajustar para dia util
        if next_date.weekday() == 5:
            next_date += pd.Timedelta(days=2)
        elif next_date.weekday() == 6:
            next_date += pd.Timedelta(days=1)

        color = "green" if result["change"] >= 0 else "red"
        fig.add_trace(
            go.Scatter(
                x=[next_date],
                y=[result["predicted_price"]],
                mode="markers+text",
                name=f"Previsao: R${result['predicted_price']:.2f}",
                marker=dict(size=14, color=color, symbol="star"),
                text=[f"R${result['predicted_price']:.2f}"],
                textposition="top center",
                textfont=dict(size=12, color=color)
            ),
            row=1, col=1
        )

        # Volume
        fig.add_trace(
            go.Bar(
                x=df_recent.index, y=df_recent["Volume"],
                name="Volume",
                marker_color="rgba(31, 119, 180, 0.3)"
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=600,
            showlegend=True,
            yaxis_title="Preco (BRL)",
            yaxis2_title="Volume"
        )

        # Resumo textual
        direction = "ALTA" if result["change"] >= 0 else "BAIXA"
        summary = (
            f"Previsao para {symbol}:\n\n"
            f"Ultimo fechamento ({result['last_date']}): R$ {result['last_close']:.2f}\n"
            f"Previsao proximo dia: R$ {result['predicted_price']:.2f}\n"
            f"Variacao esperada: R$ {result['change']:+.2f} ({result['change_pct']:+.2f}%) - {direction}\n\n"
            f"Metricas do modelo (teste):\n"
            f"  MAE:  R$ {result['model_metrics'].get('MAE', 'N/A')}\n"
            f"  RMSE: R$ {result['model_metrics'].get('RMSE', 'N/A')}\n"
            f"  MAPE: {result['model_metrics'].get('MAPE', 'N/A')}%\n\n"
            f"AVISO: Este modelo e educacional. Nao use para decisoes financeiras reais."
        )

        return fig, summary

    except Exception as e:
        logger.error(f"Erro no dashboard: {e}")
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Erro ao carregar dados")
        return empty_fig, f"Erro: {str(e)}"


def get_model_info():
    """Retorna informacoes sobre o modelo para a aba 'Sobre'."""
    if predictor is None:
        return "Modelo nao carregado."

    cfg = predictor.config
    m = cfg.get("metrics", {}).get("test", {})
    t = cfg.get("training", {})
    d = cfg.get("data_info", {})

    info = (
        f"MODELO LSTM - PREDICAO DE PRECOS DE ACOES\n"
        f"{'='*50}\n\n"
        f"Acao treinada: {t.get('symbol', 'N/A')}\n"
        f"Periodo dos dados: {t.get('start_date')} a {t.get('end_date')}\n"
        f"Total de amostras: {d.get('total_samples', 'N/A')}\n\n"
        f"IMPORTANTE: este modelo foi treinado exclusivamente com dados\n"
        f"da {t.get('symbol', 'N/A')}. Previsoes para outros tickers usam\n"
        f"o mesmo scaler e pesos, portanto nao sao confiaveis.\n\n"
        f"ARQUITETURA\n"
        f"{'-'*30}\n"
        f"Tipo: LSTM (Long Short-Term Memory)\n"
        f"Camadas LSTM: {cfg['model']['num_layers']}\n"
        f"Hidden size: {cfg['model']['hidden_size']}\n"
        f"Dropout: {cfg['model']['dropout']}\n"
        f"Janela temporal: {cfg['model']['sequence_length']} dias\n\n"
        f"METRICAS (Conjunto de Teste)\n"
        f"{'-'*30}\n"
        f"MAE:  R$ {m.get('MAE', 'N/A')}\n"
        f"RMSE: R$ {m.get('RMSE', 'N/A')}\n"
        f"MAPE: {m.get('MAPE', 'N/A')}%\n\n"
        f"O QUE SIGNIFICAM ESSAS METRICAS?\n"
        f"{'-'*30}\n"
        f"MAE (Mean Absolute Error / Erro Absoluto Medio):\n"
        f"  Media das diferencas absolutas entre previsao e valor real.\n"
        f"  Exemplo: MAE = 1.50 significa que o modelo erra, em media,\n"
        f"  R$ 1.50 para cima ou para baixo.\n\n"
        f"RMSE (Root Mean Square Error / Raiz do Erro Quadratico Medio):\n"
        f"  Similar ao MAE, mas penaliza mais os erros grandes.\n"
        f"  Se o RMSE e muito maior que o MAE, o modelo tem alguns\n"
        f"  erros grandes pontuais (outliers).\n\n"
        f"MAPE (Mean Absolute Percentage Error / Erro Percentual Medio):\n"
        f"  Erro medio em termos percentuais. E a metrica mais\n"
        f"  intuitiva: MAPE = 3% significa que a previsao erra,\n"
        f"  em media, 3%% do valor real.\n\n"
        f"TREINAMENTO\n"
        f"{'-'*30}\n"
        f"Framework: PyTorch\n"
        f"Otimizador: Adam (lr={t.get('learning_rate')})\n"
        f"Batch size: {t.get('batch_size')}\n"
        f"Epocas executadas: {t.get('num_epochs_run')}\n"
        f"Split: {t.get('train_ratio', 0)*100:.0f}% treino / "
        f"{t.get('val_ratio', 0)*100:.0f}% val / "
        f"{t.get('test_ratio', 0)*100:.0f}% teste\n\n"
        f"Exportado em: {cfg.get('exported_at', 'N/A')}"
    )
    return info


# Construir interface Gradio
with gr.Blocks(
    title="LSTM Stock Predictor - PETR4.SA",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # Predicao de Precos de Acoes com LSTM
        ### Projeto de Deep Learning aplicado a Series Temporais Financeiras

        Este dashboard utiliza um modelo LSTM (Long Short-Term Memory) treinado em PyTorch
        para prever o preco de fechamento de acoes da bolsa brasileira.
        """
    )

    with gr.Tabs():
        # Tab 1: Previsao
        with gr.TabItem("Previsao em Tempo Real"):
            gr.Markdown(
                "> **Nota:** O modelo foi treinado com dados da **PETR4.SA**. "
                "Previsoes para outros tickers utilizam o mesmo modelo e scaler, "
                "portanto sao apenas ilustrativas e nao devem ser consideradas confiaveis."
            )
            with gr.Row():
                symbol_input = gr.Textbox(
                    value="PETR4.SA",
                    label="Ticker da Acao",
                    info="Modelo treinado com PETR4.SA. Outros tickers sao apenas ilustrativos."
                )
                predict_btn = gr.Button("Gerar Previsao", variant="primary")

            plot_output = gr.Plot(label="Grafico de Precos e Previsao")
            text_output = gr.Textbox(
                label="Resumo da Previsao",
                lines=12,
                interactive=False
            )

            predict_btn.click(
                fn=create_prediction_plot,
                inputs=[symbol_input],
                outputs=[plot_output, text_output]
            )

        # Tab 2: Sobre o Modelo
        with gr.TabItem("Sobre o Modelo"):
            model_info = gr.Textbox(
                value=get_model_info,
                label="Informacoes do Modelo",
                lines=30,
                interactive=False
            )

        # Tab 3: API
        with gr.TabItem("Documentacao da API"):
            gr.Markdown(
                """
                ## Endpoints da API REST

                A API FastAPI roda em paralelo com este dashboard.
                Acesse [`/docs`](https://guifav-lstm-petr4-stock-prediction.hf.space/docs) para a documentacao interativa (Swagger).

                A URL base da API e:
                ```
                https://guifav-lstm-petr4-stock-prediction.hf.space
                ```

                ---

                ### 1. Health Check

                Verifica se a API esta funcionando.

                ```bash
                curl https://guifav-lstm-petr4-stock-prediction.hf.space/health
                ```

                **Resposta esperada:**
                ```json
                {"status": "healthy", "model_loaded": true}
                ```

                ---

                ### 2. Previsao por Ticker

                Preve o proximo preco de fechamento para uma acao.
                O modelo baixa automaticamente os ultimos 60 dias de dados via yfinance.

                ```bash
                curl https://guifav-lstm-petr4-stock-prediction.hf.space/predict/PETR4.SA
                ```

                **Resposta esperada:**
                ```json
                {
                    "symbol": "PETR4.SA",
                    "predicted_close": 38.42,
                    "last_close": 37.95,
                    "change_percent": 1.24,
                    "timestamp": "2026-02-14T12:00:00"
                }
                ```

                > **Nota:** O modelo foi treinado com dados de PETR4.SA.
                > Previsoes para outros tickers utilizam o mesmo modelo e scaler,
                > portanto sao apenas ilustrativas.

                ---

                ### 3. Previsao com Dados Proprios (POST)

                Envia uma lista de precos de fechamento (minimo 60 valores)
                e recebe a previsao do proximo dia.

                ```bash
                curl -X POST https://guifav-lstm-petr4-stock-prediction.hf.space/predict \\
                  -H "Content-Type: application/json" \\
                  -d '{"close_prices": [36.5, 36.8, 37.1, 36.9, 37.3, 37.0, 36.7, 37.2, 37.5, 37.8, 37.4, 37.6, 37.9, 38.1, 37.7, 38.0, 38.3, 38.5, 38.2, 38.4, 38.7, 38.9, 38.6, 38.8, 39.1, 39.3, 39.0, 39.2, 39.5, 39.7, 39.4, 39.6, 39.9, 40.1, 39.8, 40.0, 40.3, 40.5, 40.2, 40.4, 40.7, 40.9, 40.6, 40.8, 41.1, 41.3, 41.0, 41.2, 41.5, 41.7, 41.4, 41.6, 41.9, 42.1, 41.8, 42.0, 42.3, 42.5, 42.2, 42.4]}'
                ```

                **Resposta esperada:**
                ```json
                {
                    "predicted_close": 42.58,
                    "input_length": 60
                }
                ```

                ---

                ### 4. Metricas do Modelo

                Retorna metricas de avaliacao do modelo e estatisticas de uso da API.

                ```bash
                curl https://guifav-lstm-petr4-stock-prediction.hf.space/metrics
                ```

                **Resposta esperada:**
                ```json
                {
                    "model_metrics": {
                        "mae": 0.8234,
                        "rmse": 1.0456,
                        "mape": 2.15
                    },
                    "api_stats": {
                        "total_requests": 142,
                        "avg_latency_ms": 85.3
                    }
                }
                ```

                ---

                ### 5. Informacoes do Modelo

                Retorna detalhes sobre a arquitetura, treinamento e limitacoes.

                ```bash
                curl https://guifav-lstm-petr4-stock-prediction.hf.space/model/info
                ```
                """
            )

# Montar Gradio no FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

# Para rodar localmente:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
