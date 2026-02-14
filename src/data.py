"""
Modulo de coleta e pre-processamento de dados financeiros.

Funcoes para baixar dados via yfinance, normalizar,
criar sequencias temporais e preparar DataLoaders.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def fetch_stock_data(symbol: str, start_date: str = "2018-01-01",
                     end_date: str = None) -> pd.DataFrame:
    """
    Baixa dados historicos de uma acao via Yahoo Finance.

    Args:
        symbol: Ticker da acao (ex: 'PETR4.SA')
        start_date: Data inicial no formato 'YYYY-MM-DD'
        end_date: Data final (default: hoje)

    Returns:
        DataFrame com colunas Open, High, Low, Close, Volume
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    df = yf.download(symbol, start=start_date, end=end_date)

    # Flatten MultiIndex se presente (yfinance >= 0.2.31)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise ValueError(f"Nenhum dado retornado para {symbol}")

    return df


def preprocess_data(df: pd.DataFrame, feature_columns: list = None,
                    scaler: MinMaxScaler = None):
    """
    Normaliza os dados usando MinMaxScaler.

    Args:
        df: DataFrame com dados da acao
        feature_columns: Lista de colunas a usar (default: ['Close'])
        scaler: Scaler pre-ajustado (para inferencia). Se None, cria um novo.

    Returns:
        data_scaled: Array normalizado
        scaler: O scaler usado (para inverse_transform depois)
    """
    if feature_columns is None:
        feature_columns = ["Close"]

    data = df[feature_columns].dropna().values

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)

    return data_scaled, scaler


def create_sequences(data: np.ndarray, sequence_length: int = 60):
    """
    Cria janelas deslizantes (sliding windows) para a LSTM.

    Args:
        data: Array normalizado de shape (n_amostras, n_features)
        sequence_length: Tamanho da janela de lookback

    Returns:
        X: Array de shape (n_sequencias, sequence_length, n_features)
        y: Array de shape (n_sequencias,)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)


def get_latest_sequence(symbol: str, sequence_length: int,
                        scaler: MinMaxScaler) -> np.ndarray:
    """
    Obtem a sequencia mais recente de dados para fazer uma previsao.
    Usado pela API em tempo real.

    Args:
        symbol: Ticker da acao
        sequence_length: Tamanho da janela
        scaler: Scaler pre-ajustado

    Returns:
        Array de shape (1, sequence_length, 1) pronto para o modelo
    """
    # Baixar dados recentes (margem extra para garantir dias uteis suficientes)
    days_to_fetch = sequence_length * 2
    end_date = datetime.now().strftime("%Y-%m-%d")

    df = fetch_stock_data(symbol, end_date=end_date,
                          start_date=pd.Timestamp(end_date) - pd.Timedelta(days=days_to_fetch))

    close_data = df[["Close"]].dropna().values

    if len(close_data) < sequence_length:
        raise ValueError(
            f"Dados insuficientes: {len(close_data)} dias disponiveis, "
            f"{sequence_length} necessarios"
        )

    # Pegar os ultimos sequence_length dias
    latest = close_data[-sequence_length:]

    # Normalizar com o mesmo scaler usado no treino
    latest_scaled = scaler.transform(latest)

    # Reshape para (1, sequence_length, n_features)
    return latest_scaled.reshape(1, sequence_length, -1)
