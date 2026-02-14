"""
Modulo de inferencia: carrega o modelo treinado e faz previsoes.
"""

import json
import torch
import joblib
import numpy as np
from pathlib import Path

from src.model import LSTMModel
from src.data import get_latest_sequence, fetch_stock_data


class StockPredictor:
    """
    Carrega um modelo LSTM treinado e faz previsoes de precos de acoes.

    Uso:
        predictor = StockPredictor("models/")
        result = predictor.predict("PETR4.SA")
        print(result['predicted_price'])
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        # Detectar melhor dispositivo: CUDA (NVIDIA), MPS (Apple Silicon) ou CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Carregar configuracao
        config_path = self.models_dir / "config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Carregar scaler
        scaler_path = self.models_dir / "scaler.joblib"
        self.scaler = joblib.load(scaler_path)

        # Carregar modelo
        model_cfg = self.config["model"]
        self.model = LSTMModel(
            input_size=model_cfg["input_size"],
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"]
        ).to(self.device)

        model_path = self.models_dir / "lstm_model.pth"
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.sequence_length = model_cfg["sequence_length"]

    def predict(self, symbol: str = None) -> dict:
        """
        Faz uma previsao do proximo preco de fechamento.

        Args:
            symbol: Ticker da acao (default: o mesmo usado no treino)

        Returns:
            dict com predicted_price, last_close, symbol, etc.
        """
        if symbol is None:
            symbol = self.config["training"]["symbol"]

        # Obter dados recentes
        sequence = get_latest_sequence(symbol, self.sequence_length, self.scaler)

        # Converter para tensor
        input_tensor = torch.FloatTensor(sequence).to(self.device)

        # Inferencia
        with torch.no_grad():
            prediction_scaled = self.model(input_tensor).cpu().numpy().flatten()[0]

        # Desnormalizar
        predicted_price = self.scaler.inverse_transform(
            [[prediction_scaled]]
        )[0][0]

        # Obter ultimo preco real para referencia
        df = fetch_stock_data(symbol)
        last_close = float(df["Close"].iloc[-1])
        last_date = str(df.index[-1].date())

        change = predicted_price - last_close
        change_pct = (change / last_close) * 100

        return {
            "symbol": symbol,
            "predicted_price": round(float(predicted_price), 2),
            "last_close": round(last_close, 2),
            "last_date": last_date,
            "change": round(float(change), 2),
            "change_pct": round(float(change_pct), 2),
            "model_metrics": self.config.get("metrics", {}).get("test", {}),
            "sequence_length": self.sequence_length
        }

    def predict_from_data(self, close_prices: list) -> dict:
        """
        Faz previsao a partir de uma lista de precos fornecida pelo usuario.

        Args:
            close_prices: Lista com pelo menos sequence_length precos de fechamento

        Returns:
            dict com predicted_price
        """
        if len(close_prices) < self.sequence_length:
            raise ValueError(
                f"Necessarios pelo menos {self.sequence_length} precos, "
                f"recebidos {len(close_prices)}"
            )

        # Pegar os ultimos sequence_length precos
        prices = np.array(close_prices[-self.sequence_length:]).reshape(-1, 1)

        # Normalizar
        prices_scaled = self.scaler.transform(prices)

        # Reshape para (1, seq_len, 1)
        input_data = prices_scaled.reshape(1, self.sequence_length, -1)
        input_tensor = torch.FloatTensor(input_data).to(self.device)

        # Inferencia
        with torch.no_grad():
            prediction_scaled = self.model(input_tensor).cpu().numpy().flatten()[0]

        predicted_price = self.scaler.inverse_transform(
            [[prediction_scaled]]
        )[0][0]

        last_price = close_prices[-1]
        change = predicted_price - last_price
        change_pct = (change / last_price) * 100

        return {
            "predicted_price": round(float(predicted_price), 2),
            "last_provided_price": round(float(last_price), 2),
            "change": round(float(change), 2),
            "change_pct": round(float(change_pct), 2),
            "prices_used": self.sequence_length
        }
