"""
Script de treinamento standalone.
Pode ser executado diretamente para treinar o modelo sem usar o notebook.

Uso:
    python -m src.train --symbol PETR4.SA --epochs 100
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

from src.model import LSTMModel
from src.data import fetch_stock_data, preprocess_data, create_sequences


def train(
    symbol: str = "PETR4.SA",
    start_date: str = "2018-01-01",
    sequence_length: int = 60,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    num_epochs: int = 100,
    patience: int = 15,
    output_dir: str = "models"
):
    """Treina o modelo LSTM e salva os artefatos."""

    # Detectar melhor dispositivo disponivel: CUDA (NVIDIA), MPS (Apple Silicon) ou CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Dispositivo: {device}")

    # Seed
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Coleta de dados
    print(f"\nBaixando dados de {symbol}...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    df = fetch_stock_data(symbol, start_date, end_date)
    print(f"Registros: {len(df)}")

    # 2. Pre-processamento
    feature_columns = ["Close"]
    data_scaled, scaler = preprocess_data(df, feature_columns)
    X, y = create_sequences(data_scaled, sequence_length)

    # 3. Split temporal
    n_total = len(X)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    print(f"Treino: {len(X_train)} | Validacao: {len(X_val)} | Teste: {len(X_test)}")

    # Tensores
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size, shuffle=False
    )

    # 4. Modelo
    model = LSTMModel(
        input_size=len(feature_columns),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 5. Treinamento
    print("\nIniciando treinamento...")
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            pred = model(X_b).squeeze()
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                pred = model(X_b).squeeze()
                val_losses.append(criterion(pred, y_b).item())

        avg_val = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict().copy()
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        if (epoch + 1) % 10 == 0 or marker:
            print(f"Epoca {epoch+1:3d}/{num_epochs} | "
                  f"Train: {avg_train:.6f} | Val: {avg_val:.6f}{marker}")

        if no_improve >= patience:
            print(f"\nEarly stopping na epoca {epoch+1}")
            break

    model.load_state_dict(best_state)
    print(f"Treinamento concluido em {time.time()-start_time:.1f}s")

    # 6. Avaliacao
    def calc_metrics(X_t, y_t):
        model.eval()
        with torch.no_grad():
            preds = model(X_t).cpu().numpy().flatten()
        y_real = scaler.inverse_transform(y_t.cpu().numpy().reshape(-1, 1)).flatten()
        p_real = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        return {
            "MAE": round(float(mean_absolute_error(y_real, p_real)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_real, p_real))), 4),
            "MAPE": round(float(np.mean(np.abs((y_real - p_real) / y_real)) * 100), 4)
        }

    test_metrics = calc_metrics(X_test_t, y_test_t)
    val_metrics = calc_metrics(X_val_t, y_val_t)
    train_metrics = calc_metrics(X_train_t, y_train_t)

    print(f"\nMetricas (Teste): MAE=R${test_metrics['MAE']:.2f} | "
          f"RMSE=R${test_metrics['RMSE']:.2f} | MAPE={test_metrics['MAPE']:.2f}%")

    # 7. Salvar
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "lstm_model.pth"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

    config = {
        "model": {
            "input_size": len(feature_columns),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "sequence_length": sequence_length
        },
        "training": {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "feature_columns": feature_columns,
            "target_column": "Close",
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs_run": len(history["train_loss"]),
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15
        },
        "metrics": {
            "test": test_metrics,
            "validation": val_metrics,
            "train": train_metrics
        },
        "data_info": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "scaler_min": float(scaler.data_min_[0]),
            "scaler_max": float(scaler.data_max_[0])
        },
        "exported_at": datetime.now().isoformat()
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nArtefatos salvos em {output_dir}/")
    return model, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar modelo LSTM")
    parser.add_argument("--symbol", default="PETR4.SA")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    train(
        symbol=args.symbol,
        num_epochs=args.epochs,
        hidden_size=args.hidden_size,
        sequence_length=args.sequence_length,
        output_dir=args.output_dir
    )
