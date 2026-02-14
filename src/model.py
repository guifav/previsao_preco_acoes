"""
Definicao do modelo LSTM para predicao de precos de acoes.

Este modulo contem a classe LSTMModel que pode ser usada tanto
para treinamento quanto para inferencia.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Modelo LSTM para predicao de series temporais financeiras.

    Arquitetura:
        - N camadas LSTM empilhadas com dropout
        - Camadas fully connected (FC) para gerar a previsao
        - Ativacao ReLU entre as camadas FC

    Parametros:
        input_size (int): Numero de features de entrada (default: 1 para Close)
        hidden_size (int): Neuronios por camada LSTM (default: 128)
        num_layers (int): Camadas LSTM empilhadas (default: 2)
        dropout (float): Taxa de dropout para regularizacao (default: 0.2)
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)

        return prediction
