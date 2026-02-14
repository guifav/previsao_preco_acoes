FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar codigo
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Porta padrao do HuggingFace Spaces
EXPOSE 7860

# Variavel de ambiente para o diretorio de modelos
ENV MODELS_DIR=models

# Rodar a aplicacao
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
