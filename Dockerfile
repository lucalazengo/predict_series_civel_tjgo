# Dockerfile para o projeto de forecasting TJGO
FROM python:3.9-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY . .

# Criar diretórios necessários
RUN mkdir -p data/raw data/processed notebooks reports artifacts

# Definir variáveis de ambiente
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:///app/artifacts/mlruns

# Comando padrão
CMD ["python", "src/train_models.py"]
