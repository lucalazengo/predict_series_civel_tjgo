# Projeto de Forecasting - Casos TJGO

## Objetivo

Prever novos casos em um tribunal seguindo a metodologia CRISP-DM para forecasting de séries temporais.

## Estrutura do Projeto

```
├── data/
│   ├── raw/                    # Dados originais
│   └── processed/              # Dados processados
├── notebooks/                  # Jupyter notebooks para análise
├── src/                        # Scripts Python modulares
├── reports/                    # Relatórios e métricas
├── artifacts/                  # Modelos serializados e logs
└── requirements.txt           # Dependências Python
```

## Setup e Execução

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar EDA

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 3. Preparar dados

```bash
python src/data_preparation.py
```

### 4. Treinar modelos

```bash
python src/train_models.py
```

### 5. Avaliar modelos

```bash
python src/evaluate_models.py
```

## Métricas de Avaliacao

- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² (Coeficiente de Determinação)

## Target

- **TOTAL_CASOS**: Número total de casos por mês.
