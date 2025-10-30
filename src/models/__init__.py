# -*- coding: utf-8 -*-
"""
Pacote de Modelos de Forecasting - TJGO

Este pacote contém implementações modulares de diferentes modelos de forecasting:
- BaseForecastingModel: Classe base com métodos comuns
- ProphetModel: Modelo Prophet (Facebook)
- SARIMAXModel: Modelo SARIMAX (Statsmodels)
- ML Models: Random Forest, XGBoost, LightGBM
- BaselineModels: Modelos baseline para comparação
"""

from .base_model import BaseForecastingModel
from .prophet_model import ProphetModel, train_prophet_model
from .sarimax_model import SARIMAXModel, train_sarimax_model
from .ml_models import (
    RandomForestModel, XGBoostModel, LightGBMModel, BaselineModels,
    train_random_forest_model, train_xgboost_model, train_lightgbm_model,
    train_baseline_models
)

__all__ = [
    'BaseForecastingModel',
    'ProphetModel', 'train_prophet_model',
    'SARIMAXModel', 'train_sarimax_model',
    'RandomForestModel', 'XGBoostModel', 'LightGBMModel', 'BaselineModels',
    'train_random_forest_model', 'train_xgboost_model', 'train_lightgbm_model',
    'train_baseline_models'
]

__version__ = '1.0.0'
__author__ = 'Equipe de Data Science - TJGO'

