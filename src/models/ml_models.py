# -*- coding: utf-8 -*-
"""
Modelos de Machine Learning - Forecasting de Casos TJGO

Este módulo implementa modelos de machine learning para previsão de séries temporais:
- Random Forest
- XGBoost
- LightGBM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from .base_model import BaseForecastingModel

# Optional imports
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    xgb = None

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    lgb = None

class RandomForestModel(BaseForecastingModel):
    """
    Modelo Random Forest para forecasting
    """
    
    def __init__(self, train_data, test_data, target_col='TOTAL_CASOS'):
        super().__init__(train_data, test_data, target_col)
        
    def train(self, n_estimators=100, max_depth=10, min_samples_split=5, 
              min_samples_leaf=2, random_state=42, n_jobs=-1):
        """
        Treina modelo Random Forest
        
        Args:
            n_estimators (int): Número de árvores
            max_depth (int): Profundidade máxima
            min_samples_split (int): Mínimo para dividir
            min_samples_leaf (int): Mínimo por folha
            random_state (int): Seed para reprodutibilidade
            n_jobs (int): Número de jobs paralelos
        """
        print("🔄 Treinando Random Forest...")
        
        # Preparar features
        X_train = self.prepare_features(self.train_data, include_target=False)
        X_test = self.prepare_features(self.test_data, include_target=False)
        y_train = self.train_data[self.target_col]
        
        # Criar modelo
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        # Treinar modelo
        self.model.fit(X_train, y_train)
        
        # Fazer previsões
        self.predictions = self.model.predict(X_test)
        
        # Calcular métricas
        y_true = self.test_data[self.target_col]
        self.metrics = self.calculate_metrics(y_true, self.predictions)
        
        print("✅ Random Forest treinado com sucesso!")
        
        return self.model
    
    def optimize_hyperparameters(self, param_grid=None, cv=5):
        """
        Otimiza hiperparâmetros usando Grid Search
        
        Args:
            param_grid (dict): Grid de parâmetros
            cv (int): Número de folds para CV
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        print("🔄 Otimizando hiperparâmetros do Random Forest...")
        
        # Preparar dados
        X_train = self.prepare_features(self.train_data, include_target=False)
        y_train = self.train_data[self.target_col]
        
        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Grid Search
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Melhores parâmetros: {grid_search.best_params_}")
        print(f"✅ Melhor score: {-grid_search.best_score_:.2f}")
        
        # Treinar com melhores parâmetros
        self.model = grid_search.best_estimator_
        
        # Fazer previsões
        X_test = self.prepare_features(self.test_data, include_target=False)
        self.predictions = self.model.predict(X_test)
        
        # Calcular métricas
        y_true = self.test_data[self.target_col]
        self.metrics = self.calculate_metrics(y_true, self.predictions)
        
        return grid_search.best_params_

class XGBoostModel(BaseForecastingModel):
    """
    Modelo XGBoost para forecasting
    """
    
    def __init__(self, train_data, test_data, target_col='TOTAL_CASOS'):
        super().__init__(train_data, test_data, target_col)
        
        if not HAS_XGB:
            raise ImportError("XGBoost não está instalado. Execute: pip install xgboost")
    
    def train(self, n_estimators=100, max_depth=6, learning_rate=0.1,
              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1):
        """
        Treina modelo XGBoost
        
        Args:
            n_estimators (int): Número de árvores
            max_depth (int): Profundidade máxima
            learning_rate (float): Taxa de aprendizado
            subsample (float): Subamostragem
            colsample_bytree (float): Subamostragem de features
            random_state (int): Seed para reprodutibilidade
            n_jobs (int): Número de jobs paralelos
        """
        print("🔄 Treinando XGBoost...")
        
        # Preparar features
        X_train = self.prepare_features(self.train_data, include_target=False)
        X_test = self.prepare_features(self.test_data, include_target=False)
        y_train = self.train_data[self.target_col]
        
        # Criar modelo
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        # Treinar modelo
        self.model.fit(X_train, y_train)
        
        # Fazer previsões
        self.predictions = self.model.predict(X_test)
        
        # Calcular métricas
        y_true = self.test_data[self.target_col]
        self.metrics = self.calculate_metrics(y_true, self.predictions)
        
        print("✅ XGBoost treinado com sucesso!")
        
        return self.model
    
    def optimize_hyperparameters(self, param_grid=None, cv=5):
        """
        Otimiza hiperparâmetros usando Grid Search
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        print("🔄 Otimizando hiperparâmetros do XGBoost...")
        
        # Preparar dados
        X_train = self.prepare_features(self.train_data, include_target=False)
        y_train = self.train_data[self.target_col]
        
        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Grid Search
        grid_search = GridSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=-1),
            param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Melhores parâmetros: {grid_search.best_params_}")
        print(f"✅ Melhor score: {-grid_search.best_score_:.2f}")
        
        # Treinar com melhores parâmetros
        self.model = grid_search.best_estimator_
        
        # Fazer previsões
        X_test = self.prepare_features(self.test_data, include_target=False)
        self.predictions = self.model.predict(X_test)
        
        # Calcular métricas
        y_true = self.test_data[self.target_col]
        self.metrics = self.calculate_metrics(y_true, self.predictions)
        
        return grid_search.best_params_

class LightGBMModel(BaseForecastingModel):
    """
    Modelo LightGBM para forecasting
    """
    
    def __init__(self, train_data, test_data, target_col='TOTAL_CASOS'):
        super().__init__(train_data, test_data, target_col)
        
        if not HAS_LGB:
            raise ImportError("LightGBM não está instalado. Execute: pip install lightgbm")
    
    def train(self, n_estimators=100, max_depth=6, learning_rate=0.1,
              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1):
        """
        Treina modelo LightGBM
        
        Args:
            n_estimators (int): Número de árvores
            max_depth (int): Profundidade máxima
            learning_rate (float): Taxa de aprendizado
            subsample (float): Subamostragem
            colsample_bytree (float): Subamostragem de features
            random_state (int): Seed para reprodutibilidade
            n_jobs (int): Número de jobs paralelos
            verbose (int): Verbosidade
        """
        print("🔄 Treinando LightGBM...")
        
        # Preparar features
        X_train = self.prepare_features(self.train_data, include_target=False)
        X_test = self.prepare_features(self.test_data, include_target=False)
        y_train = self.train_data[self.target_col]
        
        # Criar modelo
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        # Treinar modelo
        self.model.fit(X_train, y_train)
        
        # Fazer previsões
        self.predictions = self.model.predict(X_test)
        
        # Calcular métricas
        y_true = self.test_data[self.target_col]
        self.metrics = self.calculate_metrics(y_true, self.predictions)
        
        print("✅ LightGBM treinado com sucesso!")
        
        return self.model
    
    def optimize_hyperparameters(self, param_grid=None, cv=5):
        """
        Otimiza hiperparâmetros usando Grid Search
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        print("🔄 Otimizando hiperparâmetros do LightGBM...")
        
        # Preparar dados
        X_train = self.prepare_features(self.train_data, include_target=False)
        y_train = self.train_data[self.target_col]
        
        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Grid Search
        grid_search = GridSearchCV(
            lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Melhores parâmetros: {grid_search.best_params_}")
        print(f"✅ Melhor score: {-grid_search.best_score_:.2f}")
        
        # Treinar com melhores parâmetros
        self.model = grid_search.best_estimator_
        
        # Fazer previsões
        X_test = self.prepare_features(self.test_data, include_target=False)
        self.predictions = self.model.predict(X_test)
        
        # Calcular métricas
        y_true = self.test_data[self.target_col]
        self.metrics = self.calculate_metrics(y_true, self.predictions)
        
        return grid_search.best_params_

class BaselineModels(BaseForecastingModel):
    """
    Modelos baseline para comparação
    """
    
    def __init__(self, train_data, test_data, target_col='TOTAL_CASOS'):
        super().__init__(train_data, test_data, target_col)
        self.baseline_results = {}
    
    def persistence_model(self):
        """
        Modelo de persistência: usa último valor conhecido
        """
        print("🔄 Treinando modelo baseline (persistência)...")
        
        # Usar último valor do treino como previsão
        last_value = self.train_data[self.target_col].iloc[-1]
        predictions = np.full(len(self.test_data), last_value)
        
        # Calcular métricas
        y_true = self.test_data[self.target_col]
        metrics = self.calculate_metrics(y_true, predictions)
        
        self.baseline_results['persistence'] = {
            'model': 'Baseline Persistência',
            'predictions': predictions,
            'metrics': metrics
        }
        
        print(f"✅ Baseline (persistência): MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        
        return predictions, metrics
    
    def moving_average_model(self, window=12):
        """
        Modelo de média móvel
        
        Args:
            window (int): Janela da média móvel
        """
        print(f"🔄 Treinando modelo baseline (média móvel {window} meses)...")
        
        # Calcular média móvel no treino
        train_ma = self.train_data[self.target_col].rolling(window=window).mean()
        last_ma = train_ma.iloc[-1]
        
        # Previsões usando a última média móvel
        predictions = np.full(len(self.test_data), last_ma)
        
        # Calcular métricas
        y_true = self.test_data[self.target_col]
        metrics = self.calculate_metrics(y_true, predictions)
        
        self.baseline_results['moving_average'] = {
            'model': f'Baseline Média Móvel ({window})',
            'predictions': predictions,
            'metrics': metrics
        }
        
        print(f"✅ Baseline (média móvel): MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        
        return predictions, metrics
    
    def plot_all_baselines(self, save_path=None):
        """
        Plota todos os modelos baseline
        """
        if not self.baseline_results:
            print("❌ Nenhum modelo baseline treinado.")
            return
        
        fig, axes = plt.subplots(1, len(self.baseline_results), figsize=(15, 5))
        if len(self.baseline_results) == 1:
            axes = [axes]
        
        for i, (name, result) in enumerate(self.baseline_results.items()):
            ax = axes[i]
            
            # Plotar dados reais e previsões
            ax.plot(self.test_data.index, self.test_data[self.target_col], 
                   label='Real', linewidth=2, color='blue')
            ax.plot(self.test_data.index, result['predictions'], 
                   label='Previsão', linewidth=2, color='red', linestyle='--')
            
            ax.set_title(f"{result['model']} (MAE: {result['metrics']['mae']:.0f})", fontweight='bold')
            ax.set_xlabel('Data')
            ax.set_ylabel('TOTAL_CASOS')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Gráficos baseline salvos em {save_path}")
        
        plt.show()

# Funções utilitárias
def train_random_forest_model(train_data, test_data, target_col='TOTAL_CASOS', **params):
    """Função utilitária para treinar Random Forest"""
    model = RandomForestModel(train_data, test_data, target_col)
    model.train(**params)
    return model

def train_xgboost_model(train_data, test_data, target_col='TOTAL_CASOS', **params):
    """Função utilitária para treinar XGBoost"""
    model = XGBoostModel(train_data, test_data, target_col)
    model.train(**params)
    return model

def train_lightgbm_model(train_data, test_data, target_col='TOTAL_CASOS', **params):
    """Função utilitária para treinar LightGBM"""
    model = LightGBMModel(train_data, test_data, target_col)
    model.train(**params)
    return model

def train_baseline_models(train_data, test_data, target_col='TOTAL_CASOS'):
    """Função utilitária para treinar modelos baseline"""
    model = BaselineModels(train_data, test_data, target_col)
    model.persistence_model()
    model.moving_average_model()
    return model

if __name__ == "__main__":
    # Exemplo de uso
    print("📊 Exemplo de uso dos modelos de ML")
    print("Execute este módulo através do script principal ou notebook.")

