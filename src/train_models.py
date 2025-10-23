"""
Script de Treinamento de Modelos
Projeto de Forecasting - Casos TJGO

Este script implementa diferentes modelos de forecasting:
- Baseline (persistência, média móvel)
- SARIMAX
- Prophet
- Modelos de árvore (RandomForest, XGBoost, LightGBM)

Com logging de experimentos usando MLflow.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
# Optional imports (handled gracefully if missing)
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    xgb = None

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
    lgb = None
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False
    Prophet = None

try:
    import mlflow
    HAS_MLFLOW = True
    # Optional providers
    try:
        import mlflow.sklearn  # noqa: F401
    except Exception:
        pass
    try:
        import mlflow.xgboost  # noqa: F401
    except Exception:
        pass
    try:
        import mlflow.lightgbm  # noqa: F401
    except Exception:
        pass
except Exception:
    HAS_MLFLOW = False
    mlflow = None
import warnings
warnings.filterwarnings('ignore')

class ForecastingModels:
    """
    Classe para treinamento e avaliação de modelos de forecasting
    """
    
    def __init__(self, train_data, test_data, target_col='TOTAL_CASOS'):
        """
        Inicializa os modelos
        
        Args:
            train_data (pd.DataFrame): Dados de treino
            test_data (pd.DataFrame): Dados de teste
            target_col (str): Nome da coluna target
        """
        self.train_data = train_data
        self.test_data = test_data
        self.target_col = target_col
        self.models = {}
        self.results = {}
        
        # Configurar MLflow (se disponível)
        if HAS_MLFLOW:
            mlflow.set_experiment("tjgo_forecasting")
        
    def prepare_features(self, data, include_target=True):
        """
        Prepara features para modelos de ML
        
        Args:
            data (pd.DataFrame): Dados
            include_target (bool): Se deve incluir target nas features
        """
        # Selecionar features numéricas
        feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not include_target and self.target_col in feature_cols:
            feature_cols.remove(self.target_col)
        
        # Remover colunas com muitos NaN
        feature_cols = [col for col in feature_cols if data[col].isnull().sum() < len(data) * 0.5]
        
        return data[feature_cols].fillna(0)
    
    def baseline_persistence(self):
        """
        Modelo baseline: persistência (último valor)
        """
        print("🔄 Treinando modelo baseline (persistência)...")
        
        # Usar último valor do treino como previsão
        last_value = self.train_data[self.target_col].iloc[-1]
        predictions = np.full(len(self.test_data), last_value)
        
        # Calcular métricas
        y_true = self.test_data[self.target_col]
        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        r2 = r2_score(y_true, predictions)
        
        self.results['baseline_persistence'] = {
            'model': 'Baseline Persistência',
            'predictions': predictions,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"✅ Baseline (persistência): MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
        
        return predictions
    
    def baseline_moving_average(self, window=12):
        """
        Modelo baseline: média móvel
        
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
        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        r2 = r2_score(y_true, predictions)
        
        self.results['baseline_moving_average'] = {
            'model': f'Baseline Média Móvel ({window})',
            'predictions': predictions,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"✅ Baseline (média móvel): MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
        
        return predictions
    
    def train_sarimax(self, order=(1,1,1), seasonal_order=(1,1,1,12), exog_vars=None):
        """
        Treina modelo SARIMAX
        
        Args:
            order (tuple): Ordem ARIMA (p,d,q)
            seasonal_order (tuple): Ordem sazonal (P,D,Q,s)
            exog_vars (list): Variáveis exógenas
        """
        print("🔄 Treinando modelo SARIMAX...")
        
        try:
            # Preparar dados
            y_train = self.train_data[self.target_col]
            
            if exog_vars is None:
                # Usar variáveis de alta correlação identificadas no EDA
                exog_vars = ['TAXA_SELIC', 'IPCA', 'TAXA_DESOCUPACAO', 'INADIMPLENCIA', 
                           'qt_acidente', 'QT_ELEITOR']
            
            if exog_vars:
                exog_train = self.train_data[exog_vars]
                exog_test = self.test_data[exog_vars]
            else:
                exog_train = None
                exog_test = None
            
            # Treinar modelo
            model = SARIMAX(
                y_train,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False)
            
            # Fazer previsões
            predictions = fitted_model.forecast(steps=len(self.test_data), exog=exog_test)
            
            # Calcular métricas
            y_true = self.test_data[self.target_col]
            mae = mean_absolute_error(y_true, predictions)
            rmse = np.sqrt(mean_squared_error(y_true, predictions))
            r2 = r2_score(y_true, predictions)
            
            self.results['sarimax'] = {
                'model': 'SARIMAX',
                'fitted_model': fitted_model,
                'predictions': predictions,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"✅ SARIMAX: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
            
            return fitted_model, predictions
            
        except Exception as e:
            print(f"❌ Erro no SARIMAX: {str(e)}")
            return None, None
    
    def train_prophet(self, exog_vars=None):
        """
        Treina modelo Prophet
        
        Args:
            exog_vars (list): Variáveis exógenas
        """
        print("🔄 Treinando modelo Prophet...")
        
        try:
            # Preparar dados para Prophet
            train_prophet = self.train_data.reset_index()
            train_prophet = train_prophet.rename(columns={'DATA': 'ds', self.target_col: 'y'})
            
            # Inicializar modelo
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive'
            )
            
            # Usar variáveis de alta correlação se não especificadas
            if exog_vars is None:
                exog_vars = ['TAXA_SELIC', 'IPCA', 'TAXA_DESOCUPACAO', 'INADIMPLENCIA', 
                           'qt_acidente', 'QT_ELEITOR']
            
            # Adicionar regressores exógenos
            if exog_vars:
                for var in exog_vars:
                    if var in train_prophet.columns:
                        model.add_regressor(var)
            
            # Treinar modelo
            model.fit(train_prophet)
            
            # Fazer previsões
            future = model.make_future_dataframe(periods=len(self.test_data), freq='MS')
            
            if exog_vars:
                for var in exog_vars:
                    if var in self.test_data.columns:
                        future[var] = np.concatenate([
                            self.train_data[var].values,
                            self.test_data[var].values
                        ])
            
            forecast = model.predict(future)
            predictions = forecast['yhat'].iloc[-len(self.test_data):].values
            
            # Calcular métricas
            y_true = self.test_data[self.target_col]
            mae = mean_absolute_error(y_true, predictions)
            rmse = np.sqrt(mean_squared_error(y_true, predictions))
            r2 = r2_score(y_true, predictions)
            
            self.results['prophet'] = {
                'model': 'Prophet',
                'fitted_model': model,
                'predictions': predictions,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"✅ Prophet: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
            
            return model, predictions
            
        except Exception as e:
            print(f"❌ Erro no Prophet: {str(e)}")
            return None, None
    
    def train_random_forest(self, n_estimators=100, max_depth=10):
        """
        Treina Random Forest
        
        Args:
            n_estimators (int): Número de árvores
            max_depth (int): Profundidade máxima
        """
        print("🔄 Treinando Random Forest...")
        
        try:
            # Preparar features
            X_train = self.prepare_features(self.train_data, include_target=False)
            X_test = self.prepare_features(self.test_data, include_target=False)
            y_train = self.train_data[self.target_col]
            
            # Treinar modelo
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Fazer previsões
            predictions = model.predict(X_test)
            
            # Calcular métricas
            y_true = self.test_data[self.target_col]
            mae = mean_absolute_error(y_true, predictions)
            rmse = np.sqrt(mean_squared_error(y_true, predictions))
            r2 = r2_score(y_true, predictions)
            
            self.results['random_forest'] = {
                'model': 'Random Forest',
                'fitted_model': model,
                'predictions': predictions,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"✅ Random Forest: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
            
            return model, predictions
            
        except Exception as e:
            print(f"❌ Erro no Random Forest: {str(e)}")
            return None, None
    
    def train_xgboost(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        """
        Treina XGBoost
        
        Args:
            n_estimators (int): Número de árvores
            max_depth (int): Profundidade máxima
            learning_rate (float): Taxa de aprendizado
        """
        print("🔄 Treinando XGBoost...")
        
        try:
            # Preparar features
            X_train = self.prepare_features(self.train_data, include_target=False)
            X_test = self.prepare_features(self.test_data, include_target=False)
            y_train = self.train_data[self.target_col]
            
            # Treinar modelo
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Fazer previsões
            predictions = model.predict(X_test)
            
            # Calcular métricas
            y_true = self.test_data[self.target_col]
            mae = mean_absolute_error(y_true, predictions)
            rmse = np.sqrt(mean_squared_error(y_true, predictions))
            r2 = r2_score(y_true, predictions)
            
            self.results['xgboost'] = {
                'model': 'XGBoost',
                'fitted_model': model,
                'predictions': predictions,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"✅ XGBoost: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
            
            return model, predictions
            
        except Exception as e:
            print(f"❌ Erro no XGBoost: {str(e)}")
            return None, None
    
    def train_lightgbm(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        """
        Treina LightGBM
        
        Args:
            n_estimators (int): Número de árvores
            max_depth (int): Profundidade máxima
            learning_rate (float): Taxa de aprendizado
        """
        print("🔄 Treinando LightGBM...")
        
        try:
            # Preparar features
            X_train = self.prepare_features(self.train_data, include_target=False)
            X_test = self.prepare_features(self.test_data, include_target=False)
            y_train = self.train_data[self.target_col]
            
            # Treinar modelo
            model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            model.fit(X_train, y_train)
            
            # Fazer previsões
            predictions = model.predict(X_test)
            
            # Calcular métricas
            y_true = self.test_data[self.target_col]
            mae = mean_absolute_error(y_true, predictions)
            rmse = np.sqrt(mean_squared_error(y_true, predictions))
            r2 = r2_score(y_true, predictions)
            
            self.results['lightgbm'] = {
                'model': 'LightGBM',
                'fitted_model': model,
                'predictions': predictions,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"✅ LightGBM: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
            
            return model, predictions
            
        except Exception as e:
            print(f"❌ Erro no LightGBM: {str(e)}")
            return None, None
    
    def log_experiment(self, model_name, model, metrics, params=None):
        """
        Log experimento no MLflow
        
        Args:
            model_name (str): Nome do modelo
            model: Modelo treinado
            metrics (dict): Métricas
            params (dict): Parâmetros do modelo
        """
        if not HAS_MLFLOW:
            # Sem MLflow, apenas imprime e retorna
            print(f"MLflow indisponível. Métricas {model_name}: {metrics}")
            return

        with mlflow.start_run(run_name=model_name):
            # Log parâmetros
            if params:
                mlflow.log_params(params)
            
            # Log métricas
            mlflow.log_metrics(metrics)
            
            # Log modelo
            if model_name in ['random_forest', 'xgboost', 'lightgbm']:
                if model_name == 'random_forest':
                    mlflow.sklearn.log_model(model, "model")
                elif model_name == 'xgboost':
                    mlflow.xgboost.log_model(model, "model")
                elif model_name == 'lightgbm':
                    mlflow.lightgbm.log_model(model, "model")
    
    def compare_models(self):
        """
        Compara todos os modelos treinados
        """
        print("\n📊 Comparação de Modelos:")
        print("="*50)
        
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Modelo': result['model'],
                'MAE': result['mae'],
                'RMSE': result['rmse'],
                'R²': result['r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('MAE')
        
        print(comparison_df.to_string(index=False))
        
        # Salvar resultados
        import os
        os.makedirs('./reports', exist_ok=True)
        comparison_df.to_csv('./reports/metrics.csv', index=False)
        print("\n✅ Métricas salvas em ./reports/metrics.csv")
        
        return comparison_df
    
    def plot_predictions(self, save_plots=True):
        """
        Plota previsões de todos os modelos
        """
        print("\n📈 Gerando visualizações...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        model_names = list(self.results.keys())
        
        for i, (model_name, result) in enumerate(self.results.items()):
            if i < 4:  # Máximo 4 subplots
                ax = axes[i]
                
                # Plotar dados reais e previsões
                ax.plot(self.test_data.index, self.test_data[self.target_col], 
                       label='Real', linewidth=2, color='blue')
                ax.plot(self.test_data.index, result['predictions'], 
                       label='Previsão', linewidth=2, color='red', linestyle='--')
                
                ax.set_title(f"{result['model']} (MAE: {result['mae']:.0f})", fontweight='bold')
                ax.set_xlabel('Data')
                ax.set_ylabel('TOTAL_CASOS')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Remover subplots vazios
        for i in range(len(model_names), 4):
            axes[i].remove()
        
        plt.tight_layout()
        
        if save_plots:
            import os
            os.makedirs('./reports', exist_ok=True)
            plt.savefig('./reports/predictions_comparison.png', dpi=300, bbox_inches='tight')
            print("✅ Gráficos salvos em ./reports/predictions_comparison.png")
        
        plt.show()

def main():
    """
    Função principal para treinar todos os modelos
    """
    print("🚀 Iniciando treinamento de modelos...")
    print("="*50)
    
    # Carregar dados processados
    try:
        # Usar caminho relativo ao diretório de execução do projeto
        train_data = pd.read_csv('./data/processed/train.csv', index_col='DATA', parse_dates=True)
        test_data = pd.read_csv('./data/processed/test.csv', index_col='DATA', parse_dates=True)
        print("✅ Dados processados carregados")
    except FileNotFoundError:
        print("❌ Dados processados não encontrados. Execute data_preparation.py primeiro.")
        return
    
    # Inicializar modelos
    models = ForecastingModels(train_data, test_data)
    
    # Treinar todos os modelos
    print("\n🔄 Treinando modelos...")
    
    # Baselines
    models.baseline_persistence()
    models.baseline_moving_average()
    
    # Modelos avançados
    models.train_sarimax()
    if HAS_PROPHET:
        models.train_prophet()
    else:
        print("⚠️ Prophet indisponível. Pulando este modelo.")

    models.train_random_forest()

    if HAS_XGB:
        models.train_xgboost()
    else:
        print("⚠️ XGBoost indisponível. Pulando este modelo.")

    if HAS_LGB:
        models.train_lightgbm()
    else:
        print("⚠️ LightGBM indisponível. Pulando este modelo.")
    
    # Comparar modelos
    comparison_df = models.compare_models()
    
    # Gerar visualizações
    models.plot_predictions()
    
    # Log no MLflow
    print("\n📝 Logging experimentos no MLflow...")
    for model_name, result in models.results.items():
        if 'fitted_model' in result:
            models.log_experiment(
                model_name, 
                result['fitted_model'],
                {'mae': result['mae'], 'rmse': result['rmse'], 'r2': result['r2']}
            )
    
    print("\n✅ Treinamento concluído!")
    print(f"🏆 Melhor modelo: {comparison_df.iloc[0]['Modelo']} (MAE: {comparison_df.iloc[0]['MAE']:.2f})")
    
    return models

if __name__ == "__main__":
    models = main()
