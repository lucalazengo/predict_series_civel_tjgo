# -*- coding: utf-8 -*-
"""
Modelo Prophet - Forecasting de Casos TJGO

Este módulo implementa o modelo Prophet (Facebook) para previsão de séries temporais.
Prophet é especialmente eficaz para séries com sazonalidade e tendências.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from .base_model import BaseForecastingModel
except ImportError:
    # Para execução direta do módulo
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base_model import BaseForecastingModel

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    Prophet = None

class ProphetModel(BaseForecastingModel):
    """
    Modelo Prophet para forecasting de séries temporais
    """
    
    def __init__(self, train_data, test_data, target_col='TOTAL_CASOS'):
        """
        Inicializa o modelo Prophet
        
        Args:
            train_data (pd.DataFrame): Dados de treino
            test_data (pd.DataFrame): Dados de teste
            target_col (str): Nome da coluna target
        """
        super().__init__(train_data, test_data, target_col)
        
        if not HAS_PROPHET:
            raise ImportError("Prophet não está instalado. Execute: pip install prophet")
        
        self.prophet_model = None
        self.forecast = None
        
    def prepare_prophet_data(self, data):
        """
        Prepara dados no formato Prophet
        
        Args:
            data (pd.DataFrame): Dados originais
            
        Returns:
            pd.DataFrame: Dados no formato Prophet (ds, y)
        """
        prophet_data = data.reset_index()
        prophet_data = prophet_data.rename(columns={'DATA': 'ds', self.target_col: 'y'})
        
        # Remover colunas desnecessárias
        keep_cols = ['ds', 'y']
        exog_vars = ['TAXA_SELIC', 'IPCA', 'TAXA_DESOCUPACAO', 'INADIMPLENCIA']
        
        for var in exog_vars:
            if var in prophet_data.columns:
                keep_cols.append(var)
        
        return prophet_data[keep_cols]
    
    def train(self, exog_vars=None, **prophet_params):
        """
        Treina o modelo Prophet
        
        Args:
            exog_vars (list): Lista de variáveis exógenas
            **prophet_params: Parâmetros do Prophet
        """
        print("🔄 Treinando modelo Prophet...")
        print(f"📊 Dados de treino: {len(self.train_data)} observações")
        print(f"📊 Dados de teste: {len(self.test_data)} observações")
        
        # Parâmetros padrão do Prophet
        default_params = {
            'yearly_seasonality': True,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'seasonality_mode': 'additive',
            'interval_width': 0.95,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0
        }
        
        # Atualizar com parâmetros fornecidos
        default_params.update(prophet_params)
        
        print(f"🔧 Parâmetros do Prophet: {default_params}")
        
        # Criar modelo Prophet
        self.prophet_model = Prophet(**default_params)
        
        # Preparar dados de treino
        train_prophet = self.prepare_prophet_data(self.train_data)
        print(f"📋 Dados preparados para Prophet: {train_prophet.shape}")
        print(f"📋 Colunas disponíveis: {list(train_prophet.columns)}")
        
        # Variáveis exógenas padrão se não especificadas
        if exog_vars is None:
            exog_vars = ['TAXA_SELIC', 'IPCA', 'TAXA_DESOCUPACAO', 'INADIMPLENCIA']
        
        print(f"🎯 Variáveis exógenas a serem adicionadas: {exog_vars}")
        
        # Adicionar regressores exógenos
        added_regressors = []
        for var in exog_vars:
            if var in train_prophet.columns:
                self.prophet_model.add_regressor(var)
                added_regressors.append(var)
                print(f"  ✅ Adicionada variável exógena: {var}")
            else:
                print(f"  ❌ Variável exógena não encontrada: {var}")
        
        print(f"📊 Total de regressores adicionados: {len(added_regressors)}")
        
        # Verificar se há dados suficientes
        if len(train_prophet) < 2:
            raise ValueError("Dados insuficientes para treinamento (mínimo 2 observações)")
        
        # Verificar valores nulos
        null_counts = train_prophet.isnull().sum()
        if null_counts.sum() > 0:
            print(f"⚠️ Valores nulos encontrados: {null_counts[null_counts > 0].to_dict()}")
            # Preencher valores nulos com média
            train_prophet = train_prophet.fillna(train_prophet.mean())
            print("✅ Valores nulos preenchidos com média")
        
        print("🚀 Iniciando treinamento do Prophet...")
        import time
        start_time = time.time()
        
        # Treinar modelo
        self.prophet_model.fit(train_prophet)
        
        training_time = time.time() - start_time
        print(f"⏱️ Tempo de treinamento: {training_time:.2f} segundos")
        
        # Fazer previsões
        print("🔮 Fazendo previsões...")
        self.predict()
        
        print("✅ Modelo Prophet treinado com sucesso!")
        
        return self.prophet_model
    
    def predict(self):
        """
        Faz previsões usando o modelo Prophet treinado
        """
        if self.prophet_model is None:
            raise ValueError("Modelo não foi treinado. Execute train() primeiro.")
        
        print("🔮 Iniciando processo de previsão...")
        
        # Criar dataframe futuro
        future = self.prophet_model.make_future_dataframe(
            periods=len(self.test_data), 
            freq='MS'
        )
        
        print(f"📅 DataFrame futuro criado: {future.shape}")
        print(f"📅 Período de previsão: {future['ds'].min()} a {future['ds'].max()}")
        
        # Adicionar variáveis exógenas para o período futuro
        exog_vars = ['TAXA_SELIC', 'IPCA', 'TAXA_DESOCUPACAO', 'INADIMPLENCIA']
        
        print("🔧 Adicionando variáveis exógenas ao período futuro...")
        for var in exog_vars:
            if var in self.test_data.columns:
                # Usar valores do período de teste
                future[var] = np.concatenate([
                    self.train_data[var].values,
                    self.test_data[var].values
                ])
                print(f"  ✅ {var}: {len(self.train_data[var])} valores treino + {len(self.test_data[var])} valores teste")
            else:
                print(f"  ❌ Variável exógena não encontrada nos dados de teste: {var}")
        
        # Verificar se há valores nulos nas variáveis exógenas
        null_counts = future[exog_vars].isnull().sum()
        if null_counts.sum() > 0:
            print(f"⚠️ Valores nulos nas variáveis exógenas: {null_counts[null_counts > 0].to_dict()}")
            future[exog_vars] = future[exog_vars].fillna(future[exog_vars].mean())
            print("✅ Valores nulos preenchidos com média")
        
        print("🚀 Fazendo previsões com Prophet...")
        import time
        start_time = time.time()
        
        # Fazer previsões
        self.forecast = self.prophet_model.predict(future)
        
        prediction_time = time.time() - start_time
        print(f"⏱️ Tempo de previsão: {prediction_time:.2f} segundos")
        
        # Extrair apenas as previsões do período de teste
        self.predictions = self.forecast['yhat'].iloc[-len(self.test_data):].values
        
        print(f"📊 Previsões geradas: {len(self.predictions)} valores")
        print(f"📊 Range das previsões: {self.predictions.min():.2f} a {self.predictions.max():.2f}")
        
        # Calcular métricas
        y_true = self.test_data[self.target_col]
        self.metrics = self.calculate_metrics(y_true, self.predictions)
        
        print(f"📈 Métricas calculadas: MAE={self.metrics['mae']:.2f}, R²={self.metrics['r2']:.3f}")
        
        return self.predictions
    
    def plot_components(self, save_path=None):
        """
        Plota componentes do modelo Prophet
        
        Args:
            save_path (str): Caminho para salvar o gráfico
        """
        if self.forecast is None:
            print("❌ Nenhuma previsão disponível. Execute predict() primeiro.")
            return
        
        # Plotar componentes
        fig = self.prophet_model.plot_components(self.forecast)
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Componentes salvos em {save_path}")
        
        plt.show()
    
    def plot_forecast_with_uncertainty(self, save_path=None):
        """
        Plota previsões com intervalos de incerteza
        
        Args:
            save_path (str): Caminho para salvar o gráfico
        """
        if self.forecast is None:
            print("❌ Nenhuma previsão disponível. Execute predict() primeiro.")
            return
        
        plt.figure(figsize=(15, 8))
        
        # Plotar dados históricos
        plt.plot(self.train_data.index, self.train_data[self.target_col], 
                label='Dados Históricos', linewidth=2, color='blue', alpha=0.8)
        
        # Plotar dados de teste
        plt.plot(self.test_data.index, self.test_data[self.target_col], 
                label='Dados Reais (Teste)', linewidth=2, color='green')
        
        # Plotar previsões
        future_data = self.forecast.tail(len(self.test_data))
        plt.plot(future_data['ds'], future_data['yhat'], 
                label='Previsão', linewidth=3, color='red', linestyle='--')
        
        # Plotar intervalos de confiança
        plt.fill_between(future_data['ds'], 
                        future_data['yhat_lower'], 
                        future_data['yhat_upper'],
                        alpha=0.3, color='red', label='Intervalo de Confiança 95%')
        
        plt.title('Prophet - Previsões com Intervalos de Incerteza', 
                 fontweight='bold', fontsize=16)
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('TOTAL_CASOS', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico com incerteza salvo em {save_path}")
        
        plt.show()
    
    def get_forecast_components(self):
        """
        Retorna componentes da previsão
        
        Returns:
            pd.DataFrame: Componentes da previsão
        """
        if self.forecast is None:
            print("❌ Nenhuma previsão disponível. Execute predict() primeiro.")
            return None
        
        # Extrair apenas o período de teste
        future_data = self.forecast.tail(len(self.test_data))
        
        components = pd.DataFrame({
            'Data': future_data['ds'],
            'Previsao': future_data['yhat'],
            'Tendencia': future_data['trend'],
            'Sazonalidade': future_data['yearly'],
            'Limite_Inferior': future_data['yhat_lower'],
            'Limite_Superior': future_data['yhat_upper']
        })
        
        return components
    
    def save_forecast_components(self, save_path):
        """
        Salva componentes da previsão
        
        Args:
            save_path (str): Caminho para salvar
        """
        components = self.get_forecast_components()
        
        if components is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            components.to_csv(save_path, index=False)
            print(f"✅ Componentes salvos em {save_path}")
    
    def print_model_info(self):
        """
        Imprime informações do modelo Prophet
        """
        if self.prophet_model is None:
            print("❌ Modelo não foi treinado.")
            return
        
        print("\n📊 INFORMAÇÕES DO MODELO PROPHET")
        print("="*60)
        print(f"Parâmetros:")
        print(f"  - Sazonalidade Anual: {self.prophet_model.yearly_seasonality}")
        print(f"  - Sazonalidade Semanal: {self.prophet_model.weekly_seasonality}")
        print(f"  - Sazonalidade Diária: {self.prophet_model.daily_seasonality}")
        print(f"  - Modo de Sazonalidade: {self.prophet_model.seasonality_mode}")
        print(f"  - Intervalo de Confiança: {self.prophet_model.interval_width}")
        print(f"  - Changepoint Prior Scale: {self.prophet_model.changepoint_prior_scale}")
        print(f"  - Seasonality Prior Scale: {self.prophet_model.seasonality_prior_scale}")
        
        # Regressores exógenos
        regressors = list(self.prophet_model.extra_regressors.keys())
        print(f"\nRegressores Exógenos ({len(regressors)}):")
        if regressors:
            for regressor in regressors:
                regressor_info = self.prophet_model.extra_regressors[regressor]
                print(f"  ✅ {regressor}: {regressor_info}")
        else:
            print("  ❌ Nenhum regressor exógeno adicionado")
        
        # Informações sobre os dados
        if hasattr(self, 'forecast') and self.forecast is not None:
            print(f"\nDados de Previsão:")
            print(f"  - Total de períodos: {len(self.forecast)}")
            print(f"  - Período de teste: {len(self.test_data)}")
            print(f"  - Período de treino: {len(self.train_data)}")
        
        print("="*60)

def train_prophet_model(train_data, test_data, target_col='TOTAL_CASOS', 
                       exog_vars=None, **prophet_params):
    """
    Função utilitária para treinar modelo Prophet
    
    Args:
        train_data (pd.DataFrame): Dados de treino
        test_data (pd.DataFrame): Dados de teste
        target_col (str): Nome da coluna target
        exog_vars (list): Variáveis exógenas
        **prophet_params: Parâmetros do Prophet
        
    Returns:
        ProphetModel: Modelo treinado
    """
    model = ProphetModel(train_data, test_data, target_col)
    model.train(exog_vars=exog_vars, **prophet_params)
    return model

if __name__ == "__main__":
    # Exemplo de uso
    print("📊 Exemplo de uso do ProphetModel")
    print("Execute este módulo através do script principal ou notebook.")

