# -*- coding: utf-8 -*-
"""
Modelo SARIMAX - Forecasting de Casos TJGO

Este módulo implementa o modelo SARIMAX (Seasonal ARIMA with eXogenous variables)
para previsão de séries temporais com variáveis exógenas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .base_model import BaseForecastingModel

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    HAS_SARIMAX = True
except ImportError:
    HAS_SARIMAX = False
    SARIMAX = None

class SARIMAXModel(BaseForecastingModel):
    """
    Modelo SARIMAX para forecasting de séries temporais
    """
    
    def __init__(self, train_data, test_data, target_col='TOTAL_CASOS'):
        """
        Inicializa o modelo SARIMAX
        
        Args:
            train_data (pd.DataFrame): Dados de treino
            test_data (pd.DataFrame): Dados de teste
            target_col (str): Nome da coluna target
        """
        super().__init__(train_data, test_data, target_col)
        
        if not HAS_SARIMAX:
            raise ImportError("Statsmodels não está instalado. Execute: pip install statsmodels")
        
        self.sarimax_model = None
        self.fitted_model = None
        self.order = None
        self.seasonal_order = None
        
    def check_stationarity(self, series, title="Série"):
        """
        Verifica estacionariedade da série
        
        Args:
            series (pd.Series): Série temporal
            title (str): Título da série
            
        Returns:
            bool: True se estacionária
        """
        result = adfuller(series.dropna())
        
        print(f"\n📊 Teste de Estacionariedade - {title}")
        print("="*40)
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"p-value: {result[1]:.6f}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.6f}")
        
        is_stationary = result[1] < 0.05
        print(f"Estacionária: {'✅ Sim' if is_stationary else '❌ Não'}")
        
        return is_stationary
    
    def make_stationary(self, series, max_diff=2):
        """
        Torna a série estacionária através de diferenciação
        
        Args:
            series (pd.Series): Série temporal
            max_diff (int): Máximo número de diferenciações
            
        Returns:
            tuple: (série diferenciada, número de diferenciações)
        """
        diff_count = 0
        current_series = series.copy()
        
        for i in range(max_diff + 1):
            if self.check_stationarity(current_series, f"Diferenciação {i}"):
                break
            
            if i < max_diff:
                current_series = current_series.diff()
                diff_count += 1
        
        return current_series, diff_count
    
    def train(self, order=(1,1,1), seasonal_order=(1,1,1,12), 
              exog_vars=None, enforce_stationarity=False, enforce_invertibility=False):
        """
        Treina o modelo SARIMAX
        
        Args:
            order (tuple): Ordem ARIMA (p,d,q)
            seasonal_order (tuple): Ordem sazonal (P,D,Q,s)
            exog_vars (list): Variáveis exógenas
            enforce_stationarity (bool): Forçar estacionariedade
            enforce_invertibility (bool): Forçar invertibilidade
        """
        print("🔄 Treinando modelo SARIMAX...")
        
        # Preparar dados
        y_train = self.train_data[self.target_col]
        
        # Variáveis exógenas padrão se não especificadas
        if exog_vars is None:
            exog_vars = ['TAXA_SELIC', 'IPCA', 'TAXA_DESOCUPACAO', 'INADIMPLENCIA']
        
        # Verificar estacionariedade
        print("\n📊 Verificando estacionariedade...")
        is_stationary = self.check_stationarity(y_train, "Série Original")
        
        if not is_stationary and not enforce_stationarity:
            print("⚠️ Série não é estacionária. Considerando diferenciação automática.")
            # Ajustar ordem de diferenciação se necessário
            _, d = self.make_stationary(y_train)
            order = (order[0], d, order[2])
            print(f"📈 Ordem ajustada para: {order}")
        
        # Preparar variáveis exógenas
        exog_train = None
        exog_test = None
        
        if exog_vars:
            # Verificar disponibilidade das variáveis
            available_vars = [var for var in exog_vars if var in self.train_data.columns]
            
            if available_vars:
                exog_train = self.train_data[available_vars]
                exog_test = self.test_data[available_vars]
                
                print(f"✅ Variáveis exógenas utilizadas: {available_vars}")
                
                # Verificar estacionariedade das variáveis exógenas
                for var in available_vars:
                    self.check_stationarity(exog_train[var], f"Exógena: {var}")
            else:
                print("⚠️ Nenhuma variável exógena disponível.")
        
        # Salvar parâmetros
        self.order = order
        self.seasonal_order = seasonal_order
        
        # Criar modelo SARIMAX
        self.sarimax_model = SARIMAX(
            y_train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility
        )
        
        # Treinar modelo
        print("🔄 Ajustando modelo...")
        self.fitted_model = self.sarimax_model.fit(disp=False)
        
        # Fazer previsões
        self.predict(exog_test)
        
        print("✅ Modelo SARIMAX treinado com sucesso!")
        
        return self.fitted_model
    
    def predict(self, exog_test=None):
        """
        Faz previsões usando o modelo SARIMAX treinado
        
        Args:
            exog_test (pd.DataFrame): Variáveis exógenas para teste
        """
        if self.fitted_model is None:
            raise ValueError("Modelo não foi treinado. Execute train() primeiro.")
        
        # Fazer previsões
        self.predictions = self.fitted_model.forecast(steps=len(self.test_data), exog=exog_test)
        
        # Calcular métricas
        y_true = self.test_data[self.target_col]
        self.metrics = self.calculate_metrics(y_true, self.predictions)
        
        return self.predictions
    
    def plot_acf_pacf(self, save_path=None):
        """
        Plota ACF e PACF da série
        
        Args:
            save_path (str): Caminho para salvar o gráfico
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # ACF e PACF da série original
        plot_acf(self.train_data[self.target_col].dropna(), ax=axes[0,0], lags=40)
        axes[0,0].set_title('ACF - Série Original')
        
        plot_pacf(self.train_data[self.target_col].dropna(), ax=axes[0,1], lags=40)
        axes[0,1].set_title('PACF - Série Original')
        
        # ACF e PACF dos resíduos (se modelo treinado)
        if self.fitted_model is not None:
            residuals = self.fitted_model.resid
            
            plot_acf(residuals.dropna(), ax=axes[1,0], lags=40)
            axes[1,0].set_title('ACF - Resíduos')
            
            plot_pacf(residuals.dropna(), ax=axes[1,1], lags=40)
            axes[1,1].set_title('PACF - Resíduos')
        
        plt.tight_layout()
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Gráficos ACF/PACF salvos em {save_path}")
        
        plt.show()
    
    def plot_diagnostics(self, save_path=None):
        """
        Plota diagnósticos do modelo SARIMAX
        
        Args:
            save_path (str): Caminho para salvar o gráfico
        """
        if self.fitted_model is None:
            print("❌ Modelo não foi treinado. Execute train() primeiro.")
            return
        
        # Plotar diagnósticos
        fig = self.fitted_model.plot_diagnostics(figsize=(15, 10))
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Diagnósticos salvos em {save_path}")
        
        plt.show()
    
    def get_model_summary(self):
        """
        Retorna resumo do modelo
        
        Returns:
            str: Resumo do modelo
        """
        if self.fitted_model is None:
            print("❌ Modelo não foi treinado. Execute train() primeiro.")
            return None
        
        return self.fitted_model.summary()
    
    def print_model_info(self):
        """
        Imprime informações do modelo SARIMAX
        """
        if self.fitted_model is None:
            print("❌ Modelo não foi treinado.")
            return
        
        print("\n📊 INFORMAÇÕES DO MODELO SARIMAX")
        print("="*50)
        print(f"Ordem ARIMA: {self.order}")
        print(f"Ordem Sazonal: {self.seasonal_order}")
        print(f"AIC: {self.fitted_model.aic:.2f}")
        print(f"BIC: {self.fitted_model.bic:.2f}")
        print(f"Log-Likelihood: {self.fitted_model.llf:.2f}")
        
        # Teste de Ljung-Box para resíduos
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(self.fitted_model.resid, lags=10, return_df=True)
        print(f"Ljung-Box p-value: {lb_test['lb_pvalue'].iloc[-1]:.4f}")
        
        print("="*50)
    
    def auto_arima(self, max_p=3, max_d=2, max_q=3, 
                   max_P=2, max_D=1, max_Q=2, seasonal=True):
        """
        Seleção automática de parâmetros ARIMA
        
        Args:
            max_p, max_d, max_q: Máximos para ordem ARIMA
            max_P, max_D, max_Q: Máximos para ordem sazonal
            seasonal (bool): Se deve considerar sazonalidade
            
        Returns:
            tuple: Melhor ordem encontrada
        """
        print("🔄 Executando seleção automática de parâmetros...")
        
        best_aic = float('inf')
        best_order = None
        best_seasonal_order = None
        
        # Preparar dados
        y_train = self.train_data[self.target_col]
        exog_vars = ['TAXA_SELIC', 'IPCA', 'TAXA_DESOCUPACAO', 'INADIMPLENCIA']
        exog_train = None
        
        available_vars = [var for var in exog_vars if var in self.train_data.columns]
        if available_vars:
            exog_train = self.train_data[available_vars]
        
        # Grid search
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if seasonal:
                        for P in range(max_P + 1):
                            for D in range(max_D + 1):
                                for Q in range(max_Q + 1):
                                    try:
                                        model = SARIMAX(
                                            y_train,
                                            exog=exog_train,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, 12),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False
                                        )
                                        fitted = model.fit(disp=False)
                                        
                                        if fitted.aic < best_aic:
                                            best_aic = fitted.aic
                                            best_order = (p, d, q)
                                            best_seasonal_order = (P, D, Q, 12)
                                            
                                    except:
                                        continue
                    else:
                        try:
                            model = SARIMAX(
                                y_train,
                                exog=exog_train,
                                order=(p, d, q),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            fitted = model.fit(disp=False)
                            
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                                best_seasonal_order = None
                                
                        except:
                            continue
        
        print(f"✅ Melhor modelo encontrado:")
        print(f"  Ordem: {best_order}")
        print(f"  Sazonal: {best_seasonal_order}")
        print(f"  AIC: {best_aic:.2f}")
        
        return best_order, best_seasonal_order

def train_sarimax_model(train_data, test_data, target_col='TOTAL_CASOS',
                       order=(1,1,1), seasonal_order=(1,1,1,12),
                       exog_vars=None, auto_arima=False):
    """
    Função utilitária para treinar modelo SARIMAX
    
    Args:
        train_data (pd.DataFrame): Dados de treino
        test_data (pd.DataFrame): Dados de teste
        target_col (str): Nome da coluna target
        order (tuple): Ordem ARIMA
        seasonal_order (tuple): Ordem sazonal
        exog_vars (list): Variáveis exógenas
        auto_arima (bool): Se deve usar seleção automática
        
    Returns:
        SARIMAXModel: Modelo treinado
    """
    model = SARIMAXModel(train_data, test_data, target_col)
    
    if auto_arima:
        best_order, best_seasonal_order = model.auto_arima()
        model.train(order=best_order, seasonal_order=best_seasonal_order, exog_vars=exog_vars)
    else:
        model.train(order=order, seasonal_order=seasonal_order, exog_vars=exog_vars)
    
    return model

if __name__ == "__main__":
    # Exemplo de uso
    print("📊 Exemplo de uso do SARIMAXModel")
    print("Execute este módulo através do script principal ou notebook.")

