# -*- coding: utf-8 -*-
"""
Script de Previsão Futura - Projeto TJGO
Usa o melhor modelo (Prophet) para fazer previsões dos próximos meses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os

# Optional imports
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False
    Prophet = None

warnings.filterwarnings('ignore')

class FutureForecaster:
    """
    Classe para fazer previsões futuras usando o melhor modelo
    """
    
    def __init__(self, data_path='./data/processed_test/data_processed_test.csv'):
        """
        Inicializa o previsor
        
        Args:
            data_path (str): Caminho para os dados processados
        """
        self.data_path = data_path
        self.data = None
        self.model = None
        self.forecast = None
        
    def load_data(self):
        """
        Carrega os dados processados
        """
        print("📊 Carregando dados processados...")
        
        try:
            self.data = pd.read_csv(self.data_path, index_col='DATA', parse_dates=True)
            print(f"✅ Dados carregados: {len(self.data)} observações")
            print(f"Período: {self.data.index.min().strftime('%Y-%m')} a {self.data.index.max().strftime('%Y-%m')}")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return False
    
    def train_best_model(self):
        """
        Treina o melhor modelo (Prophet) com todos os dados disponíveis
        """
        if not HAS_PROPHET:
            print("❌ Prophet não disponível. Instale: pip install prophet")
            return False
            
        print("🔄 Treinando modelo Prophet (melhor modelo)...")
        
        try:
            # Preparar dados para Prophet
            prophet_data = self.data.reset_index()
            prophet_data = prophet_data.rename(columns={'DATA': 'ds', 'TOTAL_CASOS': 'y'})
            
            # Inicializar modelo Prophet
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive',
                interval_width=0.95  # Intervalo de confiança de 95%
            )
            
            # Adicionar variáveis exógenas (apenas econômicas tradicionais)
            exog_vars = ['TAXA_SELIC', 'IPCA', 'TAXA_DESOCUPACAO', 'INADIMPLENCIA']
            
            for var in exog_vars:
                if var in prophet_data.columns:
                    self.model.add_regressor(var)
                    print(f"  ✅ Adicionada variável exógena: {var}")
            
            # Treinar modelo com todos os dados
            self.model.fit(prophet_data)
            print("✅ Modelo Prophet treinado com sucesso!")
            return True
            
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            return False
    
    def make_forecast(self, months_ahead=12):
        """
        Faz previsões para os próximos meses
        
        Args:
            months_ahead (int): Número de meses para prever
        """
        if self.model is None:
            print("❌ Modelo não treinado. Execute train_best_model() primeiro.")
            return None
            
        print(f"🔮 Fazendo previsões para os próximos {months_ahead} meses...")
        
        try:
            # Criar dataframe futuro
            future = self.model.make_future_dataframe(periods=months_ahead, freq='MS')
            
            # Adicionar variáveis exógenas para o período futuro
            # Para simplicidade, vamos usar os últimos valores conhecidos
            exog_vars = ['TAXA_SELIC', 'IPCA', 'TAXA_DESOCUPACAO', 'INADIMPLENCIA']
            
            for var in exog_vars:
                if var in self.data.columns:
                    # Usar o último valor conhecido para todas as previsões futuras
                    last_value = self.data[var].iloc[-1]
                    future[var] = last_value
                    print(f"  📈 {var}: usando último valor = {last_value:.2f}")
            
            # Fazer previsões
            self.forecast = self.model.predict(future)
            
            # Extrair apenas as previsões futuras
            future_forecast = self.forecast.tail(months_ahead)
            
            print("✅ Previsões geradas com sucesso!")
            print(f"Período das previsões: {future_forecast['ds'].iloc[0].strftime('%Y-%m')} a {future_forecast['ds'].iloc[-1].strftime('%Y-%m')}")
            
            return future_forecast
            
        except Exception as e:
            print(f"❌ Erro na previsão: {e}")
            return None
    
    def plot_forecast(self, months_ahead=12, save_plot=True):
        """
        Plota as previsões com dados históricos
        
        Args:
            months_ahead (int): Número de meses para mostrar
            save_plot (bool): Se deve salvar o gráfico
        """
        if self.forecast is None:
            print("❌ Nenhuma previsão disponível. Execute make_forecast() primeiro.")
            return
            
        print("📈 Gerando visualização das previsões...")
        
        try:
            # Configurar estilo
            plt.style.use('seaborn-v0_8')
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plotar dados históricos
            ax.plot(self.data.index, self.data['TOTAL_CASOS'], 
                   label='Dados Históricos', linewidth=2, color='blue', alpha=0.8)
            
            # Plotar previsões
            future_data = self.forecast.tail(months_ahead)
            ax.plot(future_data['ds'], future_data['yhat'], 
                   label='Previsão', linewidth=3, color='red', linestyle='--')
            
            # Plotar intervalo de confiança
            ax.fill_between(future_data['ds'], 
                           future_data['yhat_lower'], 
                           future_data['yhat_upper'],
                           alpha=0.3, color='red', label='Intervalo de Confiança 95%')
            
            # Configurar gráfico
            ax.set_title('Previsão de Casos TJGO - Próximos 12 Meses', fontsize=16, fontweight='bold')
            ax.set_xlabel('Data', fontsize=12)
            ax.set_ylabel('Total de Casos', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Rotacionar labels do eixo x
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_plot:
                os.makedirs('./reports_test', exist_ok=True)
                plt.savefig('./reports_test/forecast_future.png', dpi=300, bbox_inches='tight')
                print("✅ Gráfico salvo em ./reports_test/forecast_future.png")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Erro ao gerar gráfico: {e}")
    
    def save_forecast(self, months_ahead=12, filename='forecast_results.csv'):
        """
        Salva as previsões em CSV
        
        Args:
            months_ahead (int): Número de meses para salvar
            filename (str): Nome do arquivo
        """
        if self.forecast is None:
            print("❌ Nenhuma previsão disponível.")
            return
            
        print(f"💾 Salvando previsões em {filename}...")
        
        try:
            # Extrair previsões futuras
            future_data = self.forecast.tail(months_ahead)
            
            # Criar DataFrame com informações relevantes
            forecast_df = pd.DataFrame({
                'Data': future_data['ds'],
                'Previsao_Casos': future_data['yhat'].round(0),
                'Limite_Inferior': future_data['yhat_lower'].round(0),
                'Limite_Superior': future_data['yhat_upper'].round(0),
                'Intervalo_Confianca': (future_data['yhat_upper'] - future_data['yhat_lower']).round(0)
            })
            
            # Salvar
            os.makedirs('./reports_test', exist_ok=True)
            filepath = f'./reports_test/{filename}'
            forecast_df.to_csv(filepath, index=False)
            
            print(f"✅ Previsões salvas em {filepath}")
            print("\n📊 Resumo das Previsões:")
            print(forecast_df.to_string(index=False))
            
            return forecast_df
            
        except Exception as e:
            print(f"❌ Erro ao salvar: {e}")
            return None
    
    def generate_insights(self, months_ahead=12):
        """
        Gera insights das previsões
        
        Args:
            months_ahead (int): Número de meses analisados
        """
        if self.forecast is None:
            print("❌ Nenhuma previsão disponível.")
            return
            
        print("\n🔍 INSIGHTS DAS PREVISÕES:")
        print("="*50)
        
        try:
            future_data = self.forecast.tail(months_ahead)
            
            # Estatísticas básicas
            mean_forecast = future_data['yhat'].mean()
            min_forecast = future_data['yhat'].min()
            max_forecast = future_data['yhat'].max()
            
            print(f"📈 Média prevista: {mean_forecast:.0f} casos/mês")
            print(f"📉 Mínimo previsto: {min_forecast:.0f} casos")
            print(f"📊 Máximo previsto: {max_forecast:.0f} casos")
            
            # Comparar com dados históricos
            historical_mean = self.data['TOTAL_CASOS'].mean()
            historical_std = self.data['TOTAL_CASOS'].std()
            
            print(f"\n📊 Comparação com histórico:")
            print(f"  Média histórica: {historical_mean:.0f} casos/mês")
            print(f"  Desvio padrão histórico: {historical_std:.0f}")
            
            # Tendência
            first_forecast = future_data['yhat'].iloc[0]
            last_forecast = future_data['yhat'].iloc[-1]
            trend = last_forecast - first_forecast
            
            if trend > 0:
                print(f"📈 Tendência: AUMENTO de {trend:.0f} casos ao longo do período")
            elif trend < 0:
                print(f"📉 Tendência: DIMINUIÇÃO de {abs(trend):.0f} casos ao longo do período")
            else:
                print("➡️ Tendência: ESTÁVEL")
            
            # Mês com maior previsão
            max_month_idx = future_data['yhat'].idxmax()
            max_month = future_data.loc[max_month_idx, 'ds']
            max_value = future_data.loc[max_month_idx, 'yhat']
            
            print(f"\n🎯 Mês com maior previsão: {max_month.strftime('%Y-%m')} ({max_value:.0f} casos)")
            
        except Exception as e:
            print(f"❌ Erro ao gerar insights: {e}")

def main():
    """
    Função principal para executar previsões futuras
    """
    print("🔮 PREVISÕES FUTURAS - PROJETO TJGO")
    print("="*50)
    print("Usando o melhor modelo: Prophet (modelo teste)")
    print("Configuração: dados 2015+, variáveis econômicas tradicionais")
    print("="*50)
    
    # Inicializar previsor
    forecaster = FutureForecaster()
    
    # Carregar dados
    if not forecaster.load_data():
        return
    
    # Treinar modelo
    if not forecaster.train_best_model():
        return
    
    # Fazer previsões para 12 meses
    forecast_data = forecaster.make_forecast(months_ahead=12)
    
    if forecast_data is not None:
        # Gerar visualização
        forecaster.plot_forecast(months_ahead=12)
        
        # Salvar resultados
        forecaster.save_forecast(months_ahead=12)
        
        # Gerar insights
        forecaster.generate_insights(months_ahead=12)
        
        print("\n✅ Previsões futuras concluídas com sucesso!")
        print("📁 Arquivos gerados em ./reports_test/")
        print("  - forecast_future.png (gráfico)")
        print("  - forecast_results.csv (dados)")
    
    return forecaster

if __name__ == "__main__":
    forecaster = main()
