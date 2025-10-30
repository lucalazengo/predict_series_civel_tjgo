# -*- coding: utf-8 -*-
"""
Classe Base para Modelos de Forecasting
Projeto TJGO - Forecasting de Casos

Esta classe contém métodos comuns para todos os modelos de forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')

class BaseForecastingModel:
    """
    Classe base para modelos de forecasting
    """
    
    def __init__(self, train_data, test_data, target_col='TOTAL_CASOS'):
        """
        Inicializa o modelo base
        
        Args:
            train_data (pd.DataFrame): Dados de treino
            test_data (pd.DataFrame): Dados de teste
            target_col (str): Nome da coluna target
        """
        self.train_data = train_data
        self.test_data = test_data
        self.target_col = target_col
        self.model = None
        self.predictions = None
        self.metrics = {}
        
    def prepare_features(self, data, include_target=True):
        """
        Prepara features para modelos de ML
        
        Args:
            data (pd.DataFrame): Dados
            include_target (bool): Se deve incluir target nas features
            
        Returns:
            pd.DataFrame: Features preparadas
        """
        # Selecionar features numéricas
        feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not include_target and self.target_col in feature_cols:
            feature_cols.remove(self.target_col)
        
        # Remover colunas com muitos NaN
        feature_cols = [col for col in feature_cols 
                       if data[col].isnull().sum() < len(data) * 0.5]
        
        return data[feature_cols].fillna(0)
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calcula métricas de avaliação
        
        Args:
            y_true (array): Valores reais
            y_pred (array): Valores previstos
            
        Returns:
            dict: Dicionário com métricas
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def plot_predictions(self, model_name, save_path=None):
        """
        Plota previsões vs valores reais
        
        Args:
            model_name (str): Nome do modelo
            save_path (str): Caminho para salvar o gráfico
        """
        if self.predictions is None:
            print("❌ Nenhuma previsão disponível. Execute o modelo primeiro.")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plotar dados reais e previsões
        plt.plot(self.test_data.index, self.test_data[self.target_col], 
                label='Real', linewidth=2, color='blue')
        plt.plot(self.test_data.index, self.predictions, 
                label='Previsão', linewidth=2, color='red', linestyle='--')
        
        plt.title(f"{model_name} - Previsões vs Real (MAE: {self.metrics['mae']:.0f})", 
                 fontweight='bold', fontsize=14)
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('TOTAL_CASOS', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico salvo em {save_path}")
        
        plt.show()
    
    def plot_residuals(self, model_name, save_path=None):
        """
        Plota resíduos do modelo
        
        Args:
            model_name (str): Nome do modelo
            save_path (str): Caminho para salvar o gráfico
        """
        if self.predictions is None:
            print("❌ Nenhuma previsão disponível. Execute o modelo primeiro.")
            return
        
        residuals = self.test_data[self.target_col] - self.predictions
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gráfico de resíduos vs tempo
        axes[0].plot(self.test_data.index, residuals, 'o-', alpha=0.7)
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_title(f"{model_name} - Resíduos vs Tempo")
        axes[0].set_xlabel('Data')
        axes[0].set_ylabel('Resíduos')
        axes[0].grid(True, alpha=0.3)
        
        # Histograma dos resíduos
        axes[1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='red', linestyle='--')
        axes[1].set_title(f"{model_name} - Distribuição dos Resíduos")
        axes[1].set_xlabel('Resíduos')
        axes[1].set_ylabel('Frequência')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico de resíduos salvo em {save_path}")
        
        plt.show()
    
    def save_results(self, model_name, save_dir='./reports_test'):
        """
        Salva resultados do modelo
        
        Args:
            model_name (str): Nome do modelo
            save_dir (str): Diretório para salvar
        """
        if self.predictions is None:
            print("❌ Nenhuma previsão disponível. Execute o modelo primeiro.")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Salvar previsões
        results_df = pd.DataFrame({
            'Data': self.test_data.index,
            'Real': self.test_data[self.target_col],
            'Previsao': self.predictions,
            'Residuo': self.test_data[self.target_col] - self.predictions
        })
        
        results_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_results.csv')
        results_df.to_csv(results_path, index=False)
        
        # Salvar métricas
        metrics_df = pd.DataFrame([self.metrics])
        metrics_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"✅ Resultados salvos em {save_dir}")
        print(f"  - Previsões: {results_path}")
        print(f"  - Métricas: {metrics_path}")
    
    def get_feature_importance(self):
        """
        Retorna importância das features (se disponível)
        
        Returns:
            pd.DataFrame: Importância das features
        """
        if hasattr(self.model, 'feature_importances_'):
            features = self.prepare_features(self.train_data, include_target=False).columns
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            return importance_df
        else:
            print("❌ Modelo não possui feature importance.")
            return None
    
    def plot_feature_importance(self, model_name, top_n=10, save_path=None):
        """
        Plota importância das features
        
        Args:
            model_name (str): Nome do modelo
            top_n (int): Número de features para mostrar
            save_path (str): Caminho para salvar o gráfico
        """
        importance_df = self.get_feature_importance()
        
        if importance_df is None:
            return
        
        # Pegar top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_features, x='Importance', y='Feature')
        plt.title(f"{model_name} - Importância das Features (Top {top_n})", 
                 fontweight='bold', fontsize=14)
        plt.xlabel('Importância', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico de importância salvo em {save_path}")
        
        plt.show()
    
    def print_summary(self, model_name):
        """
        Imprime resumo do modelo
        
        Args:
            model_name (str): Nome do modelo
        """
        print(f"\n📊 RESUMO DO MODELO: {model_name}")
        print("="*50)
        print(f"MAE:  {self.metrics['mae']:.2f}")
        print(f"RMSE: {self.metrics['rmse']:.2f}")
        print(f"R²:   {self.metrics['r2']:.3f}")
        print("="*50)
