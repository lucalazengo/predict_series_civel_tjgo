# -*- coding: utf-8 -*-
"""
Script de Preparacao de Dados
Projeto de Forecasting - Casos TJGO

Este script implementa todas as transformacoes necessarias para preparar
os dados para modelagem de series temporais, seguindo as melhores praticas
de engenharia de features para forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreparator:
    """
    Classe para preparação de dados de séries temporais
    """
    
    def __init__(self, data_path):
        """
        Inicializa o preparador de dados
        
        Args:
            data_path (str): Caminho para o arquivo de dados
        """
        self.data_path = data_path
        self.df = None
        self.train_data = None
        self.test_data = None
        self.scaler = None
        
    def load_data(self):
        """
        Carrega e prepara os dados básicos
        """
        print("📊 Carregando dados...")
        self.df = pd.read_csv(self.data_path)
        self.df['DATA'] = pd.to_datetime(self.df['DATA'])
        self.df = self.df.set_index('DATA').sort_index()
        
        print(f"✅ Dados carregados: {self.df.shape[0]} observacoes, {self.df.shape[1]} variaveis")
        print(f"📅 Periodo: {self.df.index.min().strftime('%Y-%m')} a {self.df.index.max().strftime('%Y-%m')}")
        
        return self.df
    
    def handle_missing_values(self):
        """
        Trata valores faltantes
        """
        print("\n🔍 Tratando valores faltantes...")
        
        missing_before = self.df.isnull().sum().sum()
        print(f"Valores faltantes antes: {missing_before}")
        
        # Estratégia por tipo de variável
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if col == 'TOTAL_CASOS':
                    # Para target, usar interpolação linear
                    self.df[col] = self.df[col].interpolate(method='linear')
                else:
                    # Para exógenas, usar forward fill seguido de backward fill
                    self.df[col] = self.df[col].fillna(method='ffill').fillna(method='bfill')
        
        missing_after = self.df.isnull().sum().sum()
        print(f"Valores faltantes depois: {missing_after}")
        
        return self.df
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """
        Detecta outliers usando diferentes métodos
        
        Args:
            method (str): Método para detecção ('iqr', 'zscore', 'isolation')
            threshold (float): Threshold para detecção
        """
        print(f"\n🔍 Detectando outliers (método: {method})...")
        
        outliers_info = {}
        
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers = self.df[z_scores > threshold]
            
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(self.df) * 100
            }
        
        # Exibir resumo
        outliers_df = pd.DataFrame(outliers_info).T
        outliers_df = outliers_df.sort_values('count', ascending=False)
        print("Outliers por variável:")
        print(outliers_df)
        
        return outliers_info
    
    def create_time_features(self):
        """
        Cria features temporais
        """
        print("\n📅 Criando features temporais...")
        
        # Features básicas de tempo
        self.df['year'] = self.df.index.year
        self.df['month'] = self.df.index.month
        self.df['quarter'] = self.df.index.quarter
        self.df['day_of_year'] = self.df.index.dayofyear
        self.df['week_of_year'] = self.df.index.isocalendar().week
        
        # Features cíclicas (importantes para sazonalidade)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['quarter_sin'] = np.sin(2 * np.pi * self.df['quarter'] / 4)
        self.df['quarter_cos'] = np.cos(2 * np.pi * self.df['quarter'] / 4)
        
        # Dummies sazonais
        for month in range(1, 13):
            self.df[f'month_{month}'] = (self.df['month'] == month).astype(int)
        
        for quarter in range(1, 5):
            self.df[f'quarter_{quarter}'] = (self.df['quarter'] == quarter).astype(int)
        
        print(f"✅ Features temporais criadas: {len([col for col in self.df.columns if col.startswith(('month', 'quarter', 'year', 'day', 'week'))])} features")
        
        return self.df
    
    def create_lag_features(self, target_col='TOTAL_CASOS', lags=[1, 2, 3, 6, 12]):
        """
        Cria features de defasagem (lags)
        
        Args:
            target_col (str): Nome da coluna target
            lags (list): Lista de defasagens para criar
        """
        print(f"\n⏰ Criando features de defasagem para {target_col}...")
        
        for lag in lags:
            self.df[f'{target_col}_lag_{lag}'] = self.df[target_col].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            self.df[f'{target_col}_rolling_mean_{window}'] = self.df[target_col].rolling(window=window).mean()
            self.df[f'{target_col}_rolling_std_{window}'] = self.df[target_col].rolling(window=window).std()
            self.df[f'{target_col}_rolling_min_{window}'] = self.df[target_col].rolling(window=window).min()
            self.df[f'{target_col}_rolling_max_{window}'] = self.df[target_col].rolling(window=window).max()
        
        # Diferenças
        self.df[f'{target_col}_diff_1'] = self.df[target_col].diff(1)
        self.df[f'{target_col}_diff_12'] = self.df[target_col].diff(12)
        
        print(f"✅ Features de defasagem criadas: {len(lags) + 8} features")
        
        return self.df
    
    def create_exogenous_lags(self, exogenous_vars=None, lags=[1, 2, 3, 6]):
        """
        Cria defasagens para variáveis exógenas
        
        Args:
            exogenous_vars (list): Lista de variáveis exógenas
            lags (list): Lista de defasagens
        """
        if exogenous_vars is None:
            # Selecionar variáveis exógenas baseado na correlação
            # Incluindo variáveis de alta correlação identificadas no EDA
            exogenous_vars = ['TAXA_SELIC', 'IPCA', 'TAXA_DESOCUPACAO', 'INADIMPLENCIA', 
                            'qt_acidente', 'QT_ELEITOR']
        
        print(f"\n🔄 Criando defasagens para variáveis exógenas: {exogenous_vars}")
        
        for var in exogenous_vars:
            if var in self.df.columns:
                for lag in lags:
                    self.df[f'{var}_lag_{lag}'] = self.df[var].shift(lag)
        
        print(f"✅ Defasagens exógenas criadas: {len(exogenous_vars) * len(lags)} features")
        
        return self.df
    
    def apply_transformations(self, target_col='TOTAL_CASOS'):
        """
        Aplica transformações para normalizar distribuições
        
        Args:
            target_col (str): Nome da coluna target
        """
        print(f"\n🔄 Aplicando transformações para {target_col}...")
        
        # Log transformation (para dados positivos)
        if (self.df[target_col] > 0).all():
            self.df[f'{target_col}_log'] = np.log1p(self.df[target_col])
            print("✅ Transformação log aplicada")
        
        # Box-Cox transformation
        from scipy.stats import boxcox
        
        try:
            # Box-Cox requer dados positivos
            if (self.df[target_col] > 0).all():
                boxcox_data, lambda_param = boxcox(self.df[target_col])
                self.df[f'{target_col}_boxcox'] = boxcox_data
                print(f"✅ Transformação Box-Cox aplicada (λ = {lambda_param:.3f})")
        except:
            print("⚠️  Transformação Box-Cox não aplicada (dados não positivos)")
        
        return self.df
    
    def scale_features(self, method='standard', exclude_cols=None):
        """
        Escala features numéricas
        
        Args:
            method (str): Método de escalonamento ('standard', 'robust', 'minmax')
            exclude_cols (list): Colunas para excluir do escalonamento
        """
        if exclude_cols is None:
            exclude_cols = ['TOTAL_CASOS', 'year', 'month', 'quarter']
        
        print(f"\n📏 Escalonando features (método: {method})...")
        
        # Selecionar colunas numéricas para escalonar
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        
        # Escalonar apenas as features, não o target
        self.df[scale_cols] = self.scaler.fit_transform(self.df[scale_cols])
        
        print(f"✅ {len(scale_cols)} features escalonadas")
        
        return self.df
    
    def split_temporal_data(self, test_size=0.2, validation_size=0.1):
        """
        Divide os dados temporalmente (não aleatoriamente)
        
        Args:
            test_size (float): Proporção para teste
            validation_size (float): Proporção para validação
        """
        print(f"\n✂️  Dividindo dados temporalmente...")
        
        n_total = len(self.df)
        n_test = int(n_total * test_size)
        n_val = int(n_total * validation_size)
        n_train = n_total - n_test - n_val
        
        # Divisão temporal
        train_end = n_train
        val_end = n_train + n_val
        
        self.train_data = self.df.iloc[:train_end].copy()
        self.val_data = self.df.iloc[train_end:val_end].copy() if n_val > 0 else None
        self.test_data = self.df.iloc[val_end:].copy()
        
        print(f"📊 Divisão dos dados:")
        print(f"  Treino: {len(self.train_data)} observações ({self.train_data.index.min().strftime('%Y-%m')} a {self.train_data.index.max().strftime('%Y-%m')})")
        if self.val_data is not None:
            print(f"  Validação: {len(self.val_data)} observações ({self.val_data.index.min().strftime('%Y-%m')} a {self.val_data.index.max().strftime('%Y-%m')})")
        print(f"  Teste: {len(self.test_data)} observações ({self.test_data.index.min().strftime('%Y-%m')} a {self.test_data.index.max().strftime('%Y-%m')})")
        
        return self.train_data, self.val_data, self.test_data
    
    def remove_infinite_values(self):
        """
        Remove valores infinitos e NaN
        """
        print("\n🧹 Removendo valores infinitos e NaN...")
        
        # Substituir infinitos por NaN
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        # Contar valores problemáticos
        inf_count = np.isinf(self.df.select_dtypes(include=[np.number])).sum().sum()
        nan_count = self.df.isnull().sum().sum()
        
        print(f"Valores infinitos: {inf_count}")
        print(f"Valores NaN: {nan_count}")
        
        # Remover linhas com NaN (após criação de features)
        self.df = self.df.dropna()
        
        print(f"✅ Dados limpos: {len(self.df)} observações restantes")
        
        return self.df
    
    def save_processed_data(self, output_dir='../data/processed/'):
        """
        Salva os dados processados
        
        Args:
            output_dir (str): Diretório de saída
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Salvando dados processados em {output_dir}...")
        
        # Salvar dados completos
        self.df.to_csv(f'{output_dir}/data_processed.csv')
        
        # Salvar divisões
        self.train_data.to_csv(f'{output_dir}/train.csv')
        self.test_data.to_csv(f'{output_dir}/test.csv')
        
        if self.val_data is not None:
            self.val_data.to_csv(f'{output_dir}/validation.csv')
        
        # Salvar informações do scaler
        if self.scaler is not None:
            import joblib
            joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        
        print("✅ Dados processados salvos com sucesso!")
        
        return True
    
    def get_feature_importance_info(self):
        """
        Retorna informações sobre as features criadas
        """
        feature_info = {
            'total_features': len(self.df.columns),
            'original_features': len(pd.read_csv(self.data_path).columns),
            'time_features': len([col for col in self.df.columns if col.startswith(('month', 'quarter', 'year', 'day', 'week'))]),
            'lag_features': len([col for col in self.df.columns if 'lag' in col]),
            'rolling_features': len([col for col in self.df.columns if 'rolling' in col]),
            'transformed_features': len([col for col in self.df.columns if any(x in col for x in ['_log', '_boxcox', '_diff'])]),
        }
        
        return feature_info

def main():
    """
    Função principal para executar a preparação de dados
    """
    print("🚀 Iniciando preparação de dados...")
    print("="*50)
    
    # Inicializar preparador
    preparator = DataPreparator('../data/raw/base_consolidada_mensal_clean.csv')
    
    # Pipeline de preparação
    preparator.load_data()
    preparator.handle_missing_values()
    preparator.detect_outliers()
    preparator.create_time_features()
    preparator.create_lag_features()
    preparator.create_exogenous_lags()
    preparator.apply_transformations()
    preparator.scale_features()
    preparator.remove_infinite_values()
    preparator.split_temporal_data()
    preparator.save_processed_data()
    
    # Informações finais
    feature_info = preparator.get_feature_importance_info()
    print(f"\n📊 Resumo da preparação:")
    print(f"  Features originais: {feature_info['original_features']}")
    print(f"  Features totais: {feature_info['total_features']}")
    print(f"  Features temporais: {feature_info['time_features']}")
    print(f"  Features de defasagem: {feature_info['lag_features']}")
    print(f"  Features de rolling: {feature_info['rolling_features']}")
    print(f"  Features transformadas: {feature_info['transformed_features']}")
    
    print("\n✅ Preparação de dados concluída com sucesso!")
    
    return preparator

if __name__ == "__main__":
    preparator = main()
