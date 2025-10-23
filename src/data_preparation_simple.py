# -*- coding: utf-8 -*-
"""
Data Preparation Script for TJGO Forecasting Project
"""

import pandas as pd
import numpy as np
import os

def main():
    print("Starting data preparation...")
    
    # Load data
    data_path = './data/raw/base_consolidada_mensal_clean.csv'
    df = pd.read_csv(data_path)
    df['DATA'] = pd.to_datetime(df['DATA'])
    df = df.set_index('DATA').sort_index()
    
    print("Data loaded: " + str(df.shape[0]) + " observations, " + str(df.shape[1]) + " variables")
    print("Period: " + df.index.min().strftime('%Y-%m') + " to " + df.index.max().strftime('%Y-%m'))
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    print("Missing values handled: " + str(df.isnull().sum().sum()) + " remaining")
    
    # Create time features
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    
    # Create lag features
    for lag in [1, 2, 3, 6, 12]:
        df['TOTAL_CASOS_lag_' + str(lag)] = df['TOTAL_CASOS'].shift(lag)
    
    # Create rolling features
    for window in [3, 6, 12]:
        df['TOTAL_CASOS_rolling_mean_' + str(window)] = df['TOTAL_CASOS'].rolling(window=window).mean()
        df['TOTAL_CASOS_rolling_std_' + str(window)] = df['TOTAL_CASOS'].rolling(window=window).std()
    
    # Remove rows with NaN (after creating features)
    df = df.dropna()
    
    # Split data temporally
    n_total = len(df)
    n_test = int(n_total * 0.2)
    n_train = n_total - n_test
    
    train_data = df.iloc[:n_train].copy()
    test_data = df.iloc[n_train:].copy()
    
    print("Train data: " + str(len(train_data)) + " observations")
    print("Test data: " + str(len(test_data)) + " observations")
    
    # Create output directory
    os.makedirs('./data/processed/', exist_ok=True)
    
    # Save processed data
    df.to_csv('./data/processed/data_processed.csv')
    train_data.to_csv('./data/processed/train.csv')
    test_data.to_csv('./data/processed/test.csv')
    
    print("Data preparation completed successfully!")
    print("Total features created: " + str(len(df.columns)))
    
    return df, train_data, test_data

if __name__ == "__main__":
    df, train_data, test_data = main()