# -*- coding: utf-8 -*-
"""
Data Preparation Script for TJGO Forecasting Project - TEST VERSION
This version:
- Excludes 2014 data (starts from 2015)
- Removes high correlation variables (qt_acidente, QT_ELEITOR)
- Uses only traditional economic variables
"""

import pandas as pd
import numpy as np
import os

def main():
    print("Starting TEST data preparation...")
    print(" TEST CONFIGURATION:")
    print("  - Excluding 2014 data")
    print("  - Removing qt_acidente and QT_ELEITOR")
    print("  - Using only traditional economic variables")
    
    # Load data
    data_path = './data/raw/base_consolidada_mensal_clean.csv'
    df = pd.read_csv(data_path)
    df['DATA'] = pd.to_datetime(df['DATA'])
    df = df.set_index('DATA').sort_index()
    
    print("\nData loaded: " + str(df.shape[0]) + " observations, " + str(df.shape[1]) + " variables")
    print("Period: " + df.index.min().strftime('%Y-%m') + " to " + df.index.max().strftime('%Y-%m'))
    
    # Remove 2014 data (start from 2015)
    df = df[df.index >= '2015-01-01']
    print("After removing 2014: " + str(len(df)) + " observations")
    print("New period: " + df.index.min().strftime('%Y-%m') + " to " + df.index.max().strftime('%Y-%m'))
    
    # Remove high correlation variables
    variables_to_remove = ['qt_acidente', 'QT_ELEITOR']
    df = df.drop(columns=variables_to_remove)
    print("Removed variables: " + str(variables_to_remove))
    print("Remaining variables: " + str(list(df.columns)))
    
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
    
    # Handle missing values more carefully to preserve more data
    # Fill forward for lag features (use last known value)
    for lag in [1, 2, 3, 6, 12]:
        df['TOTAL_CASOS_lag_' + str(lag)] = df['TOTAL_CASOS_lag_' + str(lag)].fillna(method='ffill')
    
    # Fill forward for rolling statistics
    for window in [3, 6, 12]:
        df['TOTAL_CASOS_rolling_mean_' + str(window)] = df['TOTAL_CASOS_rolling_mean_' + str(window)].fillna(method='ffill')
        df['TOTAL_CASOS_rolling_std_' + str(window)] = df['TOTAL_CASOS_rolling_std_' + str(window)].fillna(method='ffill')
    
    # Handle any remaining missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Only remove rows that still have NaN (should be minimal now)
    df = df.dropna()
    
    # Split data temporally
    n_total = len(df)
    n_test = int(n_total * 0.2)
    n_train = n_total - n_test
    
    train_data = df.iloc[:n_train].copy()
    test_data = df.iloc[n_train:].copy()
    
    print("\nTrain data: " + str(len(train_data)) + " observations (" + train_data.index.min().strftime('%Y-%m') + " to " + train_data.index.max().strftime('%Y-%m') + ")")
    print("Test data: " + str(len(test_data)) + " observations (" + test_data.index.min().strftime('%Y-%m') + " to " + test_data.index.max().strftime('%Y-%m') + ")")
    
    # Create output directory for test data
    os.makedirs('./data/processed_test/', exist_ok=True)
    
    # Save processed data with TEST suffix
    df.to_csv('./data/processed_test/data_processed_test.csv')
    train_data.to_csv('./data/processed_test/train_test.csv')
    test_data.to_csv('./data/processed_test/test_test.csv')
    
    print("\n✅ TEST data preparation completed successfully!")
    print("Total features created: " + str(len(df.columns)))
    print("Files saved in: ./data/processed_test/")
    
    return df, train_data, test_data

if __name__ == "__main__":
    df, train_data, test_data = main()
