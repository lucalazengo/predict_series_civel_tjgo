# Forecasting Project - TJGO Cases

This project implements a case forecasting system for the Court of Justice of Goiás (TJGO) following the CRISP-DM methodology. **Key finding**: Simpler models outperformed complex approaches, with Prophet achieving an MAE of 3,634 cases.

## Executive Summary

### Key Results

- **Winning Model**: Prophet (data from 2015+, traditional economic variables)
- **Performance**: MAE = 3,634 cases, R² = 0.339 (excellent fit)
- **Critical Insight**: Fewer variables = better performance (principle of parsimony)
- **2025 Forecast**: Average of 58,887 cases/month with a downward trend

### Surprising Discovery

The **test model** (excluding 2014 data + excluding highly correlated variables) achieved **44% better performance** than the full model, demonstrating that:

- **Simplicity beats complexity**
- **4 well-chosen variables > 15 variables**
- **Data quality > Quantity**

## Data and Methodology

### Data Sources

- **Period**: January 2014 – December 2024 (132 months)
- **Frequency**: Monthly
- **Target Variable**: `TOTAL_CASOS` (new cases per month)
- **Exogenous Variables**: 15 economic and social indicators

### Success Metrics

- **MAE < 5,000 cases** (acceptable average error) ✅ **3,634 cases**
- **R² > 0.3** (explains at least 30% of variance) ✅ **0.339**
- **Reproducibility** (modular and documented code) ✅

## Project Structure

```
predict_series_civel_tjgo/
├── data/
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data (full model)
│   └── processed_test/         # Processed data (test model)
├── notebooks/                  # Jupyter notebooks
├── src/                        # Source code
│   ├── data_preparation.py    # Full data preparation
│   ├── data_preparation_test.py # Test data preparation
│   ├── train_models.py        # Full model training
│   ├── train_models_test.py   # Test model training
│   └── forecast_future.py    # Future forecasts
├── reports/                    # Full model reports
├── reports_test/              # Test model reports
├── requirements.txt            # Dependencies
├── RELATORIO_TECNICO_FINAL.md # Full technical report
└── README.md                   # This file
```

## Installation and Execution

### 1. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv_tjgo

# Activate environment (Linux/Mac)
source venv_tjgo/bin/activate

# Activate environment (Windows)
venv_tjgo\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Full Analysis

```bash
# 1. Exploratory Data Analysis
jupyter notebook notebooks/01_EDA.ipynb

# 2. Data Preparation (Full Model)
python src/data_preparation.py

# 3. Data Preparation (Test Model – RECOMMENDED)
python src/data_preparation_test.py

# 4. Model Training (Full Model)
python src/train_models.py

# 5. Model Training (Test Model – RECOMMENDED)
python src/train_models_test.py

# 6. Future Forecasts (Using Best Model)
python src/forecast_future.py
```

## Implemented Models and Results

### Performance Comparison

| Model                    | MAE             | RMSE            | R²             | Status             |
| ------------------------ | --------------- | --------------- | --------------- | ------------------ |
| **Prophet (Test)** | **3,634** | **4,597** | **0.339** | 🏆**WINNER** |
| Prophet (Full)           | 6,472           | 7,313           | -0.245          | ❌ Overfitting     |
| Random Forest            | 6,827           | 7,874           | -0.939          | ❌                 |
| XGBoost                  | 7,669           | 8,918           | -1.487          | ❌                 |
| LightGBM                 | 7,464           | 8,876           | -1.464          | ❌                 |
| SARIMAX                  | 9,416           | 11,290          | -2.986          | ❌                 |

### 2025 Forecasts

| Month              | Forecast         | Lower Bound      | Upper Bound      |
| ------------------ | ---------------- | ---------------- | ---------------- |
| Jan/2025           | 56,445           | 47,054           | 65,747           |
| Feb/2025           | 54,613           | 46,148           | 63,639           |
| Mar/2025           | 62,186           | 53,690           | 70,522           |
| **Jul/2025** | **63,158** | **53,992** | **72,300** |
| **Dec/2025** | **53,908** | **45,317** | **62,664** |

**Insights**:

- **Average**: 58,887 cases/month (+38.9% vs historical average)
- **Peak**: July (63,158 cases)
- **Trough**: December (53,908 cases)
- **Trend**: Decrease of 2,537 cases throughout the year

## Key Findings

### 1. **Principle of Parsimony**

> "Among competing hypotheses that predict equally well, the one with the fewest assumptions should be selected."

**Application**: The model with 4 traditional economic variables outperformed the model with 15 variables.

### 2. **Curse of Dimensionality**

**Technical Explanation**: In high-dimensional spaces, all points become equidistant, making learning difficult.

### 3. **Highly Correlated Variables = Noise**

- `qt_acidente` and `QT_ELEITOR` showed correlations >0.85 with `TOTAL_CASOS`
- **But reduced performance** when included
- **Cause**: Multicollinearity and overfitting

### 4. **Traditional Economic Variables Are Sufficient**

- **TAXA_SELIC**: Base interest rate
- **IPCA**: Consumer price index
- **TAXA_DESOCUPACAO**: Unemployment rate
- **INADIMPLENCIA**: Default rate

## Generated Files

### Reports and Metrics

- `reports/metrics.csv` – Full model metrics
- `reports_test/metrics_test.csv` – Test model metrics (**RECOMMENDED**)
- `reports_test/forecast_results.csv` – Detailed future forecasts

### Visualizations

- `reports/predictions_comparison.png` – Full model comparison
- `reports_test/predictions_comparison_test.png` – Test model comparison
- `reports_test/forecast_future.png` – Future forecasts with confidence intervals

### Documentation

- `RELATORIO_TECNICO_FINAL.md` – Full technical report
- `ANALISE_CORRELACAO.md` – Correlation analysis
- `EXECUTIVE_SUMMARY.md` – Executive summary
- `CHECKLIST.md` – Progress checklist

## Technologies Used

- **Python 3.11** – Primary language
- **Pandas, NumPy** – Data manipulation
- **Matplotlib, Seaborn** – Visualizations
- **Prophet** – Time series modeling (Facebook)
- **Scikit-learn** – Machine learning
- **XGBoost, LightGBM** – Gradient boosting
- **Statsmodels** – Statistical models (SARIMAX)
- **Jupyter** – Exploratory analysis

## CRISP-DM Methodology

1. **Business Understanding** ✅ – TJGO business context
2. **Data Understanding** ✅ – Comprehensive EDA with 1,892 lines of analysis
3. **Data Preparation** ✅ – Cleaning, feature engineering, experiments
4. **Modeling** ✅ – 7 algorithms tested
5. **Evaluation** ✅ – Rigorous metrics, statistical tests
6. **Deployment** ✅ – Modular code, complete documentation

## Implementation Recommendations

### Immediate Implementation

1. **Deploy Prophet model** (2015+ data, 4 economic variables)
2. **Retrain monthly** with new data
3. **Monitor performance** with automated alerts
4. **Build executive dashboard** with key KPIs

### Future Expansion

1. **Other case types** (criminal, family, etc.)
2. **Forecast by jurisdiction** (geographic)
3. **Other courts** (replicable methodology)
4. **AutoML** for automated optimization

## Lessons Learned

### Successes

- **Simplicity beats complexity**
- **Data quality > Quantity**
- **Temporal validation is crucial**
- **Feature engineering is fundamental**

### Cautions

- **Overfitting** with too many variables
- **Multicollinearity** among features
- **Data drift** over time
- **Stakeholder expectation management**

## Team and Contact

- **Autor** - Eng. Manuel Lucala Zengo
- **Mentorship** – UFG TI RESIDENCY
- **Team** - DIACDE TJGO
- **Methodology** – CRISP-DM adapted for time series

## Next Steps

1. **Implement monitoring system**
3. **Create executive dashboard**
4. **Train technical team**
5. **Establish monthly update process**
