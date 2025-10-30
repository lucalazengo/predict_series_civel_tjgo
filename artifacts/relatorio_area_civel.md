## Relatório Técnico Detalhado — Área Cível (TJGO)

### Objetivo
Consolidar e aprofundar, com rigor técnico, tudo que foi desenvolvido no projeto de previsão de novos casos cíveis do TJGO, descrevendo dados, métodos, modelos, variáveis exógenas, resultados, justificativas e artefatos gerados. Este relatório integra os conteúdos originais existentes e os expande com trechos de código, tabelas e referências a gráficos/outputs.

---

## 1. Conteúdo original integrado (base de referência)

Nesta seção apresentamos trechos essenciais dos documentos originais que servem de base. Em seguida, cada tema é expandido com detalhes técnicos, códigos e resultados.

- Resumo Executivo (original):

> “Modelo mais simples superou abordagens complexas, com Prophet alcançando MAE de 3.634 casos (44% melhor que o modelo completo). Variáveis econômicas tradicionais foram suficientes, enquanto variáveis de alta correlação reduziram performance.”

- Resultados consolidados (original):

> Modelo vencedor: Prophet (dados 2015+, 4 variáveis econômicas tradicionais). MAE = 3.634; R² = 0.339. Previsão média 2025: 58.887 casos/mês, com tendência de diminuição.

- Diretrizes sobre variáveis exógenas (original):

> “Variáveis de alta correlação como `qt_acidente` e `QT_ELEITOR` apresentaram multicolinearidade e pioraram a performance. As variáveis econômicas tradicionais (TAXA_SELIC, IPCA, TAXA_DESOCUPACAO, INADIMPLENCIA) tiveram melhor relação sinal-ruído e interpretabilidade.”

---

## 2. Dados e EDA

- Período: 2014–2024 (projeto), com configuração vencedora iniciando em 2015.
- Frequência: Mensal.
- Alvo: `TOTAL_CASOS` (novos casos/mês).
- Exógenas: indicadores econômicos e sociais; seleção final com 4 variáveis econômicas tradicionais.

Padrões observados (original + verificação em notebooks de EDA):
- Sazonalidade anual clara (picos em meados do ano; vales no fim do ano).
- Tendência com crescimento até 2022 e estabilização posterior.
- Impacto pandêmico (2020–2021) com outliers.

Arquivos relevantes:
- `notebooks/01_EDA.ipynb`
- `reports/eda_summary.csv`
- Visualizações de correlação: `notebooks/data/matriz_correlacao.png` e `notebooks/data/matriz_correlacao.pdf`.

Tabela de correlações destacadas (original):

| Variável | Correlação com TOTAL_CASOS | Observação |
| --- | ---: | --- |
| TAXA_SELIC | -0.23 | Fraca |
| IPCA | -0.28 | Fraca |
| TAXA_DESOCUPACAO | 0.07 | Muito fraca |
| INADIMPLENCIA | -0.03 | Praticamente nula |
| qt_acidente | -0.81 | Muito forte; multicolinear |
| QT_ELEITOR | 0.79 | Muito forte; multicolinear |

Justificativa técnica: correlação alta isolada não implica causalidade; variáveis muito colineares degradaram generalização dos modelos, confirmando o princípio de parcimônia.

---

## 3. Preparação de Dados (configuração vencedora)

Fonte: `src/data_preparation_test.py` (TEST VERSION — base do melhor resultado). Principais decisões:
- Exclusão de 2014 (início em 2015) para reduzir regime diferente/ruído inicial.
- Remoção de `qt_acidente` e `QT_ELEITOR` (alta multicolinearidade).
- Uso apenas das variáveis econômicas tradicionais.
- Criação de features temporais (ano, mês, trimestre), lags do alvo e estatísticas móveis.

Trechos de código aplicados:

```python
# Exclusão de 2014 e ordenação
df['DATA'] = pd.to_datetime(df['DATA'])
df = df.set_index('DATA').sort_index()
df = df[df.index >= '2015-01-01']

# Remoção de variáveis de alta correlação
variables_to_remove = ['qt_acidente', 'QT_ELEITOR']
df = df.drop(columns=variables_to_remove)

# Features temporais
df['year'] = df.index.year
df['month'] = df.index.month
df['quarter'] = df.index.quarter

# Lags e rolling do alvo
for lag in [1, 2, 3, 6, 12]:
    df[f'TOTAL_CASOS_lag_{lag}'] = df['TOTAL_CASOS'].shift(lag)
for window in [3, 6, 12]:
    df[f'TOTAL_CASOS_rolling_mean_{window}'] = df['TOTAL_CASOS'].rolling(window).mean()
    df[f'TOTAL_CASOS_rolling_std_{window}'] = df['TOTAL_CASOS'].rolling(window).std()

# Tratamento de ausentes (forward/backfill + consolidação)
df = df.fillna(method='ffill').fillna(method='bfill')
df = df.dropna()
```

Divisão temporal (TEST): 80% treino / 20% teste mantendo ordem temporal, consistente com o conjunto 2015–2024 salvo em `data/processed_test/`.

Arquivos gerados:
- `data/processed_test/data_processed_test.csv`
- `data/processed_test/train_test.csv`
- `data/processed_test/test_test.csv`

---

## 4. Variáveis Exógenas: fontes, tratamento e seleção

Esta seção expande o “RELATÓRIO TÉCNICO - VARIÁVEIS EXÓGENAS E MODELOS”.

- Conjunto final (seleção vencedora):
  - `TAXA_SELIC` (BACEN)
  - `IPCA` (IBGE)
  - `TAXA_DESOCUPACAO` (IBGE/PNAD Contínua)
  - `INADIMPLENCIA` (BACEN/Serasa, conforme disponibilidade consolidada na base)

Critérios de seleção:
- Baixa multicolinearidade entre si (VIF baixo no estudo); alta estabilidade histórica; cobertura mensal consistente; interpretabilidade econômica.
- Variáveis altamente correlacionadas (`qt_acidente`, `QT_ELEITOR`) foram removidas por elevarem VIF e piorarem MAE/R².

Tratamento aplicado (quando necessário):
- Preenchimento de ausentes (forward/backfill) na pipeline geral.
- Normalização não foi mandatória para Prophet/SARIMAX; modelos de ML trataram escala via árvores/boosting.
- Criação opcional de lags e estatísticas móveis para os modelos de ML.

Exemplo de pipeline de limpeza e normalização (documentado no relatório de exógenas):

```python
def clean_exogenous_variables(df):
    df = df.fillna(method='ffill').fillna(method='bfill')
    for col in exog_vars:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    return df
```

Justificativa: ao priorizar variáveis macroeconômicas amplamente aceitas e menos colineares, aumentou-se robustez e generalização, reduzindo overfitting observado com o conjunto amplo.

---

## 5. Modelos utilizados, configurações e justificativas

Modelos testados: Baselines (Persistência, Média Móvel), SARIMAX, Prophet, Random Forest, XGBoost, LightGBM.

### 5.1 Prophet (modelo vencedor)

Configuração efetiva (fonte principal: `src/models/prophet_model.py` e `src/train_models_test.py`):

```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='additive',
    interval_width=0.95,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0
)
# Regressores adicionados (quando presentes nas colunas):
# TAXA_SELIC, IPCA, TAXA_DESOCUPACAO, INADIMPLENCIA
```

Justificativas técnicas (observações dos dados):
- Sazonalidade anual marcada → `yearly_seasonality=True` e modo aditivo coerente com amplitude estável.
- Sem padrão semanal/diário em série mensal → desativados.
- `changepoint_prior_scale` e `seasonality_prior_scale` controlam sensibilidade a mudanças e amplitude sazonal; valores adotados mantêm suavidade e evitam overfitting.
- Regressores macroeconômicos adicionados como “extra_regressors” para capturar movimentos de tendência associados ao ciclo econômico.

Previsão e componentes: outputs consolidados em `reports_test/` com gráficos e CSVs.

- Imagens: `reports_test/prophet_components.png`, `reports_test/prophet_residuals.png`, `reports_test/prophet_uncertainty.png`
- Tabelas: `reports_test/prophet_metrics.csv`, `reports_test/prophet_results.csv`, `reports_test/prophet_forecast_components.csv`

### 5.2 SARIMAX

Configuração de referência (fonte: `src/models/sarimax_model.py`):

```python
SARIMAX(
    y_train,
    exog=exog_train,               # [TAXA_SELIC, IPCA, TAXA_DESOCUPACAO, INADIMPLENCIA]
    order=(1,1,1),                 # ajustado conforme estacionariedade
    seasonal_order=(1,1,1,12),     # sazonalidade anual
    enforce_stationarity=False,
    enforce_invertibility=False
)
```

Racional: série mensal com sazonalidade; verificações ADF e diferenciação automática quando p-valor > 0.05. Apesar da modelagem teórica adequada, desempenho ficou atrás do Prophet com exógenas no nosso conjunto.

### 5.3 Modelos de ML (Random Forest, XGBoost, LightGBM)

Features geradas: numéricas (incluindo `year`, `month`, `quarter`, lags do alvo, rolling stats e lags exógenos quando disponíveis), com remoção de colunas com >50% NaN.

Exemplo (fonte: `src/models/ml_models.py` e `src/train_models_test.py`):

```python
# Random Forest (parâmetros padrão adotados)
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

Observação: embora bem treinados e validados com divisão temporal, os modelos de árvore/boosting não capturaram tão bem a estrutura sazonal e tendência quanto o Prophet com regressão exógena explícita.

---

## 6. Avaliação e resultados

Fonte consolidada: `reports_test/metrics_test.csv`

| Modelo | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Prophet | 3633.99 | 4597.38 | 0.3391 |
| Random Forest | 6826.86 | 7874.27 | -0.9389 |
| LightGBM | 7464.13 | 8876.24 | -1.4638 |
| XGBoost | 7669.06 | 8918.02 | -1.4870 |
| Baseline Média Móvel (12) | 8560.47 | 9754.17 | -1.9753 |
| SARIMAX | 9415.59 | 11289.85 | -2.9858 |
| Baseline Persistência | 16541.25 | 17481.17 | -8.5562 |

Gráficos comparativos e previsões:
- Comparação de previsões (TEST): `reports_test/predictions_comparison_test.png`
- Previsão com incerteza (melhor modelo): `reports_test/forecast_future.png`

Principais conclusões técnicas:
- Prophet com 4 variáveis econômicas tradicionais e janela 2015+ superou significativamente as demais alternativas (44% melhor vs. abordagem “completa”).
- A retirada de variáveis de alta correlação reduziu multicolinearidade e melhorou generalização (MAE e R²).
- Baselines e MLs serviram como controle e exploração, mas não superaram Prophet no nosso regime de dados.

---

## 7. Previsões futuras (2025)

Baseadas no melhor modelo (Prophet), com faixas de incerteza a 95% (arquivo: `reports_test/forecast_results.csv` e figura `reports_test/forecast_future.png`). Estatísticas destacadas (originais):
- Média prevista: 58.887 casos/mês
- Máximo: 63.158 (julho)
- Mínimo: 53.908 (dezembro)
- Tendência anual: diminuição aproximada de 2.537 casos

Interpretação: padrão sazonal consistente com histórico, banda de incerteza estável, coerente com sazonalidade anual e nível atual da série.

---

## 8. Justificativa das decisões técnicas

- Parcimônia: evidências empíricas claras de que reduzir dimensionalidade e remover variáveis colineares melhora MAE e R².
- Seleção temporal (2015+): reduz regime inicial distinto, mantendo período suficientemente longo e homogêneo para estimação confiável.
- Prophet: estrutura aditiva com regressão exógena se adapta à sazonalidade anual e tendência com interpretabilidade e bom fit.
- SARIMAX: apropriado para séries estacionárias; no entanto, em nossos dados, a combinação de tendência/sazonalidade e exógenas foi melhor capturada por Prophet.
- ML: útil para teste de robustez; desempenho inferior sugere que a componente temporal explícita foi determinante.

---

## 9. Artefatos e referências

- Dados processados (TEST): `data/processed_test/`
- Métricas consolidadas: `reports_test/metrics_test.csv`
- Resultados e componentes Prophet: arquivos `reports_test/prophet_*.csv` e imagens correspondentes
- Comparação de previsões (TEST): `reports_test/predictions_comparison_test.png`
- Previsão futura (TEST): `reports_test/forecast_future.png`
- Código-fonte:
  - Preparação (TEST): `src/data_preparation_test.py`
  - Treinamento (TEST): `src/train_models_test.py`
  - Modelos: `src/models/prophet_model.py`, `src/models/sarimax_model.py`, `src/models/ml_models.py`

---

## 10. Conclusões e próximos passos

- Conclusões:
  - Prophet com variáveis econômicas tradicionais (2015+) é a configuração recomendada.
  - A remoção de variáveis de alta correlação foi decisiva para ganho de generalização.
  - O sistema está reprodutível e documentado; outputs e métricas estão versionados na pasta de relatórios de teste.

- Próximos passos:
  - Automatizar retreinamento mensal e monitoramento (drift e métricas de acurácia).
  - Incorporar dashboard executivo com comparação previsão vs. realizado.
  - Explorar granularidades (por classe/comarca) e novas exógenas com controle de multicolinearidade.

---

Gerado com base na integração dos documentos originais, códigos-fonte e relatórios do repositório. A equipe técnica e magistrados podem, a partir deste material, auditar cada etapa, reproduzir os resultados e entender o racional por trás das escolhas realizadas.
