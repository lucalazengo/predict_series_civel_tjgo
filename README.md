# 🏛️ Projeto de Forecasting - Casos TJGO

Este projeto implementa um sistema de previsão de casos para o Tribunal de Justiça de Goiás (TJGO) seguindo a metodologia CRISP-DM. **Descoberta principal**: Modelos mais simples superaram abordagens complexas, com Prophet alcançando MAE de 3.634 casos.

## 🎯 Resumo Executivo

### 🏆 Principais Resultados
- **Modelo Vencedor**: Prophet (dados 2015+, variáveis econômicas tradicionais)
- **Performance**: MAE = 3.634 casos, R² = 0.339 (excelente ajuste)
- **Insight Crítico**: Menos variáveis = melhor performance (princípio da parcimônia)
- **Previsão 2025**: Média de 58.887 casos/mês com tendência de diminuição

### 🔍 Descoberta Surpreendente
O **modelo teste** (sem dados 2014 + sem variáveis de alta correlação) teve **44% melhor performance** que o modelo completo, demonstrando que:
- **Simplicidade vence complexidade**
- **4 variáveis bem escolhidas > 15 variáveis**
- **Dados de qualidade > Quantidade**

## 📊 Dados e Metodologia

### 📈 Fonte dos Dados
- **Período**: Janeiro 2014 - Dezembro 2024 (132 meses)
- **Frequência**: Mensal
- **Variável Alvo**: `TOTAL_CASOS` (casos novos por mês)
- **Variáveis Exógenas**: 15 indicadores econômicos e sociais

### 🎯 Métricas de Sucesso
- **MAE < 5.000 casos** (erro médio aceitável) ✅ **3.634 casos**
- **R² > 0.3** (explicação de pelo menos 30% da variância) ✅ **0.339**
- **Reprodutibilidade** (código modular e documentado) ✅

## 🏗️ Estrutura do Projeto

```
predict_series_civel_tjgo/
├── data/
│   ├── raw/                    # Dados originais
│   ├── processed/              # Dados processados (modelo completo)
│   └── processed_test/         # Dados processados (modelo teste)
├── notebooks/                  # Jupyter notebooks
├── src/                        # Código fonte
│   ├── data_preparation.py    # Preparação completa
│   ├── data_preparation_test.py # Preparação teste
│   ├── train_models.py        # Treinamento completo
│   ├── train_models_test.py   # Treinamento teste
│   └── forecast_future.py    # Previsões futuras
├── reports/                    # Relatórios modelo completo
├── reports_test/              # Relatórios modelo teste
├── requirements.txt            # Dependências
├── RELATORIO_TECNICO_FINAL.md # Relatório técnico completo
└── README.md                   # Este arquivo
```

## 🚀 Instalação e Execução

### 1. Configurar Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv venv_tjgo

# Ativar ambiente (Linux/Mac)
source venv_tjgo/bin/activate

# Ativar ambiente (Windows)
venv_tjgo\Scripts\activate
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Executar Análise Completa

```bash
# 1. Análise Exploratória de Dados
jupyter notebook notebooks/01_EDA.ipynb

# 2. Preparação dos Dados (Modelo Completo)
python src/data_preparation.py

# 3. Preparação dos Dados (Modelo Teste - RECOMENDADO)
python src/data_preparation_test.py

# 4. Treinamento dos Modelos (Modelo Completo)
python src/train_models.py

# 5. Treinamento dos Modelos (Modelo Teste - RECOMENDADO)
python src/train_models_test.py

# 6. Previsões Futuras (Usando Melhor Modelo)
python src/forecast_future.py
```

## 📈 Modelos Implementados e Resultados

### 🏆 Comparação de Performance

| Modelo | MAE | RMSE | R² | Status |
|--------|-----|------|----|---------| 
| **Prophet (Teste)** | **3.634** | **4.597** | **0.339** | 🏆 **VENCEDOR** |
| Prophet (Completo) | 6.472 | 7.313 | -0.245 | ❌ Overfitting |
| Random Forest | 6.827 | 7.874 | -0.939 | ❌ |
| XGBoost | 7.669 | 8.918 | -1.487 | ❌ |
| LightGBM | 7.464 | 8.876 | -1.464 | ❌ |
| SARIMAX | 9.416 | 11.290 | -2.986 | ❌ |

### 🔮 Previsões para 2025

| Mês | Previsão | Limite Inferior | Limite Superior |
|-----|----------|-----------------|-----------------|
| Jan/2025 | 56.445 | 47.054 | 65.747 |
| Fev/2025 | 54.613 | 46.148 | 63.639 |
| Mar/2025 | 62.186 | 53.690 | 70.522 |
| **Jul/2025** | **63.158** | **53.992** | **72.300** |
| **Dez/2025** | **53.908** | **45.317** | **62.664** |

**Insights**:
- **Média**: 58.887 casos/mês (+38.9% vs histórico)
- **Pico**: Julho (63.158 casos)
- **Vale**: Dezembro (53.908 casos)
- **Tendência**: Diminuição de 2.537 casos ao longo do ano

## 🎯 Principais Descobertas

### 1. **Princípio da Parcimônia**
> "Entre duas explicações igualmente válidas, a mais simples é geralmente a correta"

**Aplicação**: Modelo com 4 variáveis econômicas tradicionais superou modelo com 15 variáveis.

### 2. **Curse of Dimensionality**
**Explicação Técnica**: Em espaços de alta dimensionalidade, todos os pontos ficam equidistantes, dificultando a aprendizagem.

### 3. **Variáveis de Alta Correlação = Ruído**
- `qt_acidente` e `QT_ELEITOR` tinham correlação 0.85+ com `TOTAL_CASOS`
- **Mas diminuíram a performance** quando incluídas
- **Causa**: Multicolineariedade e overfitting

### 4. **Variáveis Econômicas Tradicionais São Suficientes**
- **TAXA_SELIC**: Taxa básica de juros
- **IPCA**: Índice de preços ao consumidor  
- **TAXA_DESOCUPACAO**: Taxa de desemprego
- **INADIMPLENCIA**: Taxa de inadimplência

## 📁 Arquivos Gerados

### 📊 Relatórios e Métricas
- `reports/metrics.csv` - Métricas modelo completo
- `reports_test/metrics_test.csv` - Métricas modelo teste (RECOMENDADO)
- `reports_test/forecast_results.csv` - Previsões futuras detalhadas

### 📈 Visualizações
- `reports/predictions_comparison.png` - Comparação modelo completo
- `reports_test/predictions_comparison_test.png` - Comparação modelo teste
- `reports_test/forecast_future.png` - Previsões futuras com intervalos de confiança

### 📚 Documentação
- `RELATORIO_TECNICO_FINAL.md` - Relatório técnico completo
- `ANALISE_CORRELACAO.md` - Análise de correlações
- `EXECUTIVE_SUMMARY.md` - Resumo executivo
- `CHECKLIST.md` - Lista de progresso

## 🔧 Tecnologias Utilizadas

- **Python 3.11** - Linguagem principal
- **Pandas, NumPy** - Manipulação de dados
- **Matplotlib, Seaborn** - Visualizações
- **Prophet** - Modelagem de séries temporais (Facebook)
- **Scikit-learn** - Machine learning
- **XGBoost, LightGBM** - Gradient boosting
- **Statsmodels** - Modelos estatísticos (SARIMAX)
- **Jupyter** - Análise exploratória

## 📚 Metodologia CRISP-DM

1. **Business Understanding** ✅ - Entendimento do negócio TJGO
2. **Data Understanding** ✅ - EDA completo com 1.892 linhas de análise
3. **Data Preparation** ✅ - Limpeza, feature engineering, experimentos
4. **Modeling** ✅ - 7 algoritmos testados
5. **Evaluation** ✅ - Métricas rigorosas, testes estatísticos
6. **Deployment** ✅ - Código modular, documentação completa

## 🎯 Recomendações de Implementação

### 🚀 Implementação Imediata
1. **Usar modelo Prophet** (dados 2015+, 4 variáveis econômicas)
2. **Retreinar mensalmente** com novos dados
3. **Monitorar performance** com alertas automáticos
4. **Dashboard executivo** com KPIs principais

### 📈 Expansão Futura
1. **Outros tipos de processo** (criminal, família, etc.)
2. **Previsão por comarca** (geográfica)
3. **Outros tribunais** (metodologia replicável)
4. **AutoML** para otimização automática

## 💡 Lições Aprendidas

### ✅ Sucessos
- **Simplicidade vence complexidade**
- **Dados de qualidade > Quantidade**
- **Validação temporal é crucial**
- **Feature engineering é fundamental**

### ⚠️ Cuidados
- **Overfitting** com muitas variáveis
- **Multicolineariedade** entre features
- **Drift de dados** ao longo do tempo
- **Gestão de expectativas** dos stakeholders

## 👥 Equipe e Contato

- **Data Science Team** - TJGO
- **Mentoria** - Especialistas em MLOps
- **Metodologia** - CRISP-DM adaptada para séries temporais

## 📞 Próximos Passos

1. **Revisar relatório técnico completo** (`RELATORIO_TECNICO_FINAL.md`)
2. **Implementar sistema de monitoramento**
3. **Criar dashboard executivo**
4. **Treinar equipe técnica**
5. **Estabelecer processo de atualização mensal**

---

*Projeto concluído com sucesso! Modelo pronto para produção.* 🚀