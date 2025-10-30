# Análise de Correlação

## **Descoberta** 

**Comparação de Correlações**

### **Variáveis Originalmente Usadas (Correlações Fracas)**

| Variável        | Correlação | Status            |
| ---------------- | ------------ | ----------------- |
| TAXA_SELIC       | -0.23        | Fraca             |
| IPCA             | -0.28        | Fraca             |
| TAXA_DESOCUPACAO | 0.07         | Muito fraca       |
| INADIMPLENCIA    | -0.03        | Praticamente nula |

### **Variáveis de Alta Correlação (Não Utilizadas Inicialmente)**

| Variável             | Correlação    | Status                  |
| --------------------- | --------------- | ----------------------- |
| **qt_acidente** | **-0.81** | 🔴**MUITO FORTE** |
| **QT_ELEITOR**  | **0.79**  | 🔴**MUITO FORTE** |
| VAREJO_RESTRITO       | 0.65            | 🟡 Forte                |
| VAREJO_AMPLIADO       | 0.62            | 🟡 Forte                |

## 1. **Multicolinearidade Detectada**

- **qt_acidente vs QT_ELEITOR**: -0.61 (correlação forte)
- **VAREJO_RESTRITO vs VAREJO_AMPLIADO**: 0.94 (quase perfeita)

### 2. **Questões de Causalidade**

- **QT_ELEITOR**: Proxy para população (causalidade indireta)
- **qt_acidente**: Pode gerar processos, mas também ser influenciado por fatores externos

### 3. **Dificuldade de Previsão**

- Precisamos prever valores futuros das variáveis exógenas
- Algumas podem ser difíceis de prever com precisão

## **1. Notebook EDA**

- Adicionada análise específica de variáveis de alta correlação
- Análise de multicolinearidade entre essas variáveis
- Avaliação de causalidade e interpretabilidade
- Recomendações de inclusão

### **Monitoramento Necessário**

- **VIF**: Verificar multicolinearidade
- **Cross-correlation**: Análise de defasagens
- **Causalidade**: Interpretação dos resultados
