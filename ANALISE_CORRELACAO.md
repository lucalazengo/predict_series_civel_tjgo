# Análise de Correlação

## **Descoberta Principal**

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

## **Por que não foram usadas inicialmente?**

### 1. **Multicolinearidade Detectada**

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

## **Recomendações** 

### **1. Incluir**

- **qt_acidente**: Alta correlação negativa (-0.81) e causalidade plausível
- **QT_ELEITOR**: Alta correlação positiva (0.79) e fácil previsão

### **2. Avaliar Cuidadosamente**

- **VAREJO_RESTRITO/AMPLIADO**: Boa correlação mas alta multicolinearidade
- **Escolher apenas uma** das variáveis de varejo

### **3. Manter**

- **Variáveis econômicas tradicionais**: Para contexto macroeconômico
- **Monitoramento**: VIF e análise de multicolinearidade

## 🔬 **Próximos Passos**

1. **Executar EDA atualizado** com nova análise
2. **Rodar preparação de dados** com variáveis expandidas
3. **Treinar modelos** com variáveis de alta correlação
4. **Comparar performance** antes/depois
5. **Validar multicolinearidade** com VIF

---

**Conclusão**: Sua observação foi **extremamente valiosa** e levou a uma descoberta importante que deve melhorar significativamente a performance dos modelos de forecasting!
