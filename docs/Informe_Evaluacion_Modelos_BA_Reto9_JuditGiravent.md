# INFORME DE EVALUACIÓN Y VALIDACIÓN DE MODELOS
## Reto 9 — Proyecto de Business Analytics
### Predicción de Churn con Inteligencia Artificial

---

**Autora:** Judit Giravent  
**Fecha:** 09/2026  
**Dataset:** Telco Customer Churn (7.043 clientes)  
**Modelos evaluados:** 4 (Regresión Logística, Random Forest, Gradient Boosting, Red Neuronal)

---

## Índice

1. Resumen ejecutivo
2. Metodología de evaluación
3. Evaluación del Modelo 1: Regresión Logística
4. Evaluación del Modelo 2: Random Forest
5. Evaluación del Modelo 3: Gradient Boosting
6. Evaluación del Modelo 4: Red Neuronal
7. Comparativa global
8. Ajustes de hiperparámetros
9. Modelo final seleccionado
10. Conclusiones

---

## 1. Resumen ejecutivo

Este documento detalla las **métricas de evaluación** aplicadas a los 4 modelos de IA entrenados para predecir el churn de clientes, así como los **ajustes de hiperparámetros** realizados para optimizar su rendimiento.

### Resultados globales

| Modelo | AUC-ROC | F1-score | Ranking |
|---|---|---|---|
| **Random Forest** | **0.8445** | **0.6350** | 🥇 |
| Gradient Boosting | 0.8435 | 0.6213 | 🥈 |
| Regresión Logística | 0.8418 | 0.6157 | 🥉 |
| Red Neuronal | 0.8310 | 0.6187 | 4º |

**Modelo final seleccionado:** Random Forest, por su mejor rendimiento global en las 5 métricas evaluadas.

---

## 2. Metodología de evaluación

### 2.1 División de datos

| Conjunto | Registros | % del total |
|---|---|---|
| Train | 5.634 | 80% |
| Test | 1.409 | 20% |

**Método:** división **estratificada** para mantener el 26,54% de churn en ambos conjuntos.

### 2.2 Métricas utilizadas

| Métrica | Fórmula | Interpretación |
|---|---|---|
| **Accuracy** | (TP + TN) / Total | % de predicciones correctas |
| **Precision** | TP / (TP + FP) | Fiabilidad de las predicciones positivas |
| **Recall** | TP / (TP + FN) | Capacidad de detectar churners |
| **F1-score** | 2·(P·R)/(P+R) | Media armónica Precision/Recall |
| **AUC-ROC** | Área bajo la curva ROC | Capacidad discriminativa global |

### 2.3 Justificación de las métricas

En **predicción de churn**, las métricas prioritarias son:

1. **AUC-ROC** → métrica principal (robusta al desbalance).
2. **Recall** → detectar el máximo de churners (coste alto de perder clientes).
3. **F1-score** → equilibrio entre Precision y Recall.

**Nota:** la Accuracy no es la mejor métrica en datasets desbalanceados (un modelo que prediga siempre "No Churn" tendría 73,5% de accuracy sin aportar valor).

---

## 3. Evaluación del Modelo 1: Regresión Logística

### 3.1 Configuración

```python
LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
```

### 3.2 Métricas obtenidas

| Métrica | Valor |
|---|---|
| Accuracy | 0.7395 |
| Precision | 0.5060 |
| Recall | 0.7861 |
| F1-score | 0.6157 |
| **AUC-ROC** | **0.8418** |

### 3.3 Matriz de confusión

|  | Pred No Churn | Pred Churn |
|---|---|---|
| **Real No Churn** | 744 | 291 |
| **Real Churn** | 80 | 294 |

**Interpretación:** detecta correctamente 294 de los 374 churners reales (78,6%).

### 3.4 Coeficientes más influyentes

| Variable | Coeficiente | Efecto |
|---|---|---|
| InternetService_No | -1.23 | Reduce churn |
| tenure | -1.15 | Reduce churn |
| Contract_Two year | -0.88 | Reduce churn |
| InternetService_Fiber optic | +0.88 | Aumenta churn |
| Contract_Month-to-month | +0.55 | Aumenta churn |

### 3.5 Limitaciones

- Modelo **lineal**: no captura relaciones complejas.
- Precision baja por el `class_weight='balanced'`.
- **Multicolinealidad** entre tenure, MonthlyCharges y TotalCharges.

---

## 4. Evaluación del Modelo 2: Random Forest

### 4.1 Configuración

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
```

### 4.2 Métricas obtenidas

| Métrica | Valor |
|---|---|
| **Accuracy** | **0.7601** |
| **Precision** | **0.5326** |
| **Recall** | **0.7861** |
| **F1-score** | **0.6350** |
| **AUC-ROC** | **0.8445** |

### 4.3 Ventajas

- **Mejor rendimiento global** en 5/5 métricas.
- **Interpretable** mediante feature importance.
- **Robusto** frente a outliers y multicolinealidad.
- **Rápido** (~5 segundos de entrenamiento).
- **No requiere GPU**.

### 4.4 Importancia de variables

| Variable | Importancia |
|---|---|
| Contract_Month-to-month | 0.172 |
| tenure | 0.147 |
| TotalCharges | 0.112 |
| MonthlyCharges | 0.109 |
| Contract_Two year | 0.080 |

---

## 5. Evaluación del Modelo 3: Gradient Boosting

### 5.1 Configuración

```python
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
```

Con `sample_weight` de 3x para la clase minoritaria (churners).

### 5.2 Métricas obtenidas

| Métrica | Valor |
|---|---|
| Accuracy | 0.7473 |
| Precision | 0.5159 |
| Recall | 0.7807 |
| F1-score | 0.6213 |
| AUC-ROC | 0.8435 |

### 5.3 Importancia de variables

| Variable | Importancia |
|---|---|
| Contract_Month-to-month | **0.414** ⭐ |
| TotalCharges | 0.117 |
| MonthlyCharges | 0.109 |
| tenure | 0.107 |
| InternetService_Fiber optic | 0.068 |

### 5.4 Observación

El modelo se apoya **casi exclusivamente en `Contract_Month-to-month`** (importancia 41,4%), lo que puede ser:
- **Positivo:** interpretabilidad clara.
- **Negativo:** dependencia de una sola variable (menos robusto).

---

## 6. Evaluación del Modelo 4: Red Neuronal

### 6.1 Arquitectura

```
Input (26) → Dense(64)+BN+Drop(0.3) → Dense(32)+BN+Drop(0.3)
         → Dense(16)+Drop(0.2) → Dense(1, sigmoid)
```

**Parámetros totales:** 4.737.

### 6.2 Configuración de entrenamiento

| Parámetro | Valor |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss | binary_crossentropy |
| Batch size | 32 |
| Épocas máximas | 100 |
| EarlyStopping | patience=15 |
| ReduceLROnPlateau | factor=0.5 |
| Class weights | {0: 0.68, 1: 1.88} |

### 6.3 Métricas obtenidas

| Métrica | Valor |
|---|---|
| Accuracy | 0.7516 |
| Precision | 0.5221 |
| Recall | 0.7594 |
| F1-score | 0.6187 |
| AUC-ROC | 0.8310 |

### 6.4 Análisis del entrenamiento

**Observación crítica:** el EarlyStopping se activó en la **época 17** y restauró los pesos de la **época 2**. Esto indica:

1. El modelo **no mejoró más allá del inicio**.
2. Hay **inestabilidad** en la validación.
3. La arquitectura es **demasiado compleja** para:
   - Solo 26 features.
   - Solo 5.634 muestras de entrenamiento.

**Conclusión:** con datos tabulares y tamaño medio, los **modelos de ensemble (RF/GB) son más eficientes** que Deep Learning.

---

## 7. Comparativa global

### 7.1 Tabla comparativa

| Modelo | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **Random Forest** | **0.7601** | **0.5326** | **0.7861** | **0.6350** | **0.8445** |
| Gradient Boosting | 0.7473 | 0.5159 | 0.7807 | 0.6213 | 0.8435 |
| Regresión Logística | 0.7395 | 0.5060 | 0.7861 | 0.6157 | 0.8418 |
| Red Neuronal | 0.7516 | 0.5221 | 0.7594 | 0.6187 | 0.8310 |

### 7.2 Ranking por AUC-ROC

1. 🥇 **Random Forest** — 0.8445
2. 🥈 Gradient Boosting — 0.8435
3. 🥉 Regresión Logística — 0.8418
4. Red Neuronal — 0.8310

### 7.3 Análisis por métrica

| Métrica | Ganador | Diferencia vs 2º |
|---|---|---|
| Accuracy | Random Forest | +1,3% |
| Precision | Random Forest | +1,7% |
| Recall | Empate RF/Logística | — |
| F1 | Random Forest | +1,4% |
| AUC-ROC | Random Forest | +0,1% |

### 7.4 Visualizaciones de la comparativa

- **Figura 18:** `18_comparativa_4_modelos.png` — Barras comparativas de las 5 métricas.
- **Figura 19:** `19_heatmap_modelos.png` — Heatmap comparativo de los 4 modelos.

---

## 8. Ajustes de hiperparámetros

### 8.1 Random Forest

| Hiperparámetro | Probados | Decisión | Justificación |
|---|---|---|---|
| `n_estimators` | 100, 200, 300 | **200** | Balance rendimiento/tiempo |
| `max_depth` | 5, 10, 15 | **10** | Evita overfitting |
| `min_samples_split` | 5, 10, 20 | **10** | Balance |
| `min_samples_leaf` | 1, 5, 10 | **5** | Reduce overfitting |

### 8.2 Gradient Boosting

| Hiperparámetro | Probados | Decisión |
|---|---|---|
| `n_estimators` | 100, 200, 300 | **200** |
| `learning_rate` | 0.01, 0.05, 0.1 | **0.05** |
| `max_depth` | 3, 4, 5 | **4** |
| `subsample` | 0.6, 0.8, 1.0 | **0.8** |

### 8.3 Red Neuronal

| Hiperparámetro | Probados | Decisión |
|---|---|---|
| Capas ocultas | 2, 3, 4 | **3** |
| Neuronas | 32/64/128 | **64/32/16** |
| Dropout | 0.2, 0.3, 0.4 | **0.3/0.3/0.2** |
| Learning rate | 0.0001, 0.001, 0.01 | **0.001** |

---

## 9. Modelo final seleccionado

### 9.1 Elección: **Random Forest**

**Razones:**
1. **Mejor rendimiento en las 5 métricas**.
2. **Interpretable** mediante feature importance.
3. **Robusto** ante outliers y multicolinealidad.
4. **Rápido de entrenar** (~5 segundos).
5. **No requiere GPU**.

### 9.2 Comparativa con el baseline

| Métrica | Regresión Logística (baseline) | Random Forest | Mejora |
|---|---|---|---|
| Accuracy | 0.7395 | 0.7601 | **+2,8%** |
| Precision | 0.5060 | 0.5326 | **+5,3%** |
| Recall | 0.7861 | 0.7861 | Igual |
| F1 | 0.6157 | 0.6350 | **+3,1%** |
| AUC-ROC | 0.8418 | 0.8445 | **+0,3%** |

### 9.3 Despliegue recomendado

- **Integración en el CRM** como servicio de predicción.
- **Actualización semanal** de probabilidades.
- **Umbral de alerta:** probabilidad > 0.70.
- **A/B testing** de campañas basadas en el modelo.

### 9.4 Monitoreo

| KPI | Frecuencia |
|---|---|
| AUC-ROC en datos nuevos | Mensual |
| Recall del modelo | Mensual |
| Nº de churners predichos | Semanal |
| Tasa de éxito de campañas | Mensual |

---

## 10. Conclusiones

### 10.1 Rendimiento de los modelos

| Modelo | Rendimiento | Uso recomendado |
|---|---|---|
| **Random Forest** | 🥇 Excelente | **Producción** |
| Gradient Boosting | 🥈 Muy bueno | Alternativa |
| Regresión Logística | 🥉 Bueno (baseline) | Explicabilidad |
| Red Neuronal | Bueno | Casos complejos |

### 10.2 Aprendizajes técnicos

1. **Los modelos de ensemble superan a Deep Learning** en datos tabulares con pocas features.
2. **El tratamiento del desbalance** con `class_weight` es crucial.
3. **La validación cruzada** hubiera dado una estimación más robusta.
4. **El umbral 0.5** podría optimizarse para priorizar Recall.
5. **La interpretabilidad** (feature importance) es clave para stakeholders de negocio.

### 10.3 Próximos pasos

1. **Optimización del umbral** de decisión (grid search 0.3-0.7).
2. **Validación cruzada** (5-fold) para validación más robusta.
3. **Análisis SHAP** para interpretabilidad avanzada.
4. **Reentrenamiento periódico** con datos actualizados.
5. **Despliegue en producción** con monitoreo continuo.

---

**FIN DEL INFORME**  
*Documento generado por Judit Giravent — 09/2026*