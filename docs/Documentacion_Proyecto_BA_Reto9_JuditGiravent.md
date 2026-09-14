# DOCUMENTACIÓN DEL PROYECTO DE BUSINESS ANALYTICS
## Reto 9 — Predicción de Churn con Inteligencia Artificial
### El Mercado de las Especias de Dataclysm

---

**Autora:** Judit Giravent  
**Fecha:** 09/2026  
**Herramientas:** Python 3.13 · scikit-learn · TensorFlow/Keras · Power BI  
**Dataset:** Telco Customer Churn (Kaggle — IBM)

---

## Índice

1. Resumen ejecutivo
2. Definición del problema y objetivos
3. Selección y recolección de datos
4. Análisis Exploratorio de Datos (EDA)
5. Preparación y limpieza de datos
6. Desarrollo de modelos avanzados de IA
7. Evaluación de modelos
8. Interpretación de resultados
9. Recomendaciones estratégicas
10. Conclusiones
11. Anexos

---

## 1. Resumen ejecutivo

Este proyecto desarrolla un **sistema de predicción de churn** (abandono de clientes) para una empresa de telecomunicaciones, aplicando **4 modelos avanzados de Inteligencia Artificial** sobre un dataset de 7.043 clientes.

### Objetivos principales

- **Predecir** qué clientes tienen alta probabilidad de abandonar el servicio.
- **Identificar** los factores más influyentes en el churn.
- **Recomendar** acciones concretas para reducir la tasa de abandono.

### Técnicas aplicadas

| # | Modelo | Tipo | AUC-ROC | F1 |
|---|---|---|---|---|
| 1 | Regresión Logística | Baseline | 0.8418 | 0.6157 |
| 2 | **Random Forest** | **Ensemble (ganador)** | **0.8445** ⭐ | **0.6350** ⭐ |
| 3 | Gradient Boosting | Ensemble | 0.8435 | 0.6213 |
| 4 | Red Neuronal | Deep Learning | 0.8310 | 0.6187 |

### Hallazgos principales

1. **El 26,54% de los clientes abandonan** (1.869 de 7.043).
2. **El tipo de contrato es el factor más determinante**:
   - Month-to-month: **42,7% churn**
   - One year: 11,3% churn
   - Two year: **2,8% churn**
3. **Los clientes nuevos abandonan más**: tenure medio de 18 meses (churners) vs 37,6 (no churners).
4. **El servicio de Fibra Óptica tiene más churn** (41,9%) que DSL (19%) o sin internet (7,4%).
5. **El Random Forest es el mejor modelo** con AUC-ROC 0.8445 y F1 0.6350.

### Conclusión

El modelo **Random Forest** permite identificar al **78,6% de los clientes que abandonarán**, ofreciendo una herramienta accionable para **reducir el churn hasta un 20%** mediante campañas segmentadas.

---

## 2. Definición del problema y objetivos

### 2.1 Contexto empresarial

El sector de las **telecomunicaciones** se caracteriza por:
- Alta competencia y saturación del mercado.
- Coste elevado de captación de nuevos clientes (~5-7x más que retener).
- Tasa de churn media del 15-25% anual.

**El churn es uno de los problemas más costosos para las empresas de telecomunicaciones.**

### 2.2 Problema central

> **¿Qué clientes tienen alta probabilidad de abandonar el servicio, y por qué?**

Identificar proactivamente a los clientes en riesgo permite **actuar antes de perderlos**.

### 2.3 Objetivos específicos

| # | Objetivo | Métrica |
|---|---|---|
| 1 | Construir un modelo predictivo de churn | AUC-ROC > 0.80 |
| 2 | Identificar las variables más influyentes | Feature importance |
| 3 | Segmentar a los clientes por riesgo | Probabilidad de churn |
| 4 | Proponer acciones de retención | Recomendaciones |

### 2.4 Preguntas clave del análisis

1. ¿Qué características de los clientes están asociadas con el churn?
2. ¿Qué modelo predice mejor el abandono?
3. ¿Qué acciones concretas pueden reducir la tasa de churn?
4. ¿Cuál es el impacto económico esperado de las recomendaciones?

### 2.5 Enfoque analítico

| Tipo de análisis | Uso |
|---|---|
| **Descriptivo** | Entender el dataset y patrones históricos |
| **Predictivo** | Predecir el churn con ML/Deep Learning |
| **Prescriptivo** | Recomendar acciones de retención |

### 2.6 Métricas clave

| Métrica | Objetivo | Justificación |
|---|---|---|
| **AUC-ROC** | > 0.80 | Métrica principal para clasificación binaria |
| **F1-score** | > 0.60 | Balance Precision/Recall |
| **Recall** | Priorizar | Detectar el máximo de churners |

---

## 3. Selección y recolección de datos

### 3.1 Fuente de datos

| Característica | Valor |
|---|---|
| **Nombre** | Telco Customer Churn (IBM Sample Data) |
| **Fuente** | Kaggle |
| **URL** | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| **Registros** | 7.043 clientes |
| **Variables** | 21 columnas |
| **Formato** | CSV |

### 3.2 Justificación

- **Dataset real** de IBM, ampliamente usado en la industria.
- **Problema de negocio claro**: predecir churn.
- **Compatible con todos los modelos** requeridos (ML clásico + Deep Learning).
- **Tamaño adecuado** para entrenar modelos sin necesidad de infraestructura pesada.
- **Métricas estándar** (AUC-ROC, F1, etc.) directamente aplicables.

### 3.3 Variables del dataset

| Variable | Tipo | Descripción |
|---|---|---|
| `customerID` | Texto | ID único del cliente |
| `gender` | Categórica | Male / Female |
| `SeniorCitizen` | Binaria | 0/1 (¿es mayor de 65 años?) |
| `Partner` | Categórica | Yes / No |
| `Dependents` | Categórica | Yes / No |
| `tenure` | Numérica | Meses como cliente |
| `PhoneService` | Categórica | Yes / No |
| `MultipleLines` | Categórica | Yes / No / No phone service |
| `InternetService` | Categórica | DSL / Fiber optic / No |
| `OnlineSecurity` | Categórica | Yes / No / No internet service |
| `OnlineBackup` | Categórica | Yes / No / No internet service |
| `DeviceProtection` | Categórica | Yes / No / No internet service |
| `TechSupport` | Categórica | Yes / No / No internet service |
| `StreamingTV` | Categórica | Yes / No / No internet service |
| `StreamingMovies` | Categórica | Yes / No / No internet service |
| `Contract` | Categórica | Month-to-month / One year / Two year |
| `PaperlessBilling` | Categórica | Yes / No |
| `PaymentMethod` | Categórica | 4 métodos de pago |
| `MonthlyCharges` | Numérica | Cargo mensual (€) |
| `TotalCharges` | Numérica | Cargo total acumulado (€) |
| `Churn` | **Objetivo** | Yes / No |

### 3.4 Validación inicial

- **Sin valores nulos** en ninguna columna.
- **Sin filas duplicadas**.
- **11 valores vacíos en `TotalCharges`** (tratados en la preparación).

---

## 4. Análisis Exploratorio de Datos (EDA)

### 4.1 Distribución de la variable objetivo

| Churn | Clientes | % |
|---|---|---|
| No | 5.174 | 73,46% |
| **Yes** | **1.869** | **26,54%** |

**Observación:** el dataset está **moderadamente desbalanceado** (73/27). Requiere tratamiento con `class_weight='balanced'` en los modelos.

### 4.2 Estadísticas descriptivas

| Variable | Media | Mediana | Desv. Est. | Máximo |
|---|---|---|---|---|
| tenure (meses) | 32,37 | 29,00 | 24,56 | 72 |
| MonthlyCharges (€) | 64,76 | 70,35 | 30,09 | 118,75 |

### 4.3 Distribución de variables categóricas clave

#### Por tipo de contrato

| Contract | No Churn | Churn | % Churn |
|---|---|---|---|
| Month-to-month | 2.220 | 1.655 | **42,7%** |
| One year | 1.307 | 166 | 11,3% |
| Two year | 1.647 | 48 | **2,8%** |

**Hallazgo:** los contratos mensuales tienen **15x más churn** que los de 2 años.

#### Por servicio de internet

| InternetService | No Churn | Churn | % Churn |
|---|---|---|---|
| DSL | 1.961 | 459 | 19,0% |
| Fiber optic | 1.799 | 1.297 | **41,9%** |
| No | 1.414 | 113 | 7,4% |

**Hallazgo:** la **Fibra Óptica tiene el mayor churn** (probablemente por precio elevado o expectativas no cumplidas).

#### Por antigüedad y facturación

| Variable | No Churn | Churn |
|---|---|---|
| tenure medio | 37,57 meses | 17,98 meses |
| MonthlyCharges medio | 61,27 € | 74,44 € |

**Hallazgo:** los churners son clientes **más nuevos** y con **mayor facturación mensual**.

### 4.4 Visualizaciones clave

| Figura | Descripción |
|---|---|
| `01_distribucion_churn.png` | Distribución de la variable objetivo |
| `02_variables_numericas_churn.png` | Boxplots de tenure, MonthlyCharges y TotalCharges por Churn |
| `03_categoricas_vs_churn.png` | Churn por tipo de contrato, servicio, etc. |
| `04_correlaciones.png` | Matriz de correlación de variables numéricas |

### 4.5 Hallazgos del EDA

1. **El churn afecta al 26,54%** de los clientes.
2. **Tipo de contrato** es el factor más discriminante.
3. **Los clientes nuevos** (bajo tenure) son los más propensos a abandonar.
4. **Fibra óptica** concentra el mayor churn (41,9%).
5. **Los churners pagan más** (74,44€ vs 61,27€).

---

## 5. Preparación y limpieza de datos

### 5.1 Tratamiento de valores faltantes

**Problema:** `TotalCharges` venía como texto con **11 valores vacíos**.

**Solución aplicada:**
1. Conversión a numérico con `pd.to_numeric(errors='coerce')`.
2. Imputación de los 11 vacíos con `MonthlyCharges` (clientes nuevos).

### 5.2 Eliminación de variables irrelevantes

- **`customerID`**: eliminado (identificador único sin valor predictivo).

### 5.3 Codificación de variables categóricas

#### Variables binarias (Yes/No → 1/0)

| Variable | Tipo |
|---|---|
| `Partner` | Binaria |
| `Dependents` | Binaria |
| `PhoneService` | Binaria |
| `PaperlessBilling` | Binaria |
| `gender` | Male=1, Female=0 |

#### Variables de servicio (Yes/No/No service → 1/0)

Se simplificaron las categorías "No internet service" y "No phone service" a "No":
- MultipleLines
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

#### Variables multicategoría (One-Hot Encoding)

- **InternetService**: 3 columnas (`DSL`, `Fiber optic`, `No`)
- **Contract**: 3 columnas (`Month-to-month`, `One year`, `Two year`)
- **PaymentMethod**: 4 columnas

**Resultado:** 27 columnas en total (26 features + 1 target).

### 5.4 División train/test

| Conjunto | Registros | % Churn |
|---|---|---|
| **Train** | 5.634 (80%) | 26,54% |
| **Test** | 1.409 (20%) | 26,54% |

**Método:** división estratificada para mantener el desbalance.

### 5.5 Normalización

Se aplicó **StandardScaler** (media=0, std=1) a las 3 variables numéricas continuas:
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

**Importante:** el scaler se ajustó **solo en train** para evitar data leakage, y se aplicó a test con `transform()`.

### 5.6 Visualización

- `05_train_test_split.png`: Distribución de Churn en train y test.

## 6. Desarrollo de modelos avanzados de IA

### 6.1 Estrategia de modelado

Se entrenaron **4 modelos** con complejidad creciente:

| # | Modelo | Tipo | Justificación |
|---|---|---|---|
| 1 | Regresión Logística | Baseline | Interpretabilidad, punto de referencia |
| 2 | Random Forest | Ensemble (bagging) | Robusto, captura no linealidad |
| 3 | Gradient Boosting | Ensemble (boosting) | Precisión, mejor en datos tabulares |
| 4 | Red Neuronal | Deep Learning | Captura relaciones muy no lineales |

### 6.2 Tratamiento del desbalance

Todos los modelos usan **`class_weight='balanced'`** (o sample_weight equivalente) para compensar el desbalance 73/27. Esto prioriza detectar churners aunque baje ligeramente la Precision.

### 6.3 Modelo 1: Regresión Logística

**Configuración:**
- `max_iter=1000`
- `class_weight='balanced'`
- `random_state=42`

**Top 5 coeficientes más influyentes:**

| Variable | Coeficiente | Efecto |
|---|---|---|
| InternetService_No | -1.23 | Reduce churn |
| tenure | -1.15 | Reduce churn |
| Contract_Two year | -0.88 | Reduce churn |
| InternetService_Fiber optic | +0.88 | Aumenta churn |
| Contract_Month-to-month | +0.55 | Aumenta churn |

**Figuras:** `06_confusion_logistica.png`, `07_roc_logistica.png`, `08_coeficientes_logistica.png`.

### 6.4 Modelo 2: Random Forest

**Configuración:**
- `n_estimators=200`
- `max_depth=10`
- `min_samples_split=10`, `min_samples_leaf=5`
- `class_weight='balanced'`

**Top 5 variables más importantes:**

| Variable | Importancia |
|---|---|
| Contract_Month-to-month | 0.172 |
| tenure | 0.147 |
| TotalCharges | 0.112 |
| MonthlyCharges | 0.109 |
| Contract_Two year | 0.080 |

**Figuras:** `09_confusion_random_forest.png`, `10_roc_random_forest.png`, `11_importancia_random_forest.png`.

### 6.5 Modelo 3: Gradient Boosting

**Configuración:**
- `n_estimators=200`
- `learning_rate=0.05`
- `max_depth=4`
- `subsample=0.8`
- `sample_weight` con peso 3x para churners

**Top 5 variables:**

| Variable | Importancia |
|---|---|
| Contract_Month-to-month | **0.414** ⭐ |
| TotalCharges | 0.117 |
| MonthlyCharges | 0.109 |
| tenure | 0.107 |
| InternetService_Fiber optic | 0.068 |

**Observación:** el modelo se apoya casi exclusivamente en el **tipo de contrato** (41,4% de la importancia).

**Figuras:** `12_confusion_gradient_boosting.png`, `13_roc_gradient_boosting.png`, `14_importancia_gradient_boosting.png`.

### 6.6 Modelo 4: Red Neuronal (Deep Learning)

**Entorno:** Google Colab con TensorFlow 2.20.

**Arquitectura:**
```
Input (26 features)
  ↓
Dense(64, relu) + BatchNormalization + Dropout(0.3)
  ↓
Dense(32, relu) + BatchNormalization + Dropout(0.3)
  ↓
Dense(16, relu) + Dropout(0.2)
  ↓
Dense(1, sigmoid)  ← clasificación binaria
```

**Total parámetros:** 4.737

**Configuración:**
- Optimizer: `Adam(learning_rate=0.001)`
- Loss: `binary_crossentropy`
- Métricas: `accuracy`, `AUC`
- **EarlyStopping** (patience=15)
- **ReduceLROnPlateau** (factor=0.5)
- **class_weight** calculado automáticamente

**Observación importante:** el EarlyStopping se activó en la **época 17** y restauró los pesos de la **época 2**. Esto sugiere que el modelo no mejoró más allá del inicio del entrenamiento, probablemente porque:
- El dataset tiene **pocas features (26)** para una red neuronal.
- **5.600 muestras** son pocas para Deep Learning.
- Los modelos de **ensemble (RF, GB) son más adecuados** para datos tabulares.

**Figuras:** `15_curvas_entrenamiento.png`, `16_confusion_red_neuronal.png`, `17_roc_red_neuronal.png`.

### 6.7 Modelos guardados

| Archivo | Formato |
|---|---|
| `modelo_regresion_logistica.pkl` | pickle |
| `modelo_random_forest.pkl` | pickle |
| `modelo_gradient_boosting.pkl` | pickle |
| `modelo_red_neuronal.keras` | Keras |
| `scaler.pkl` | pickle (StandardScaler) |

---

## 7. Evaluación de modelos

### 7.1 Tabla comparativa final

| Modelo | Accuracy | Precision | Recall | F1 | **AUC-ROC** |
|---|---|---|---|---|---|
| **Random Forest** | **0.7601** ⭐ | **0.5326** ⭐ | **0.7861** ⭐ | **0.6350** ⭐ | **0.8445** ⭐ |
| Gradient Boosting | 0.7473 | 0.5159 | 0.7807 | 0.6213 | 0.8435 |
| Regresión Logística | 0.7395 | 0.5060 | 0.7861 | 0.6157 | 0.8418 |
| Red Neuronal | 0.7516 | 0.5221 | 0.7594 | 0.6187 | 0.8310 |

### 7.2 Análisis por métrica

- **Accuracy:** Random Forest gana (0.7601).
- **Precision:** Random Forest gana (0.5326) → menos falsos positivos.
- **Recall:** Empate entre RF y Logística (0.7861) → detectan 294 churners de 374.
- **F1-score:** Random Forest gana (0.6350).
- **AUC-ROC:** Random Forest gana (0.8445) → mejor capacidad discriminativa.

### 7.3 Modelo seleccionado: Random Forest

**Razones de la elección:**
1. **Mejor en las 5 métricas**.
2. **Interpretable** mediante feature importance.
3. **Robusto** frente a outliers.
4. **Rápido de entrenar** (~5 segundos).
5. **No requiere GPU**.

### 7.4 Curvas ROC comparativas

**Figura:** `18_comparativa_4_modelos.png` y `19_heatmap_modelos.png`.

### 7.5 Ajuste de hiperparámetros

| Modelo | Hiperparámetros probados | Decisión |
|---|---|---|
| Random Forest | `max_depth`: 5, 10, 15 | 10 |
| Random Forest | `n_estimators`: 100, 200, 300 | 200 |
| Gradient Boosting | `learning_rate`: 0.01, 0.05, 0.1 | 0.05 |
| Red Neuronal | Dropout: 0.2, 0.3, 0.4 | 0.3, 0.3, 0.2 |

---

## 8. Interpretación de resultados

### 8.1 Factores clave del churn

#### Factor 1: Tipo de contrato

| Contract | Tasa de churn |
|---|---|
| Month-to-month | **42,7%** |
| One year | 11,3% |
| Two year | 2,8% |

**Interpretación:** los contratos mensuales permiten al cliente abandonar en cualquier momento sin penalización. Los contratos largos **reducen drásticamente el churn** (15x menos).

#### Factor 2: Antigüedad

| Segmento | tenure medio |
|---|---|
| No Churn | 37,57 meses |
| Churn | 17,98 meses |

**Interpretación:** el **riesgo se concentra en los primeros 18 meses**. Superado ese umbral, el cliente tiende a quedarse.

#### Factor 3: Servicio de internet

| InternetService | Churn |
|---|---|
| Fiber optic | **41,9%** |
| DSL | 19,0% |
| No | 7,4% |

**Interpretación:** la **Fibra Óptica tiene 2x más churn** que DSL, probablemente por:
- Mayor precio mensual.
- Expectativas más altas.
- Competencia más agresiva en ese segmento.

#### Factor 4: Cargo mensual

| Segmento | MonthlyCharges |
|---|---|
| No Churn | 61,27 € |
| Churn | 74,44 € |

**Interpretación:** los churners pagan **+21% más al mes**, lo que refuerza la hipótesis de **sensibilidad al precio**.

### 8.2 Segmentación por riesgo de churn

**Segmentación propuesta basada en probabilidad del modelo:**

| Segmento | Probabilidad | Clientes | Acción |
|---|---|---|---|
| **Crítico** | > 0.70 | ~500 | Contacto urgente |
| **Alto** | 0.50 - 0.70 | ~700 | Campaña de retención |
| **Medio** | 0.30 - 0.50 | ~1.500 | Seguimiento |
| **Bajo** | < 0.30 | ~4.300 | Marketing general |

---

## 9. Recomendaciones estratégicas

### 9.1 Recomendación 1: Incentivar contratos largos (Prioridad ALTA)

**Hallazgo:** los contratos Month-to-month tienen 42,7% churn vs 2,8% de Two year.

**Acción:**
- Ofrecer descuentos del **15-20%** por cambiar a contrato anual.
- Añadir **beneficios exclusivos** (streaming gratis, upgrade de velocidad).
- **Targeting:** clientes con >6 meses de tenure y contrato mensual.

**Impacto esperado:**
- Reducir churn mensual en **~10 puntos porcentuales**.
- **Retener ~150 clientes/año** con valor medio de ~900€/año.

**Impacto económico estimado:** ~135.000 €/año.

### 9.2 Recomendación 2: Programa de retención temprana (Prioridad ALTA)

**Hallazgo:** los churners tienen tenure medio de 18 meses.

**Acción:**
- **Contacto proactivo** en los primeros 6 meses.
- Onboarding mejorado con tutoriales personalizados.
- **Seguimiento del CSAT** en el mes 3, 6 y 12.
- Ofertas especiales en el **mes 12** (renovación anual).

**Impacto esperado:**
- Reducir churn en clientes nuevos en un **20%**.
- **Retener ~100 clientes/año** adicionales.

**Impacto económico estimado:** ~90.000 €/año.

### 9.3 Recomendación 3: Revisar la oferta de Fibra Óptica (Prioridad MEDIA-ALTA)

**Hallazgo:** Fibra Óptica tiene 41,9% churn.

**Acción:**
- **Auditar la calidad** del servicio (velocidad, caídas, atención).
- **Comparar precios** con la competencia.
- **Rediseñar el plan** con beneficios adicionales (streaming, cloud).
- Ofrecer **pruebas gratuitas** de servicios premium.

**Impacto esperado:**
- Reducir churn de Fibra en un **15%**.
- **Retener ~200 clientes/año**.

**Impacto económico estimado:** ~180.000 €/año.

### 9.4 Recomendación 4: Segmentación por valor (Prioridad MEDIA)

**Hallazgo:** los churners pagan más (+21%).

**Acción:**
- **Identificar clientes de alto valor** (MonthlyCharges > 80€).
- Ofrecer **beneficios adicionales** sin subir el precio.
- **Upselling inteligente**: por el mismo precio, añadir servicios.
- **Programa VIP** para clientes top.

**Impacto esperado:**
- Aumentar la retención en segmento premium en un **25%**.
- **Retener ~120 clientes/año**.

**Impacto económico estimado:** ~130.000 €/año.

### 9.5 Recomendación 5: Despliegue del modelo (Prioridad MEDIA)

**Acción:**
- **Integrar el modelo Random Forest** en el CRM.
- **Actualización semanal** de probabilidades de churn.
- **Alertas automáticas** cuando un cliente supere 0.70.
- **A/B testing** de las campañas de retención.

**Impacto esperado:**
- **ROI del modelo**: reducción del churn del 5% en el primer año.

---

## 10. Conclusiones

### 10.1 Conclusiones técnicas

1. **Random Forest es el mejor modelo** con AUC-ROC 0.8445 y F1 0.6350.
2. **Los modelos tradicionales superan a la red neuronal** en este dataset tabular (hallazgo habitual con pocas features).
3. **El tratamiento del desbalance** con `class_weight='balanced'` es crucial.
4. **El tipo de contrato** es el factor más determinante (importancia 41,4% en GB).

### 10.2 Conclusiones de negocio

1. **El 26,54% de los clientes abandonan** cada año.
2. **Los contratos mensuales y la Fibra Óptica** concentran el mayor churn.
3. **Los primeros 18 meses** son críticos para la retención.
4. **El churn es prevenible** con acciones concretas.

### 10.3 Impacto económico combinado

| Recomendación | Clientes retenidos/año | Impacto (€) |
|---|---|---|
| Contratos largos | 150 | 135.000 |
| Retención temprana | 100 | 90.000 |
| Mejora Fibra Óptica | 200 | 180.000 |
| Segmentación por valor | 120 | 130.000 |
| **TOTAL** | **570** | **535.000 €/año** |

**Inversión estimada:** 80.000 € (campañas + desarrollo).
**ROI estimado:** **6.7x** en el primer año.

---

## 11. Anexos

### Anexo A: Estructura del proyecto

```
Reto9_BusinessAnalytics/
├── datos/
│   ├── raw/                     (dataset original)
│   └── limpios/                 (X_train, X_test, y_train, y_test)
├── scripts/                     (6 scripts Python)
├── notebooks/                   (Colab - red neuronal)
├── modelos/                     (4 modelos + scaler)
├── salidas/
│   ├── figuras/                 (19 PNG)
│   └── resultados/              (CSVs de métricas)
├── dashboard/                   (Power BI .pbix)
└── docs/                        (documentación)
```

### Anexo B: Tecnologías utilizadas

| Herramienta | Versión | Uso |
|---|---|---|
| Python | 3.13 | Lenguaje principal |
| pandas | 3.0.5 | Manipulación de datos |
| scikit-learn | 1.5.x | Modelos ML |
| TensorFlow | 2.20 | Red neuronal |
| matplotlib | 3.11.2 | Visualizaciones |
| seaborn | 0.13.2 | Visualizaciones estadísticas |
| Power BI | Desktop | Dashboard |

### Anexo C: Glosario

| Término | Definición |
|---|---|
| **Churn** | Abandono de clientes |
| **AUC-ROC** | Área bajo la curva ROC |
| **F1-score** | Media armónica de Precision y Recall |
| **Feature importance** | Importancia de cada variable en el modelo |
| **Class weight** | Peso asignado a cada clase para compensar desbalance |

---

**FIN DE LA DOCUMENTACIÓN**  
*Documento generado por Judit Giravent — 09/2026*

