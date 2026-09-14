cd /c/Users/jdthg/Documents/curso4BI/Reto9_BusinessAnalytics
cat > README.md << 'EOF'
# 🧠 Reto 9 — Proyecto completo de Business Analytics

Proyecto integrador final que aplica **técnicas avanzadas de Inteligencia Artificial** (Machine Learning y Deep Learning) sobre el dataset **Telco Customer Churn** para predecir el abandono de clientes.

## 🎯 Objetivo

Desarrollar un **proyecto completo de Business Analytics** que incluya:
- Análisis descriptivo, predictivo y prescriptivo.
- 4 modelos de IA: Regresión Logística, Random Forest, Gradient Boosting y Red Neuronal.
- Recomendaciones estratégicas para reducir el churn.

## 📊 Dataset

- **Fuente:** Kaggle (IBM Telco Customer Churn)
- **Registros:** 7.043 clientes
- **Variables:** 21 (demografía, servicios, facturación)
- **Variable objetivo:** `Churn` (Yes/No)

## 🛠️ Tecnologías

- Python 3.13 · pandas · scikit-learn
- TensorFlow / Keras (Deep Learning)
- matplotlib · seaborn
- Power BI (dashboard)

## 📁 Estructura

Reto9_BusinessAnalytics/
├── datos/ (raw + limpios)
├── scripts/ (por fases)
├── notebooks/ (análisis + Colab)
├── modelos/ (modelos entrenados)
├── salidas/ (figuras + resultados)
├── dashboard/ (Power BI)
└── docs/ (documentación)

## 📊 Resultados de los modelos

| Modelo | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **Random Forest** 🥇 | **0.7601** | **0.5326** | **0.7861** | **0.6350** | **0.8445** |
| Gradient Boosting | 0.7473 | 0.5159 | 0.7807 | 0.6213 | 0.8435 |
| Regresión Logística | 0.7395 | 0.5060 | 0.7861 | 0.6157 | 0.8418 |
| Red Neuronal | 0.7516 | 0.5221 | 0.7594 | 0.6187 | 0.8310 |

## 💡 Hallazgos clave

1. **El 26,54% de los clientes abandonan** cada año (1.869 de 7.043).
2. **El contrato Month-to-month tiene 42,7% churn** vs 2,8% del Two year.
3. **Los churners tienen tenure medio de 18 meses** vs 37,6 de los que se quedan.
4. **La Fibra Óptica concentra el 41,9% del churn**.
5. **Random Forest es el mejor modelo** (AUC-ROC 0.8445).

## 🎯 Recomendaciones estratégicas

1. **Incentivar contratos largos** → retener 150 clientes/año (135.000 €)
2. **Programa de retención temprana** → retener 100 clientes/año (90.000 €)
3. **Mejorar la oferta de Fibra** → retener 200 clientes/año (180.000 €)
4. **Segmentación por valor** → retener 120 clientes/año (130.000 €)

**Impacto total: 535.000 €/año · ROI 6,7x**

## 📄 Documentación

- [Documentación completa del proyecto](docs/Documentacion_Proyecto_BA_Reto9_JuditGiravent.pdf)
- [Informe de evaluación de modelos](docs/Informe_Evaluacion_Modelos_BA_Reto9_JuditGiravent.pdf)
- [Visualizaciones](docs/Visualizaciones_Dashboard_BA_Reto9_JuditGiravent.pdf)
- [Presentación ejecutiva](docs/Presentacion_Proyecto_BA_Reto9_JuditGiravent.pptx)
- [Código completo (ZIP)](codigo/Codigo_Proyecto_BA_Reto9_JuditGiravent.zip)

## 👤 Autora

**Judit Giravent** — [@jdthgp27](https://github.com/jdthgp27)

## 📜 Licencia

MIT
EOF