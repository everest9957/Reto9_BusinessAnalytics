# ============================================================
# RETO 9 — Modelo 1: Regresión Logística (Baseline)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
import pickle

BASE = Path(__file__).resolve().parent.parent
RUTA_LIMPIOS = BASE / "datos" / "limpios"
RUTA_FIGURAS = BASE / "salidas" / "figuras"
RUTA_RESULTADOS = BASE / "salidas" / "resultados"
RUTA_MODELOS = BASE / "modelos"

sns.set_style("whitegrid")

# ============================================================
# 1. CARGA DE DATOS
# ============================================================
print("=" * 60)
print("REGRESIÓN LOGÍSTICA (BASELINE)")
print("=" * 60)

X_train = pd.read_csv(RUTA_LIMPIOS / "X_train.csv")
X_test = pd.read_csv(RUTA_LIMPIOS / "X_test.csv")
y_train = pd.read_csv(RUTA_LIMPIOS / "y_train.csv").squeeze()
y_test = pd.read_csv(RUTA_LIMPIOS / "y_test.csv").squeeze()

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ============================================================
# 2. ENTRENAR MODELO
# ============================================================
print("\n" + "=" * 60)
print("ENTRENANDO REGRESIÓN LOGÍSTICA")
print("=" * 60)

modelo = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Compensa el desbalance
    random_state=42
)

modelo.fit(X_train, y_train)
print("✅ Modelo entrenado")

# ============================================================
# 3. PREDICCIONES Y EVALUACIÓN
# ============================================================
print("\n" + "=" * 60)
print("EVALUACIÓN DEL MODELO")
print("=" * 60)

y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

print("\n--- Reporte completo ---")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Guardar métricas
metricas = pd.DataFrame({
    'Modelo': ['Regresión Logística'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1': [f1],
    'AUC_ROC': [auc]
})
metricas.to_csv(RUTA_RESULTADOS / "metricas_regresion_logistica.csv", index=False)

# ============================================================
# 4. INTERPRETACIÓN DE COEFICIENTES
# ============================================================
print("\n" + "=" * 60)
print("COEFICIENTES DEL MODELO")
print("=" * 60)

coeficientes = pd.DataFrame({
    'Variable': X_train.columns,
    'Coeficiente': modelo.coef_[0]
}).sort_values('Coeficiente', key=abs, ascending=False)

print(coeficientes.head(15).to_string(index=False))
coeficientes.to_csv(RUTA_RESULTADOS / "coeficientes_logistica.csv", index=False)

# ============================================================
# 5. VISUALIZACIONES
# ============================================================
print("\n" + "=" * 60)
print("VISUALIZACIONES")
print("=" * 60)

# --- 5.1 Matriz de confusión ---
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
ax.set_title('Matriz de Confusión — Regresión Logística', fontweight='bold')
ax.set_xlabel('Predicho')
ax.set_ylabel('Real')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "06_confusion_logistica.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 06_confusion_logistica.png")

# --- 5.2 Curva ROC ---
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax.plot(fpr, tpr, color='steelblue', linewidth=2, label=f'AUC = {auc:.4f}')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Aleatorio')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Curva ROC — Regresión Logística', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "07_roc_logistica.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 07_roc_logistica.png")

# --- 5.3 Importancia de variables ---
fig, ax = plt.subplots(figsize=(10, 8))
top_coefs = coeficientes.head(15)
colores = ['#e74c3c' if c > 0 else '#2ecc71' for c in top_coefs['Coeficiente']]
ax.barh(top_coefs['Variable'], top_coefs['Coeficiente'], color=colores, edgecolor='black')
ax.axvline(x=0, color='black', linewidth=1)
ax.set_title('Top 15 variables más influyentes (Regresión Logística)', fontweight='bold')
ax.set_xlabel('Coeficiente')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "08_coeficientes_logistica.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 08_coeficientes_logistica.png")

# ============================================================
# 6. GUARDAR MODELO
# ============================================================
with open(RUTA_MODELOS / "modelo_regresion_logistica.pkl", 'wb') as f:
    pickle.dump(modelo, f)
print(f"\n✅ Modelo guardado: modelo_regresion_logistica.pkl")

print("\n" + "=" * 60)
print("✅ REGRESIÓN LOGÍSTICA COMPLETADA")
print("=" * 60)