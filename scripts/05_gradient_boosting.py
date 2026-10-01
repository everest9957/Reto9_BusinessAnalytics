# ============================================================
# RETO 9 — Modelo 3: Gradient Boosting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
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
print("GRADIENT BOOSTING")
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
print("ENTRENANDO GRADIENT BOOSTING")
print("=" * 60)

modelo = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)

# GradientBoosting no acepta class_weight, usamos sample_weight
sample_weights = np.where(y_train == 1, 3, 1)  # Peso 3x para churners

modelo.fit(X_train, y_train, sample_weight=sample_weights)
print("✅ Modelo entrenado")

# ============================================================
# 3. PREDICCIONES Y EVALUACIÓN
# ============================================================
print("\n" + "=" * 60)
print("EVALUACIÓN DEL MODELO")
print("=" * 60)

y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

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
    'Modelo': ['Gradient Boosting'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1': [f1],
    'AUC_ROC': [auc]
})
metricas.to_csv(RUTA_RESULTADOS / "metricas_gradient_boosting.csv", index=False)

# ============================================================
# 4. IMPORTANCIA DE VARIABLES
# ============================================================
print("\n" + "=" * 60)
print("IMPORTANCIA DE VARIABLES")
print("=" * 60)

importancias = pd.DataFrame({
    'Variable': X_train.columns,
    'Importancia': modelo.feature_importances_
}).sort_values('Importancia', ascending=False)

print(importancias.head(15).to_string(index=False))
importancias.to_csv(RUTA_RESULTADOS / "importancia_gradient_boosting.csv", index=False)

# ============================================================
# 5. VISUALIZACIONES
# ============================================================
print("\n" + "=" * 60)
print("VISUALIZACIONES")
print("=" * 60)

# --- 5.1 Matriz de confusión ---
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
ax.set_title('Matriz de Confusión — Gradient Boosting', fontweight='bold')
ax.set_xlabel('Predicho')
ax.set_ylabel('Real')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "12_confusion_gradient_boosting.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 12_confusion_gradient_boosting.png")

# --- 5.2 Curva ROC ---
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax.plot(fpr, tpr, color='darkorange', linewidth=2, label=f'AUC = {auc:.4f}')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Aleatorio')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Curva ROC — Gradient Boosting', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "13_roc_gradient_boosting.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 13_roc_gradient_boosting.png")

# --- 5.3 Importancia de variables ---
fig, ax = plt.subplots(figsize=(10, 8))
top_importancias = importancias.head(15)
ax.barh(top_importancias['Variable'], top_importancias['Importancia'],
        color='darkorange', edgecolor='black')
ax.set_title('Top 15 variables más importantes (Gradient Boosting)', fontweight='bold')
ax.set_xlabel('Importancia')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "14_importancia_gradient_boosting.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 14_importancia_gradient_boosting.png")

# ============================================================
# 6. GUARDAR MODELO
# ============================================================
with open(RUTA_MODELOS / "modelo_gradient_boosting.pkl", 'wb') as f:
    pickle.dump(modelo, f)
print(f"\n✅ Modelo guardado: modelo_gradient_boosting.pkl")

print("\n" + "=" * 60)
print("✅ GRADIENT BOOSTING COMPLETADO")
print("=" * 60)