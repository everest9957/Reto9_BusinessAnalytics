# ============================================================
# RETO 9 — Evaluación comparativa de los 4 modelos
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RUTA_FIGURAS = BASE / "salidas" / "figuras"
RUTA_RESULTADOS = BASE / "salidas" / "resultados"

sns.set_style("whitegrid")

# ============================================================
# 1. CARGAR TODAS LAS MÉTRICAS
# ============================================================
print("=" * 60)
print("EVALUACIÓN COMPARATIVA FINAL")
print("=" * 60)

# Modelos entrenados en local
metricas_log = pd.read_csv(RUTA_RESULTADOS / "metricas_regresion_logistica.csv")
metricas_rf = pd.read_csv(RUTA_RESULTADOS / "metricas_random_forest.csv")
metricas_gb = pd.read_csv(RUTA_RESULTADOS / "metricas_gradient_boosting.csv")

# Red neuronal (introducir manualmente los valores de Colab)
metricas_nn = pd.DataFrame({
    'Modelo': ['Red Neuronal'],
    'Accuracy': [0.7516],
    'Precision': [0.5221],
    'Recall': [0.7594],
    'F1': [0.6187],
    'AUC_ROC': [0.8310]
})

# Consolidar
metricas_total = pd.concat([metricas_log, metricas_rf, metricas_gb, metricas_nn], ignore_index=True)
metricas_total = metricas_total.sort_values('AUC_ROC', ascending=False).reset_index(drop=True)

print("\n" + metricas_total.to_string(index=False))

# Guardar
metricas_total.to_csv(RUTA_RESULTADOS / "comparativa_final_modelos.csv", index=False)

# ============================================================
# 2. VISUALIZACIÓN COMPARATIVA
# ============================================================
print("\n" + "=" * 60)
print("GENERANDO VISUALIZACIONES")
print("=" * 60)

# --- 2.1 Barras comparativas de todas las métricas ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metricas_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC_ROC']
colores = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

for i, metrica in enumerate(metricas_plot):
    ax = axes[i // 3, i % 3]
    bars = ax.bar(metricas_total['Modelo'], metricas_total[metrica],
                  color=colores, edgecolor='black')
    ax.set_title(metrica, fontweight='bold', fontsize=12)
    ax.set_ylim([0, 1])
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Añadir valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', fontsize=8, fontweight='bold')

# Ocultar subplot sobrante
axes[1, 2].axis('off')

plt.suptitle('Comparativa de los 4 modelos de IA', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "18_comparativa_4_modelos.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 18_comparativa_4_modelos.png")

# --- 2.2 Heatmap comparativo ---
fig, ax = plt.subplots(figsize=(12, 6))
metricas_heatmap = metricas_total.set_index('Modelo')[metricas_plot]
sns.heatmap(metricas_heatmap, annot=True, fmt='.4f', cmap='RdYlGn',
            vmin=0.4, vmax=0.9, linewidths=1, ax=ax,
            cbar_kws={'label': 'Valor'})
ax.set_title('Comparativa de métricas por modelo', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "19_heatmap_modelos.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 19_heatmap_modelos.png")

# ============================================================
# 3. ANÁLISIS DE LOS RESULTADOS
# ============================================================
print("\n" + "=" * 60)
print("ANÁLISIS DE LOS RESULTADOS")
print("=" * 60)

# Mejor modelo por métrica
for metrica in metricas_plot:
    mejor = metricas_total.loc[metricas_total[metrica].idxmax()]
    print(f"\n🏆 Mejor {metrica}: {mejor['Modelo']} ({mejor[metrica]:.4f})")

# Mejor modelo global (por AUC-ROC)
mejor_modelo = metricas_total.iloc[0]
print(f"\n🥇 MEJOR MODELO GLOBAL: {mejor_modelo['Modelo']}")
print(f"   AUC-ROC: {mejor_modelo['AUC_ROC']:.4f}")
print(f"   F1-score: {mejor_modelo['F1']:.4f}")

print("\n" + "=" * 60)
print("✅ EVALUACIÓN COMPARATIVA COMPLETADA")
print("=" * 60)