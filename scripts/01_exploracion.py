# ============================================================
# RETO 9 — Fase 1: Exploración inicial del dataset
# Dataset: Telco Customer Churn (Kaggle)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Rutas ---
BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE / "datos" / "raw" / "telco_churn.csv"
RUTA_FIGURAS = BASE / "salidas" / "figuras"
RUTA_RESULTADOS = BASE / "salidas" / "resultados"
RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

# --- Configuración de gráficos ---
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ============================================================
# 1. CARGA DEL DATASET
# ============================================================
print("=" * 60)
print("1. CARGA DEL DATASET")
print("=" * 60)

df = pd.read_csv(RUTA_DATOS)

print(f"\nDimensiones: {df.shape}")
print(f"Columnas: {df.columns.tolist()}")

print("\n--- Primeras 5 filas ---")
print(df.head())

print("\n--- Tipos de datos ---")
print(df.dtypes)

# ============================================================
# 2. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================
print("\n" + "=" * 60)
print("2. ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 60)

print("\n--- Variables numéricas ---")
print(df.describe().round(2))

# ============================================================
# 3. VARIABLES CATEGÓRICAS
# ============================================================
print("\n" + "=" * 60)
print("3. VARIABLES CATEGÓRICAS")
print("=" * 60)

variables_categoricas = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                         'PhoneService', 'MultipleLines', 'InternetService',
                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                         'TechSupport', 'StreamingTV', 'StreamingMovies',
                         'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

for var in variables_categoricas:
    if var in df.columns:
        print(f"\n--- {var} ---")
        print(df[var].value_counts())

# ============================================================
# 4. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO
# ============================================================
print("\n" + "=" * 60)
print("4. DISTRIBUCIÓN DE CHURN (VARIABLE OBJETIVO)")
print("=" * 60)

churn_counts = df['Churn'].value_counts()
print(churn_counts)
print(f"\nPorcentaje de churn: {churn_counts['Yes'] / len(df) * 100:.2f}%")

# Guardar
churn_counts.to_csv(RUTA_RESULTADOS / "distribucion_churn.csv")

# ============================================================
# 5. CALIDAD DE LOS DATOS
# ============================================================
print("\n" + "=" * 60)
print("5. CALIDAD DE LOS DATOS")
print("=" * 60)

print("\n--- Valores nulos ---")
print(df.isnull().sum())

print(f"\n--- Filas duplicadas ---")
print(f"Total duplicados: {df.duplicated().sum()}")

# ⚠️ Detectar problema con TotalCharges (a veces viene como object)
if df['TotalCharges'].dtype == 'object':
    print("\n⚠️  TotalCharges está como object. Contiene valores no numéricos:")
    # Contar los vacíos
    vacios = (df['TotalCharges'].str.strip() == '').sum()
    print(f"Valores vacíos: {vacios}")
    print("⚠️  Habrá que tratar esto en la fase de preparación")

# ============================================================
# 6. VISUALIZACIONES
# ============================================================
print("\n" + "=" * 60)
print("6. VISUALIZACIONES")
print("=" * 60)

# --- 6.1 Distribución de Churn ---
fig, ax = plt.subplots(figsize=(8, 6))
colores = ['#2ecc71', '#e74c3c']
churn_counts.plot(kind='bar', color=colores, edgecolor='black', ax=ax)
ax.set_title('Distribución de Churn', fontsize=14, fontweight='bold')
ax.set_xlabel('Churn')
ax.set_ylabel('Número de clientes')
for i, v in enumerate(churn_counts):
    ax.text(i, v + 50, str(v), ha='center', fontweight='bold')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "01_distribucion_churn.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 01_distribucion_churn.png")

# --- 6.2 Variables numéricas por Churn ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
variables_num = ['tenure', 'MonthlyCharges']

for i, var in enumerate(variables_num):
    ax = axes[i]
    sns.boxplot(x='Churn', y=var, data=df, ax=ax, hue='Churn',
                palette={'No': '#2ecc71', 'Yes': '#e74c3c'}, legend=False)
    ax.set_title(f'{var} por Churn', fontweight='bold')
    ax.grid(True, alpha=0.3)

# TotalCharges (conversión temporal para visualizar)
df['TotalCharges_num'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
ax = axes[2]
sns.boxplot(x='Churn', y='TotalCharges_num', data=df, ax=ax, hue='Churn',
            palette={'No': '#2ecc71', 'Yes': '#e74c3c'}, legend=False)
ax.set_title('TotalCharges por Churn', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('Distribución de variables numéricas por Churn', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "02_variables_numericas_churn.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 02_variables_numericas_churn.png")

# --- 6.3 Variables categóricas vs Churn ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
variables_clave = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                   'Contract', 'PaymentMethod']

for i, var in enumerate(variables_clave):
    ax = axes[i // 3, i % 3]
    pd.crosstab(df[var], df['Churn']).plot(kind='bar', ax=ax,
                                             color=['#2ecc71', '#e74c3c'],
                                             edgecolor='black')
    ax.set_title(f'{var} vs Churn', fontweight='bold')
    ax.set_xlabel(var)
    ax.set_ylabel('Número de clientes')
    ax.legend(title='Churn')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.suptitle('Variables categóricas vs Churn', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "03_categoricas_vs_churn.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 03_categoricas_vs_churn.png")

# --- 6.4 Mapa de calor de correlación (variables numéricas) ---
df_numerico = df[['tenure', 'MonthlyCharges', 'TotalCharges_num']].copy()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df_numerico.corr(), annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=1, ax=ax,
            cbar_kws={'label': 'Correlación'})
ax.set_title('Matriz de correlación (variables numéricas)', fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "04_correlaciones.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 04_correlaciones.png")

# ============================================================
# 7. HALLAZGOS INICIALES
# ============================================================
print("\n" + "=" * 60)
print("7. HALLAZGOS INICIALES")
print("=" * 60)

# Churn por tipo de contrato
print("\n--- Churn por Contract ---")
print(pd.crosstab(df['Contract'], df['Churn'], normalize='index').round(3) * 100)

# Churn por InternetService
print("\n--- Churn por InternetService ---")
print(pd.crosstab(df['InternetService'], df['Churn'], normalize='index').round(3) * 100)

# Tenure medio por Churn
print("\n--- Tenure medio por Churn ---")
print(df.groupby('Churn')['tenure'].mean().round(2))

# MonthlyCharges medio por Churn
print("\n--- MonthlyCharges medio por Churn ---")
print(df.groupby('Churn')['MonthlyCharges'].mean().round(2))

print("\n" + "=" * 60)
print("✅ EDA COMPLETADO")
print("=" * 60)
print(f"\nFiguras guardadas en: {RUTA_FIGURAS}")