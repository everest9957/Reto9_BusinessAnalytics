# ============================================================
# RETO 9 — Fase 2: Preparación de datos
# Limpieza, codificación y normalización
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE / "datos" / "raw" / "telco_churn.csv"
RUTA_LIMPIOS = BASE / "datos" / "limpios"
RUTA_FIGURAS = BASE / "salidas" / "figuras"
RUTA_RESULTADOS = BASE / "salidas" / "resultados"
RUTA_LIMPIOS.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")

# ============================================================
# 1. CARGA
# ============================================================
print("=" * 60)
print("1. CARGA DEL DATASET")
print("=" * 60)

df = pd.read_csv(RUTA_DATOS)
print(f"Dimensiones originales: {df.shape}")

# ============================================================
# 2. TRATAMIENTO DE TotalCharges
# ============================================================
print("\n" + "=" * 60)
print("2. TRATAMIENTO DE TotalCharges")
print("=" * 60)

# Convertir a numérico, los vacíos se convierten en NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Contar NaN
nulos_total = df['TotalCharges'].isnull().sum()
print(f"Valores nulos en TotalCharges tras conversión: {nulos_total}")

# Imputación: como son clientes nuevos (tenure=0), imputamos con MonthlyCharges
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
print(f"Valores nulos tras imputación: {df['TotalCharges'].isnull().sum()}")

# ============================================================
# 3. ELIMINAR customerID (no aporta al modelo)
# ============================================================
print("\n" + "=" * 60)
print("3. ELIMINAR customerID")
print("=" * 60)

df = df.drop(columns=['customerID'])
print(f"Columnas tras eliminar customerID: {df.shape[1]}")

# ============================================================
# 4. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ============================================================
print("\n" + "=" * 60)
print("4. CODIFICACIÓN DE VARIABLES CATEGÓRICAS")
print("=" * 60)

# 4.1 Variable objetivo: Churn (Yes/No → 1/0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print(f"Churn codificado: {df['Churn'].value_counts().to_dict()}")

# 4.2 Variables binarias (Yes/No → 1/0)
variables_binarias = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for var in variables_binarias:
    df[var] = df[var].map({'Yes': 1, 'No': 0})
print(f"Variables binarias codificadas: {variables_binarias}")

# 4.3 Gender
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
print(f"Gender codificado")

# 4.4 Variables con "No internet service" o "No phone service" → tratarlas como "No"
# Convertimos esas categorías en "No" para simplificar
columnas_servicio = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in columnas_servicio:
    df[col] = df[col].replace({
        'No internet service': 'No',
        'No phone service': 'No'
    })
    df[col] = df[col].map({'Yes': 1, 'No': 0})
print(f"Variables de servicio codificadas: {columnas_servicio}")

# 4.5 Variables multicategoría → One-Hot Encoding
variables_multicat = ['InternetService', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=variables_multicat, drop_first=False, dtype=int)

print(f"\nColumnas tras One-Hot Encoding: {df.shape[1]}")
print(f"Columnas nuevas: {[c for c in df.columns if any(x in c for x in ['InternetService', 'Contract', 'PaymentMethod'])]}")

# ============================================================
# 5. VERIFICAR QUE TODO ES NUMÉRICO
# ============================================================
print("\n" + "=" * 60)
print("5. VERIFICACIÓN DE TIPOS")
print("=" * 60)

tipos_no_numericos = df.select_dtypes(exclude=[np.number]).columns.tolist()
if tipos_no_numericos:
    print(f"⚠️  Columnas no numéricas: {tipos_no_numericos}")
else:
    print("✅ Todas las columnas son numéricas")

# ============================================================
# 6. DIVISIÓN TRAIN/TEST
# ============================================================
print("\n" + "=" * 60)
print("6. DIVISIÓN TRAIN/TEST (estratificada)")
print("=" * 60)

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"\nDistribución de Churn en train: {y_train.value_counts().to_dict()}")
print(f"Distribución de Churn en test: {y_test.value_counts().to_dict()}")

# ============================================================
# 7. NORMALIZACIÓN
# ============================================================
print("\n" + "=" * 60)
print("7. NORMALIZACIÓN (StandardScaler)")
print("=" * 60)

# Identificar variables numéricas continuas
variables_continuas = ['tenure', 'MonthlyCharges', 'TotalCharges']
print(f"Variables a estandarizar: {variables_continuas}")

scaler = StandardScaler()

# Ajustar SOLO en train (evitar data leakage)
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[variables_continuas] = scaler.fit_transform(X_train[variables_continuas])
X_test_scaled[variables_continuas] = scaler.transform(X_test[variables_continuas])

print(f"\n--- Estadísticas tras normalización (train) ---")
print(X_train_scaled[variables_continuas].describe().round(3))

# Guardar el scaler para producción
with open(BASE / "modelos" / "scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
print("\n✅ Scaler guardado: scaler.pkl")

# ============================================================
# 8. GUARDAR DATASETS
# ============================================================
print("\n" + "=" * 60)
print("8. GUARDAR DATASETS")
print("=" * 60)

# Dataset completo codificado (sin normalizar)
df.to_csv(RUTA_LIMPIOS / "telco_preparado.csv", index=False)
print(f"✅ Guardado: telco_preparado.csv ({df.shape})")

# Datasets train/test normalizados
X_train_scaled.to_csv(RUTA_LIMPIOS / "X_train.csv", index=False)
X_test_scaled.to_csv(RUTA_LIMPIOS / "X_test.csv", index=False)
y_train.to_csv(RUTA_LIMPIOS / "y_train.csv", index=False)
y_test.to_csv(RUTA_LIMPIOS / "y_test.csv", index=False)

print(f"✅ Guardado: X_train.csv, X_test.csv, y_train.csv, y_test.csv")

# ============================================================
# 9. VISUALIZACIONES
# ============================================================
print("\n" + "=" * 60)
print("9. VISUALIZACIONES")
print("=" * 60)

# Distribución de Churn en train/test
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

y_train.value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[0].set_title('Distribución de Churn en TRAIN', fontweight='bold')
axes[0].set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)
for i, v in enumerate(y_train.value_counts()):
    axes[0].text(i, v + 30, str(v), ha='center', fontweight='bold')

y_test.value_counts().plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[1].set_title('Distribución de Churn en TEST', fontweight='bold')
axes[1].set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)
for i, v in enumerate(y_test.value_counts()):
    axes[1].text(i, v + 10, str(v), ha='center', fontweight='bold')

plt.suptitle('Distribución de Churn tras la división estratificada', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "05_train_test_split.png", dpi=100, bbox_inches='tight')
plt.close()
print("✅ Guardado: 05_train_test_split.png")

# ============================================================
# 10. RESUMEN
# ============================================================
print("\n" + "=" * 60)
print("✅ PREPARACIÓN COMPLETADA")
print("=" * 60)
print(f"\nDataset final: {df.shape}")
print(f"Features: {X.shape[1]}")
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Churn en train: {y_train.mean()*100:.2f}%")
print(f"Churn en test: {y_test.mean()*100:.2f}%")