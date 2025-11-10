import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
# import joblib  <- Ya no se usa
import pickle  # <- Usamos pickle
import json    
import os

print("Iniciando el proceso de entrenamiento (con Pickle)...")

# --- Funciones Auxiliares (sin cambios) ---
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

# --- 1. Carga y Preparación de Datos (sin cambios) ---
print("Cargando el dataset...")
DATA_PATH = os.path.join('TotalFeatures-ISCXFlowMeter.csv')
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: No se encontró '{DATA_PATH}'.")
    exit()

print("Dividiendo los datos...")
train_set, val_set, test_set = train_val_test_split(df, stratify='calss')
X_train, y_train = remove_labels(train_set, 'calss')
X_val, y_val = remove_labels(val_set, 'calss')

# --- 2. Modelo 1: Todas las Características (sin cambios) ---
print("Entrenando Modelo 1 (Todas las características)...")
clf_rnd_all = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf_rnd_all.fit(X_train, y_train)
y_pred_all = clf_rnd_all.predict(X_val)
f1_all_features = f1_score(y_pred_all, y_val, average='weighted')
print(f"F1 Score (Todas las características): {f1_all_features}")

# --- 3. Selección de Características (sin cambios) ---
print("Obteniendo importancia de características...")
feature_importances = {
    name: score for name, score in zip(list(df.drop('calss', axis=1)), clf_rnd_all.feature_importances_)
}
feature_importances_sorted = pd.Series(feature_importances).sort_values(ascending=False)
top_10_features = list(feature_importances_sorted.head(10).index)
print(f"Top 10 características: {top_10_features}")

# --- Guardando la lista de características con Pickle ---
with open('top_10_features.pkl', 'wb') as f:
    pickle.dump(top_10_features, f)

# --- 4. Modelo 2: Características Reducidas (sin cambios) ---
print("Preparando datos reducidos...")
X_train_reduced = X_train[top_10_features].copy()
X_val_reduced = X_val[top_10_features].copy()

print("Entrenando Modelo 2 (Características reducidas)...")
clf_rnd_reduced = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf_rnd_reduced.fit(X_train_reduced, y_train)

print("Evaluando Modelo 2...")
y_pred_reduced = clf_rnd_reduced.predict(X_val_reduced)
f1_reduced_features = f1_score(y_pred_reduced, y_val, average='weighted')
print(f"F1 Score (Reducidas): {f1_reduced_features}")

# --- 5. Guardar Artefactos ---

# --- Guardando el modelo reducido con Pickle ---
print("Guardando modelo reducido en 'model_reduced.pkl'...")
with open('model_reduced.pkl', 'wb') as f:
    pickle.dump(clf_rnd_reduced, f)

# --- Guardando resultados (sin cambios) ---
print("Guardando resultados en 'training_results.json'...")
results = {
    "f1_score_all_features": f1_all_features,
    "f1_score_reduced_features": f1_reduced_features,
    "feature_importance_table": feature_importances_sorted.to_dict()
}
with open('training_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("¡Entrenamiento completado y artefactos (.pkl) guardados!")