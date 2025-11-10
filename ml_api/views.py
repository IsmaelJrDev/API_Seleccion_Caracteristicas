# En: ml_api/views.py

from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser

# Importaciones para el análisis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import io

# --- Funciones Auxiliares (las movimos aquí) ---
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

# --- Vista para tu página de inicio (index.html) ---
def home_page_view(request):
    """
    Esta vista renderiza la página principal del usuario (index.html).
    """
    return render(request, 'index.html')


# --- Vista Principal de la API ---
class AnalyzeFeaturesView(APIView):
    # Permite que la vista acepte 'multipart/form-data' (archivos)
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        """
        Esta vista recibe un archivo CSV, ejecuta el análisis completo
        de Random Forest y devuelve los resultados.
        """
        
        # 1. Validar el archivo
        if 'dataset' not in request.FILES:
            return Response(
                {"success": False, "error": "No se proporcionó ningún archivo 'dataset'."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        file_obj = request.FILES['dataset']
        
        if not file_obj.name.endswith('.csv'):
             return Response(
                {"success": False, "error": "El archivo debe ser un .csv"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # 2. Cargar el CSV en Pandas
            # Usamos io.StringIO para leer el archivo en memoria
            csv_data = file_obj.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_data))

            if 'calss' not in df.columns:
                 return Response(
                    {"success": False, "error": "El CSV debe tener una columna llamada 'calss'."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # --- 3. Ejecutar el análisis (lógica de train.py) ---
            
            # Partición de datos
            train_set, val_set, _ = train_val_test_split(df, stratify='calss')
            X_train, y_train = remove_labels(train_set, 'calss')
            X_val, y_val = remove_labels(val_set, 'calss')

            # Modelo 1: Todas las Características
            clf_rnd_all = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            clf_rnd_all.fit(X_train, y_train)
            y_pred_all = clf_rnd_all.predict(X_val)
            f1_all_features = f1_score(y_pred_all, y_val, average='weighted')
            initial_features_count = len(X_train.columns)

            # Selección de Características
            feature_importances = {
                name: score for name, score in zip(X_train.columns, clf_rnd_all.feature_importances_)
            }
            feature_importances_sorted = pd.Series(feature_importances).sort_values(ascending=False)
            top_10_features = list(feature_importances_sorted.head(10).index)

            # Modelo 2: Características Reducidas
            X_train_reduced = X_train[top_10_features]
            X_val_reduced = X_val[top_10_features]
            
            clf_rnd_reduced = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            clf_rnd_reduced.fit(X_train_reduced, y_train)
            y_pred_reduced = clf_rnd_reduced.predict(X_val_reduced)
            f1_reduced_features = f1_score(y_pred_reduced, y_val, average='weighted')
            reduced_features_count = len(X_train_reduced.columns)

            # --- 4. Preparar la Respuesta JSON (como la espera tu HTML) ---
            
            # Formatear vistas previas de datos
            initial_preview_df = df.head()
            reduced_preview_df = df[top_10_features + ['calss']].head()

            response_data = {
                "success": True,
                "initial_model": {
                    "f1_score": f1_all_features,
                    "features_count": initial_features_count,
                    "data_preview": initial_preview_df.to_dict('list') # 'list' es el formato que espera tu JS
                },
                "reduced_model": {
                    "f1_score": f1_reduced_features,
                    "features_count": reduced_features_count,
                    "feature_importance": feature_importances_sorted.to_dict(),
                    "data_preview": reduced_preview_df.to_dict('list')
                },
                "comparison": {
                    "f1_score_change": f1_reduced_features - f1_all_features,
                    "features_reduced_by": initial_features_count - reduced_features_count,
                    "features_reduced_percentage": ((initial_features_count - reduced_features_count) / initial_features_count) * 100
                }
            }
            
            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            # Capturar cualquier error durante el análisis
            return Response(
                {"success": False, "error": f"Error interno del servidor: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )