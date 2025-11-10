import os
import time
import traceback
import math
import gc
import pickle
import hashlib
# import base64  <-- No se usa si quitamos la función de imagen
# from io import BytesIO <-- No se usa si quitamos la función de imagen
from datetime import datetime

# --- Imports de Django y REST Framework ---
from django.shortcuts import render
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# --- Imports de Data Science ---
import pandas as pd
import numpy as np
from pymongo import MongoClient
# import matplotlib.pyplot as plt  <-- ¡QUITADO! Consume mucha memoria y tiempo al iniciar

# --- Imports de Scikit-learn ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
# from sklearn.tree import plot_tree <-- No se usa en la vista

# ==============================================================================
# --- FUNCIONES AUXILIARES ---
# ==============================================================================

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    """
    Divide el dataset en conjuntos de entrenamiento, validación y prueba.
    """
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    
    return train_set, val_set, test_set

def remove_labels(df, label_name):
    """
    Separa las características (X) de las etiquetas (y).
    """
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return X, y

# def image_to_base64(fig):  <-- ¡QUITADO! No se usa en la API
#     """
#     Convierte una figura de matplotlib a formato base64 para enviarla en JSON.
#     """
#     buffer = BytesIO()
#     fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
#     buffer.seek(0)
#     image_base64 = base64.b64encode(buffer.read()).decode()
#     plt.close(fig)
#     return f"data:image/png;base64,{image_base64}"


# ==============================================================================
# --- VISTAS DE PÁGINA WEB (HTML) ---
# ==============================================================================

def home(request):
    """
    Renderiza la página de inicio (index.html) que contiene el formulario.
    """
    return render(request, 'index.html')


# ==============================================================================
# --- VISTAS DE API (JSON) ---
# ==============================================================================

@api_view(['GET'])
def test_api(request):
    """
    Endpoint de prueba para verificar que la API está funcionando.
    """
    return Response({
        'message': 'API de Random Forest funcionando correctamente',
        'status': 'online',
        'endpoints': {
            'test': '/api/test/',
            'feature-selection': '/api/feature-selection/ (POST con dataset CSV)',
        }
    })


@api_view(['GET', 'POST'])
@csrf_exempt
def feature_selection(request):
    """
    Endpoint principal de la API para la selección de características.
    """
    if request.method == 'GET':
        return Response({
            'message': 'Este endpoint requiere una solicitud POST con un archivo CSV',
            'instrucciones': {
                'método': 'POST',
                'endpoint': '/api/feature-selection/',
                'parámetros_requeridos': {
                    'dataset': 'Archivo CSV que debe incluir una columna llamada "calss"'
                }
            }
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Si es POST, procesa la solicitud
    start_time = time.time()
    
    try:
        # 1. Validar la carga del archivo
        if 'dataset' not in request.FILES:
            return Response(
                {'success': False, 'error': 'No se proporcionó archivo CSV'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 2. Leer el dataset
        dataset_file = request.FILES['dataset']
        try:
            df = pd.read_csv(dataset_file)
        except Exception as e:
            return Response(
                {'success': False, 'error': f'Error al leer CSV: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # 3. Validar el contenido del dataset
        if len(df) == 0:
            return Response(
                {'success': False, 'error': 'El archivo CSV está vacío'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if 'calss' not in df.columns:
            return Response(
                {'success': False, 'error': 'El CSV debe tener una columna llamada "calss"'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 4. Dividir el dataset
        train_set, val_set, test_set = train_val_test_split(df)
        
        # 5. Separar características y etiquetas
        X_train, y_train = remove_labels(train_set, 'calss')
        X_val, y_val = remove_labels(val_set, 'calss')

        # 6. Entrenar modelo inicial (con todas las características)
        print("Entrenando modelo inicial con todas las características...")
        clf_rnd_initial = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf_rnd_initial.fit(X_train, y_train)
        
        y_pred_initial = clf_rnd_initial.predict(X_val)
        f1_initial = f1_score(y_pred_initial, y_val, average='weighted')
        print(f"F1 Score Inicial (todas): {f1_initial}")
        
        # 7. Obtener importancia de características
        print("Calculando importancia de características...")
        feature_importances = {
            name: float(score) 
            for name, score in zip(list(df.columns), clf_rnd_initial.feature_importances_)
        }
        
        # 8. Obtener las 10 características más importantes
        feature_importances_sorted = dict(sorted(
            feature_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])
        
        # 9. Entrenar modelo reducido (solo con las 10 mejores)
        print("Entrenando modelo reducido (top 10)...")
        columns = list(feature_importances_sorted.keys())
        X_train_reduced = X_train[columns].copy()
        X_val_reduced = X_val[columns].copy()
        
        clf_rnd_reduced = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf_rnd_reduced.fit(X_train_reduced, y_train)
        
        y_pred = clf_rnd_reduced.predict(X_val_reduced)
        f1_reduced = f1_score(y_pred, y_val, average='weighted')
        print(f"F1 Score Reducido (top 10): {f1_reduced}")
        
        execution_time = time.time() - start_time
        
        # 10. Preparar la respuesta JSON
        response_data = {
            'success': True,
            'initial_model': {
                'f1_score': round(float(f1_initial), 4),
                'features_count': X_train.shape[1],
                'data_preview': df.head(10).to_dict(orient='list'),
                'features_list': list(df.columns)
            },
            'reduced_model': {
                'f1_score': round(float(f1_reduced), 4),
                'features_count': len(columns),
                'data_preview': df[columns].head(10).to_dict(orient='list'),
                'features_list': columns,
                'feature_importance': feature_importances_sorted
            },
            'execution_time': round(execution_time, 2),
            'dataset_info': {
                'total_samples': len(df),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(test_set)
            },
            'comparison': {
                'f1_score_change': round(float(f1_reduced - f1_initial), 4),
                'features_reduced_by': X_train.shape[1] - len(columns),
                'features_reduced_percentage': round((1 - len(columns)/X_train.shape[1]) * 100, 2)
            }
        }
        
        # 11. Guardar en MongoDB (Opcional)
        try:
            mongo_uri = os.environ.get('MONGO_URI')
            if mongo_uri:
                client = MongoClient(mongo_uri)
                db = client['feature_selection_db']
                collection = db['feature_selections']
                document = {
                    'timestamp': datetime.utcnow(),
                    'initial_model': {
                        'f1_score': response_data['initial_model']['f1_score'],
                        'features_count': response_data['initial_model']['features_count']
                    },
                    'reduced_model': {
                        'f1_score': response_data['reduced_model']['f1_score'],
                        'features_count': response_data['reduced_model']['features_count'],
                        'selected_features': response_data['reduced_model']['features_list']
                    },
                    'comparison': response_data['comparison'],
                    'execution_time': response_data['execution_time']
                }
                collection.insert_one(document)
                client.close()
                print("Resultados guardados en MongoDB")
                
        except Exception as mongo_exc:
            print(f"Error al guardar en MongoDB: {mongo_exc}")

        # 12. Enviar respuesta exitosa
        return Response(response_data, status=status.HTTP_200_OK)
    
    except Exception as e:
        # Manejo de errores inesperados
        error_traceback = traceback.format_exc()
        print(f"Error 500: {error_traceback}")
        return Response(
            {
                'success': False,
                'error': str(e),
                'details': error_traceback
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    
def home(request):
    return render(request, 'index.html')