from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import time
import traceback
from datetime import datetime
from pymongo import MongoClient
import os

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    """Split dataset into train, validation and test sets"""
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
    """Separate features and labels"""
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return X, y
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidor
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from django.shortcuts import render
import math  # { changed code }
import os
import gc  # Para limpieza de memoria
from datetime import datetime
import pickle  # Para guardar/cargar modelos
import hashlib  # Para generar hashes únicos
from pymongo import MongoClient  # <-- Agrega esto

@api_view(['GET'])
def test_api(request):
    """
    Endpoint de prueba para verificar que la API está funcionando
    """
    return Response({
        'message': 'API de Random Forest funcionando correctamente',
        'status': 'online',
        'endpoints': {
            'test': '/api/test/',
            'predict': '/api/predict/ (POST con dataset CSV y train_size)',
        }
    })


def image_to_base64(fig):
    """
    Convierte una figura de matplotlib a base64 para enviar en JSON
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"


def remove_labels(df, label_name):
    """
    Separa features y etiquetas
    """
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return X, y


@api_view(['GET', 'POST'])
def feature_selection(request):
    """API endpoint for feature selection using Random Forest"""
    if request.method == 'GET':
        return Response({
            'message': 'Este endpoint requiere una solicitud POST con un archivo CSV',
            'instrucciones': {
                'método': 'POST',
                'endpoint': '/api/feature-selection/',
                'parámetros_requeridos': {
                    'dataset': 'Archivo CSV que debe incluir una columna llamada "calss"'
                },
                'ejemplo_uso': 'Usa un cliente HTTP (como Postman) o un formulario HTML para enviar el archivo CSV'
            }
        }, status=status.HTTP_400_BAD_REQUEST)
    start_time = time.time()
    
    try:
        # Validate file upload
        if 'dataset' not in request.FILES:
            return Response(
                {'error': 'No se proporcionó archivo CSV'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Read dataset
        dataset_file = request.FILES['dataset']
        try:
            df = pd.read_csv(dataset_file)
        except Exception as e:
            return Response(
                {'error': f'Error al leer CSV: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate dataset
        if len(df) == 0:
            return Response(
                {'error': 'El archivo CSV está vacío'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if 'calss' not in df.columns:
            return Response(
                {'error': 'El CSV debe tener una columna llamada "calss"'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Split dataset
        train_set, val_set, test_set = train_val_test_split(df)
        
        # Separate features and labels
        X_train, y_train = remove_labels(train_set, 'calss')
        X_val, y_val = remove_labels(val_set, 'calss')

        # Train initial Random Forest with all features
        print("Training initial model with all features...")
        clf_rnd_initial = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf_rnd_initial.fit(X_train, y_train)
        
        # Get initial predictions and score
        y_pred_initial = clf_rnd_initial.predict(X_val)
        f1_initial = f1_score(y_pred_initial, y_val, average='weighted')
        print(f"Initial F1 Score with all features: {f1_initial}")
        
        # Get feature importance
        print("Calculating feature importance...")
        feature_importances = {
            name: float(score) 
            for name, score in zip(list(df.columns), clf_rnd_initial.feature_importances_)
        }
        
        # Get top 10 features
        feature_importances_sorted = dict(sorted(
            feature_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])
        
        # Train model with reduced features
        print("Training reduced model with top 10 features...")
        columns = list(feature_importances_sorted.keys())
        X_train_reduced = X_train[columns].copy()
        X_val_reduced = X_val[columns].copy()
        
        clf_rnd_reduced = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf_rnd_reduced.fit(X_train_reduced, y_train)
        
        # Get scores
        y_pred = clf_rnd_reduced.predict(X_val_reduced)
        f1_reduced = f1_score(y_pred, y_val, average='weighted')
        print(f"F1 Score with top 10 features: {f1_reduced}")
        
            
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Prepare response with both initial and reduced model results
        response_data = {
            'success': True,
            'initial_model': {
                'f1_score': round(float(f1_initial), 4),
                'features_count': X_train.shape[1],
                'data_preview': df.head(10).to_dict(),
                'features_list': list(df.columns)
            },
            'reduced_model': {
                'f1_score': round(float(f1_reduced), 4),
                'features_count': len(columns),
                'data_preview': df[columns].head(10).to_dict(),
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
        
        # Save to MongoDB (feature_selection_db)
        try:
            mongo_uri = os.environ.get('MONGO_URI')
            if mongo_uri:
                client = MongoClient(mongo_uri)
                db = client['feature_selection_db']  # Nueva base de datos
                collection = db['feature_selections']  # Nueva colección

                # Save results
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
                print("Results saved to MongoDB")
                
        except Exception as mongo_exc:
            print("Error guardando en MongoDB:", mongo_exc)

        return Response(response_data, status=status.HTTP_200_OK)
    
    except Exception as e:
        error_traceback = traceback.format_exc()
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