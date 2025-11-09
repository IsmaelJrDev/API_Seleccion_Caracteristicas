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

@api_view(['POST'])
def feature_selection(request):
    """API endpoint for feature selection using Random Forest"""
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
        
        # Train Random Forest
        clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf_rnd.fit(X_train, y_train)
        
        # Get feature importance
        feature_importances = {
            name: float(score) 
            for name, score in zip(list(df.columns), clf_rnd.feature_importances_)
        }
        
        # Get top 10 features
        feature_importances_sorted = dict(sorted(
            feature_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])
        
        # Train model with reduced features
        columns = list(feature_importances_sorted.keys())
        X_train_reduced = X_train[columns].copy()
        X_val_reduced = X_val[columns].copy()
        
        clf_rnd_reduced = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf_rnd_reduced.fit(X_train_reduced, y_train)
        
        # Get scores
        y_pred = clf_rnd_reduced.predict(X_val_reduced)
        f1 = f1_score(y_pred, y_val, average='weighted')
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            'success': True,
            'feature_importance': feature_importances_sorted,
            'f1_score': round(float(f1), 4),
            'execution_time': round(execution_time, 2),
            'dataset_info': {
                'total_samples': len(df),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(test_set),
                'original_features': X_train.shape[1],
                'selected_features': len(columns)
            },
            'selected_features': columns,
            'sample_data': df[columns].head(10).to_dict()
        }

        # Save to MongoDB (feature_selection_db)
        try:
            mongo_uri = os.environ.get('MONGO_URI')
            if mongo_uri:
                client = MongoClient(mongo_uri)
                db = client['feature_selection_db']  # Nueva base de datos
                collection = db['feature_selections']  # Nueva colección

                # Guardar resultados
                document = {
                    'timestamp': datetime.utcnow(),
                    'feature_importance': response_data['feature_importance'],
                    'f1_score': response_data['f1_score'],
                    'execution_time': response_data['execution_time'],
                    'dataset_info': response_data['dataset_info'],
                    'selected_features': response_data['selected_features']
                }
                collection.insert_one(document)
                client.close()
                
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

@api_view(['GET'])
def test_api(request):
    """Test endpoint to verify API is working"""
    return Response({
        'message': 'Feature Selection API is working',
        'status': 'online',
        'endpoints': {
            'test': '/api/test/',
            'feature_selection': '/api/feature-selection/'
        }
    })

def home(request):
    return render(request, 'index.html')