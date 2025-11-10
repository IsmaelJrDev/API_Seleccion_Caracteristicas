# ml_api/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # 1. Endpoint de la API para procesar el archivo CSV
    path('api/feature-selection/', views.feature_selection, name='feature_selection'),
    
    # 2. Ruta raíz para mostrar la página index.html
    path('', views.home, name='home'),
]