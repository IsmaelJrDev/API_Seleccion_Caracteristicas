# En: ml_api/urls.py

from django.urls import path
from . import views

# Esta ruta ahora se accederá como:
# POST /api/v1/analizar/

urlpatterns = [
    # Renombramos la vista para que sea más claro
    path('analizar/', views.AnalyzeFeaturesView.as_view(), name='analizar-caracteristicas'),
    
    # Mantenemos la vista para tu página de inicio (index.html)
    # Esta ruta está en rf_project/urls.py, así que aquí no hace falta.
]