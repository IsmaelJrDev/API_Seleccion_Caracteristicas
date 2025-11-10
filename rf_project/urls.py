# En: rf_project/urls.py

from django.contrib import admin
from django.urls import path, include
from ml_api import views as ml_api_views  

urlpatterns = [
    path('admin/', admin.site.urls),

    # Ruta para la página principal (index.html)
    path('', ml_api_views.home_page_view, name='home'),

    # Rutas para tu API (incluirá la nueva vista 'analizar/')
    path('api/v1/', include('ml_api.urls')), 
]