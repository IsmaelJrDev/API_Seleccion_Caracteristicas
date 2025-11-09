from django.urls import path
from . import views

urlpatterns = [
    path('api/test/', views.test_api, name='test_api'),
    path('api/feature-selection/', views.feature_selection, name='feature_selection'),
    path('', views.home, name='home'),
]