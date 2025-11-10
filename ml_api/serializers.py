# Contenido para: ml_api/serializers.py

from rest_framework import serializers

class FeatureInputSerializer(serializers.Serializer):
    """
    Valida las 10 características necesarias para el modelo optimizado.
    """
    # Estas son las 10 features de tu notebook
    Init_Win_bytes_forward = serializers.FloatField()
    max_flowiat = serializers.FloatField()
    min_seg_size_forward = serializers.FloatField()
    mean_flowpktl = serializers.FloatField()
    min_flowpktl = serializers.FloatField()
    std_flowpktl = serializers.FloatField()
    duration = serializers.FloatField()
    mean_flowiat = serializers.FloatField()
    flowPktsPerSecond = serializers.FloatField()
    max_fpktl = serializers.FloatField()