# genre_classifier/urls.py
from django.urls import path
from . import views


urlpatterns = [
 path('', views.upload_audio, name='index'),
]
