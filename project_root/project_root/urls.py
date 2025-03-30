# project_root/urls.py
from django.contrib import admin
from django.urls import include, path


urlpatterns = [
 path('genre_classifier/', include('genre_classifier.urls')),
 path('admin/', admin.site.urls),
]
