from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('detect/', views.detect_news, name='detect'),
    path('stats/', views.stats, name='stats'),  # optional stats page
]
