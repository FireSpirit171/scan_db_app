from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_dataset, name='upload'),
    path('results/', views.results, name='results'),
    path('analyze/', views.analyze_dataset, name='analyze'),
    path('history/', views.history, name='history'),
    path('history/<str:history_id>/', views.history_detail, name='history_detail'),
]
