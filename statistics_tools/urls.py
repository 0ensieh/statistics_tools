from django.urls import path
from . import views

app_name = 'statistics_tools'


urlpatterns = [
    path('data_distribution/', views.data_distribution, name='data_distribution'),
    path('predict_data/', views.predict_data, name='predict_data'), 
    path('clean_data/', views.clean_data, name='clean_data'), 
]