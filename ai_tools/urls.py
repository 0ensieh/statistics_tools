from django.urls import path
from . import views

app_name = 'image_to_text_convertor'

urlpatterns = [
    path('image_to_text_converter/', views.image_to_text_converter, name='image_to_text_converter'), 

] 