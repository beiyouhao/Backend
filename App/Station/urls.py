# App/Station/urls.py
# import from framework
from django.urls import path
# import from project
from App.Station import views

app_name="Station"
urlpatterns = [
    path('get_station/', views.get_station, name='get_station'),
    path('edit_station/', views.edit_station, name='edit_station'),
    path('lock_station/', views.lock_station, name='lock_station'),
]