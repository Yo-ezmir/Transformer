from django.urls import path
from . import views

urlpatterns = [
    path('finishtext', views.finish_text, name='finish_text'),
    path('evaluate', views.evaluate, name='evaluate'),
]