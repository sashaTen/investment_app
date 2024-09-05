from django.urls import path
from . import views

urlpatterns = [
   
    path('' ,   views.sentiment ,    name =  'sentiment' ),
    path('sentimentResult'  ,   views.sentimentResult   ,  name   =  'sentimentResult')
  
   
   
]   