from django.urls import path
from . import views
from django.contrib import admin
urlpatterns = [
   
    path('' ,   views.sentiment ,    name =  'sentiment' ),
    path('admin/', admin.site.urls),
    path('test/'   , views.testing ,   name  = 'testing '),
    path('sentimentResult'  ,   views.sentimentResult   ,  name   =  'sentimentResult')
  
   
   
]   