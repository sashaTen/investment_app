from django.urls import path
from . import views
from django.contrib import admin
urlpatterns = [
   
    path('' ,   views.portfolioHome,    name =  'portfolioHome' ),

]   