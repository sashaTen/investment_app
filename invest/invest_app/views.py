from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def sentiment(request):
    return HttpResponse("Hello, World!")
