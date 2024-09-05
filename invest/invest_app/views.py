from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def sentiment(request):
    return render(request , 'home.html')

def  sentimentResult(request): 
    text    =   request.POST['sentiment']
    return   HttpResponse(text)
