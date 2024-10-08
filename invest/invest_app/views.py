from django.shortcuts import render
import   joblib
# Create your views here.
from django.http import HttpResponse
import pickle
import mlflow
import mlflow.pyfunc
from zenml.client import Client
from  .orchestra   import   load_data     ,   zen_sentiment_analysis_pipeline
from .util_functions import   load_current_vectorizer_and_model


def sentiment(request):
    return render(request , 'home.html')

def  sentimentResult(request): 
    

# Load the latest model
   
  

   
    model , vectorizer   = load_current_vectorizer_and_model()
    sentiment = request.POST['sentiment']
    sentiment =  [sentiment]
    sentiment_vectorized =  vectorizer.transform(sentiment)
    prediction = model.predict(sentiment_vectorized)
    

# Output the prediction
    return HttpResponse(f'Predicted sentiment: {prediction}')



