from django.shortcuts import render
import   joblib
# Create your views here.
from django.http import HttpResponse
import pickle
import mlflow
import mlflow.pyfunc
from zenml.client import Client
try:
    # Try relative imports for Django
    from util_functions import load_current_vectorizer_and_model, check_number_samples
    from pseudo_pipeline import load_data
except ImportError:
    # Use absolute imports for ZenML standalone script
    from .util_functions import load_current_vectorizer_and_model, check_number_samples
    from .pseudo_pipeline import load_data


def sentiment(request):
    return render(request , 'home.html')

def  sentimentResult(request): 
    '''
      url = 'https://raw.githubusercontent.com/surge-ai/stock-sentiment/main/sentiment.csv'
    df   =   load_data(url)
    check_number_samples(df ,  url)
    '''
  
# Load the latest model
    model , vectorizer   = load_current_vectorizer_and_model()
    sentiment = request.POST['sentiment']
    sentiment =  [sentiment]
    sentiment_vectorized =  vectorizer.transform(sentiment)
    prediction = model.predict(sentiment_vectorized)
# Output the prediction
    return HttpResponse(f'Predicted sentiment: {prediction   , model}')



