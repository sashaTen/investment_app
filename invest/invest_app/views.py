from django.shortcuts import render
import   joblib
# Create your views here.
from django.http import HttpResponse
import pickle
import mlflow
import mlflow.pyfunc
from zenml.client import Client
import subprocess
from .util_functions import load_current_vectorizer_and_model, check_number_samples
from  .orchestra  import  zen_sentiment_analysis_pipeline

def sentiment(request):
    return render(request , 'home.html')

def  sentimentResult(request): 
    '''
     result = subprocess.run(
        ['python', r'C:\Users\HP\Desktop\stock_app\invest\invest_app\util_functions.py'],  # Use raw string or fix path
        capture_output=True,
        text=True
    )
    print(result)
    '''
   
# Load the latest model
    model , vectorizer   = load_current_vectorizer_and_model()
    sentiment = request.POST['sentiment']
    sentiment =  [sentiment]
    sentiment_vectorized =  vectorizer.transform(sentiment)
    prediction = model.predict(sentiment_vectorized)
    
# Output the prediction
    return HttpResponse(f'Predicted sentiment: {prediction   , model}')



#     python   manage.py  runserver 