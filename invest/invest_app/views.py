from django.shortcuts import render
import   joblib
# Create your views here.
from django.http import HttpResponse
import pickle
import mlflow
import mlflow.pyfunc
from zenml.client import Client




def sentiment(request):
    return render(request , 'home.html')

def  sentimentResult(request): 

    vectorizer_artifact = Client().get_artifact_version('ae428915-e9f8-46ad-80a3-f223ebb4e6ce')
    vectorizer = vectorizer_artifact.load()

    model_artifact = Client().get_artifact_version('6a4a2d3b-8ba9-4248-9c4e-752080717532')
    model = model_artifact.load()
    sentiment = request.POST['sentiment']
    sentiment =  [sentiment]
    sentiment_vectorized = vectorizer.transform(sentiment)
    prediction = model.predict(sentiment_vectorized)
    

# Output the prediction
    return HttpResponse(f'Predicted sentiment: {prediction}')



