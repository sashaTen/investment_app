from django.shortcuts import render
import   joblib
# Create your views here.
from django.http import HttpResponse
import pickle
import mlflow
import mlflow.pyfunc
from zenml.client import Client

vectorizer_artifact = Client().get_artifact_version('ae428915-e9f8-46ad-80a3-f223ebb4e6ce')
vectorizer = vectorizer_artifact.load()

model_artifact = Client().get_artifact_version('6a4a2d3b-8ba9-4248-9c4e-752080717532')
model = model_artifact.load()


def sentiment(request):
    return render(request , 'home.html')

def  sentimentResult(request): 
    vectorizer_path = r"C:\Users\HP\Desktop\cyber\cyber\vectorizer.pkl"
    with open(vectorizer_path, "rb") as f:
        loaded_vectorizer = pickle.load(f)
    loaded_model_path = r"C:\Users\HP\Desktop\cyber\cyber\model.pkl"
    with open(loaded_model_path, "rb") as c:
        loaded_model = pickle.load(c)
    # Load the model from MLflow
     
    
    
    sentiment = request.POST['sentiment']
    sentiment =  [sentiment]
    sentiment_vectorized = vectorizer.transform(sentiment)

# Make a prediction
   
    #prediction = make_prediction(loaded_model, sentiment_vectorized)
    prediction = model.predict(sentiment_vectorized)
    

# Output the prediction
    return HttpResponse(f'Predicted sentiment: {prediction}')



