from django.shortcuts import render
import   joblib
# Create your views here.
from django.http import HttpResponse

import pickle
import mlflow
import mlflow.pyfunc



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
    sentiment_vectorized = loaded_vectorizer.transform(sentiment)

# Make a prediction
    prediction = loaded_model.predict(sentiment_vectorized)

# Output the prediction
    return HttpResponse(f'Predicted sentiment: {prediction[0]}')
