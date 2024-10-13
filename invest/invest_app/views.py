from django.shortcuts import render
import    joblib
# Create your views here.
from django.http import HttpResponse
import pickle
import mlflow
import mlflow.pyfunc
from zenml.client import Client
import subprocess
from .util_functions import load_current_vectorizer_and_model, check_number_samples
from  .orchestra  import  zen_sentiment_analysis_pipeline
from .models  import TweetSentiment

def sentiment(request):
    return render(request , 'home.html')


def    testing(request):
    return   HttpResponse('testing')


def  sentimentResult(request): 
#  the   script  for   subproccess and the autoretrain  you   will  find  in   notes
# all  you  need  is   just  in   copy  paste  it  here     
# Load the latest model
    model , vectorizer,  accuracy   = load_current_vectorizer_and_model()
    tweet = request.POST['tweet']
    sentiment  = request.POST['sentiment']
    tweet =  [tweet]
    tweet_vectorized =  vectorizer.transform(tweet)
    prediction = model.predict(tweet_vectorized)
    new_tweet = TweetSentiment(tweet_text=tweet, prediction=prediction , sentiment = sentiment)
    new_tweet.save()
# Output the prediction
    return HttpResponse(f'Predicted sentiment: {prediction   , model ,  accuracy}')



#     python   manage.py  runserver



