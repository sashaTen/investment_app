from django.shortcuts import render
import    pandas  as  pd 
# Create your views here.
from django.http import HttpResponse
import pickle
import mlflow
import mlflow.pyfunc
from zenml.client import Client
from .util_functions import load_current_vectorizer_and_model,   turn_database_into_dataframe
from  .orchestra  import  zen_sentiment_analysis_pipeline
from .models  import TweetSentiment ,  Count_samples_for_retrain
from .pseudo_pipeline  import  load_data




def sentiment(request):
    return render(request , 'home.html')


def    testing(request):
    zen_sentiment_analysis_pipeline()
    
# Create a DataFrame from the list of dictionaries
    #df = turn_database_into_dataframe()
    return   HttpResponse( 'hrllo ')


def  sentimentResult(request): 
#  the   script  for   subproccess and the autoretrain  you   will  find  in   notes
#  all  you  need  is   just  in   copy  paste  it  here     
#  Load the latest model
   
    prediction = 'positive'
    tweet = request.POST['tweet']
    sentiment  = request.POST['sentiment']
    tweet =  [tweet]
    sample_count =  Count_samples_for_retrain.objects.first()
    tweet_db_length= TweetSentiment.objects.count()
    new_tweet = TweetSentiment(tweet_text=tweet, prediction=prediction , sentiment = sentiment)
    new_tweet.save()
    new_tweet_db_length = TweetSentiment.objects.count()
    
    if (new_tweet_db_length>tweet_db_length):
        sample_count.samples_number= sample_count.samples_number+1
        sample_count.save()
    df =   turn_database_into_dataframe(10)
    print(df.head(2))
   
    '''  
    model , vectorizer,  accuracy   = load_current_vectorizer_and_model()
    tweet = request.POST['tweet']
    sentiment  = request.POST['sentiment']
    tweet =  [tweet]
    tweet_vectorized =  vectorizer.transform(tweet)
    prediction = model.predict(tweet_vectorized)
    new_tweet = TweetSentiment(tweet_text=tweet, prediction=prediction , sentiment = sentiment)
    new_tweet.save()
    df =   turn_database_into_dataframe()
    print(df.head())'''
   
# Output the prediction
    return HttpResponse(sample_count.samples_number)
    



#     python   manage.py  runserver



