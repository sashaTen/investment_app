from django.shortcuts import render
import    pandas  as  pd 
# Create your views here.
from django.http import HttpResponse
import pickle
import mlflow
import mlflow.pyfunc
from zenml.client import Client
from .util_functions import load_current_vectorizer_and_model
from  .orchestra  import  zen_sentiment_analysis_pipeline
from .models  import TweetSentiment ,  Count_samples_for_retrain
from .pseudo_pipeline  import  load_data
from .utilities2   import   turn_database_into_dataframe
from transformers import T5Tokenizer, T5ForConditionalGeneration


vectorizer_artifact = Client().get_artifact_version("91734428-5393-42ff-a7c2-45973edd995d")
vectorizer = vectorizer_artifact.load()


model_artifact = Client().get_artifact_version("121c294e-3bba-40bb-9e11-a3f9d2dd42f5")
model = model_artifact.load()

def sentiment(request):
    return render(request , 'home.html')


def    testing(request):
   # zen_sentiment_analysis_pipeline()
   
# Load the T5-small model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Define input text (for summarization task)
    input_text = "summarize: The stock market performed well today with major indexes rising."

    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids

    # Generate the summary
    outputs = model.generate(input_ids, max_length=50, num_beams=2, early_stopping=True)

    # Decode and print the output
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
   # print(summary)
        
# Create a DataFrame from the list of dictionaries
    #df = turn_database_into_dataframe(50)
    
    return   HttpResponse( summary)


def  sentimentResult(request): 

    tweet = request.POST['tweet']
    tweet =  [tweet]
    
    tweet_vectorized =  vectorizer.transform(tweet)
    prediction = model.predict(tweet_vectorized)
# Output the prediction

    #return HttpResponse(f'samples  count   :   {sample_count.samples_number} ,  prediction is   :{prediction} ,   the  model is  : {model}' )
    return HttpResponse(prediction)
    



#     python   manage.py  runserver



