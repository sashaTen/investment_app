from zenml.client import Client
from  .orchestra   import   load_data     ,   zen_sentiment_analysis_pipeline
client = Client()

url = 'https://raw.githubusercontent.com/surge-ai/stock-sentiment/main/sentiment.csv'

def check_number_samples(df ,  url):
    if  df.shape[0]> 490:
        print('the   autoretrain  is  caused ')
        zen_sentiment_analysis_pipeline(url)
        return    
    else :
        print('not   enough samples  for  retrain ' ,    df.shape[0])
        return 
    

 

#df   =   load_data(url)
#check_number_samples(df ,  url)







def load_current_vectorizer_and_model():
    my_runs_on_current_stack = client.list_pipeline_runs(
        stack_id=client.active_stack_model.id,
        user_id=client.active_user.id,
        sort_by="desc:start_time",
        size=10,
    )

    if not my_runs_on_current_stack:
        raise ValueError("No pipeline runs found for the current stack.")

    latest_pipeline_run_id = my_runs_on_current_stack[0].id
    pipeline_run = client.get_pipeline_run(latest_pipeline_run_id)

    # Fetch the vectorizer and output matrices
    zen_preprocess_text_step = pipeline_run.steps.get('zen_preprocess_text')
    
    # Load vectorizer and matrices
    vectorizer_artifact = zen_preprocess_text_step.outputs['output_0'].load()  # Ensure this key matches how it's saved

    # Load the model
    zen_train_model_step = pipeline_run.steps.get('zen_train_model')
    model_artifact = zen_train_model_step.outputs['output'].load()

    return model_artifact, vectorizer_artifact


