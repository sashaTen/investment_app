
    # Try relative imports for Django
from .orchestra import zen_sentiment_analysis_pipeline
from .pseudo_pipeline import load_data


# Now you can use absolute imports


import pandas as pd
from zenml.client import Client

client = Client()

url = 'https://raw.githubusercontent.com/surge-ai/stock-sentiment/main/sentiment.csv'

def check_number_samples(df):
    if  df.shape[0]> 490:
        print('the  number   is more  than  460')
        return True
            
    else :
        print('not   enough samples  for  retrain ' ,    df.shape[0])
        return False
    

'''

df   =   load_data(url)
check_number_samples(df ,  url)
'''
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

    zen_evaluate_model_step = pipeline_run.steps.get('zen_evaluate_model')
    accuracy_artifact =   zen_evaluate_model_step.outputs['output'].load()

    return model_artifact, vectorizer_artifact, accuracy_artifact 





def  auto_retrain_on_new_data(df):
    if check_number_samples(df):
      zen_sentiment_analysis_pipeline(url)
      print('you caused   the   auto - retrain')
      return   

    else :
      print('not   enough  samples')
      return 



if __name__ == "__main__":
    df   =   load_data(url)
    check_number_samples(df ,  url)