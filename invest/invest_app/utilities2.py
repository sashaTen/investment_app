import  pandas  as pd
from  .models  import  TweetSentiment

def turn_database_into_dataframe(number_samples):
    # Query the database, order by ID in descending order, and limit to the latest 500 entries
    queryset = TweetSentiment.objects.all().order_by('-id')[:number_samples]
    
    # Convert the QuerySet to a list of dictionaries
    data = list(queryset.values())
    
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    # Rename columns for clarity
    df = df.rename(columns={
        'tweet_text': 'Tweet Text',
        'sentiment': 'Sentiment'
    })
    return df