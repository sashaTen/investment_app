from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
import pytest 
from scipy.sparse import csr_matrix
from  .orchestra   import load_data ,  split_data   , preprocess_text



def test_load_data():
    url = 'https://raw.githubusercontent.com/surge-ai/stock-sentiment/main/sentiment.csv'
    # Check if the input is a string
    assert isinstance(url, str), "Input should be a string"

    # Load the data
    df = load_data(url)
    assert isinstance(df, pd.DataFrame), "Output should be a DataFrame"
    # Optionally, you can also check if the DataFrame is not empty
    assert not df.empty, "DataFrame should not be empty"



@pytest.fixture
def sample_df():
    data = {
        'Tweet Text': ['This is a great day', 'I hate traffic', 'Love this product', 
                       'Worst experience ever', 'Amazing service', 'Will not recommend', 
                       'Had a good time', 'Terrible app', 'Very satisfied', 'Not bad at all'],
        'Sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


###    train    test  split   test 

def test_split_data(sample_df):
    # Call the split_data function
    X_train, X_test, y_train, y_test = split_data(sample_df)
    
    # Test if the split data are pandas Series
    assert isinstance(X_train, pd.Series), "X_train should be a pandas Series"
    assert isinstance(X_test, pd.Series), "X_test should be a pandas Series"
    assert isinstance(y_train, pd.Series), "y_train should be a pandas Series"
    assert isinstance(y_test, pd.Series), "y_test should be a pandas Series"





###  testing  the   preproccessing  step
@pytest.fixture
def sample_text_data():
    X_train = ['I love pizza', 'Pizza is great', 'I hate bad service', 'Terrible experience', 'Great product']
    X_test = ['Pizza is delicious', 'Bad service is annoying']
    return X_train, X_test

def test_preprocess_text(sample_text_data):
    X_train, X_test = sample_text_data
    
    # Call the preprocess_text function
    vectorizer, X_train_vec, X_test_vec = preprocess_text(X_train, X_test)
    
    # Check if the returned vectorizer is an instance of CountVectorizer
    assert isinstance(vectorizer, CountVectorizer), "Expected a CountVectorizer object to be returned"
    
    # Check if the transformed data is in csr_matrix format
    assert isinstance(X_train_vec, csr_matrix), "X_train_vec should be a csr_matrix"
    assert isinstance(X_test_vec, csr_matrix), "X_test_vec should be a csr_matrix"

