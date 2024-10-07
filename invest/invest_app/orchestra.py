from zenml import pipeline, step
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  ,  f1_score
from typing import Tuple
import random
from scipy.sparse import csr_matrix

# Step 1: Load data





url = 'https://raw.githubusercontent.com/surge-ai/stock-sentiment/main/sentiment.csv'






def load_data(url):
    df = pd.read_csv(url)
    return df

# Step 2: Split data into train and test sets

def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['Tweet Text'], df['Sentiment'], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# Step 3: Preprocess and vectorize the text data

def preprocess_text(X_train, X_test) :
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return vectorizer, X_train_vec, X_test_vec


def train_model(X_train_vec, y_train):
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    return model

# Step 5: Evaluate the model

def evaluate_model(model, X_test_vec, y_test) :
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Model Accuracy: {accuracy:.2f}")
    return   f1

#



# Step 1: Load data
@step
def zen_load_data(url : str) -> pd.DataFrame:
   
    df = pd.read_csv(url)
    return df

# Step 2: Split data into train and test sets
@step
def  zen_split_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        df['Tweet Text'], df['Sentiment'], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# Step 3: Preprocess and vectorize the text data
@step
def  zen_preprocess_text(X_train: pd.Series, X_test: pd.Series) -> Tuple[CountVectorizer,csr_matrix, csr_matrix]:
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return vectorizer, X_train_vec, X_test_vec

# Step 4: Train the model
@step
def  zen_train_model(X_train_vec: csr_matrix, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    return model

# Step 5: Evaluate the model
@step
def  zen_evaluate_model(model: LogisticRegression, X_test_vec: csr_matrix, y_test: pd.Series) -> float:
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Model Accuracy: {accuracy:.2f}")
    return   f1

# Pipeline to connect all the steps
@pipeline
def  zen_sentiment_analysis_pipeline(url):
    df =  zen_load_data(url)
    X_train, X_test, y_train, y_test =  zen_split_data(df)
    vectorizer, X_train_vec, X_test_vec =  zen_preprocess_text(X_train, X_test)
    model =  zen_train_model(X_train_vec, y_train)
    zen_evaluate_model(model, X_test_vec, y_test)

# Run the pipeline
if __name__ == "__main__":
    zen_sentiment_analysis_pipeline(url)
   


    
