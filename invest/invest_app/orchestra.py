from zenml import pipeline, step
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Tuple
import random
from scipy.sparse import csr_matrix

# Step 1: Load data
@step
def load_data() -> pd.DataFrame:
    url = 'https://raw.githubusercontent.com/surge-ai/stock-sentiment/main/sentiment.csv'
    df = pd.read_csv(url)
    return df

# Step 2: Split data into train and test sets
@step
def split_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        df['Tweet Text'], df['Sentiment'], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# Step 3: Preprocess and vectorize the text data
@step
def preprocess_text(X_train: pd.Series, X_test: pd.Series) -> Tuple[CountVectorizer,csr_matrix, csr_matrix]:
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return vectorizer, X_train_vec, X_test_vec

# Step 4: Train the model
@step
def train_model(X_train_vec: csr_matrix, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    return model

# Step 5: Evaluate the model
@step
def evaluate_model(model: LogisticRegression, X_test_vec: csr_matrix, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

# Pipeline to connect all the steps
@pipeline
def sentiment_analysis_pipeline():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    vectorizer, X_train_vec, X_test_vec = preprocess_text(X_train, X_test)
    model = train_model(X_train_vec, y_train)
    evaluate_model(model, X_test_vec, y_test)

# Run the pipeline
if __name__ == "__main__":
   sentiment_analysis_pipeline()
    
