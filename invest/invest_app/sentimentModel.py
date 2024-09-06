import pandas as pd
import nltk
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
  # WordNet data (optional, for improved lemmatization)

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/surge-ai/stock-sentiment/main/sentiment.csv')

# Extract X and y


# Function to preprocess and vectorize the text data

#C:\Program Files\Python311\Scripts




X_train, X_test, y_train, y_test = train_test_split(df['Tweet Text'],  df['Sentiment'], test_size=0.2, random_state=42)
# Initialize the CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=1000)

# Fit and transform the training data
X_train_vec = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_vec = vectorizer.transform(X_test)

# Initialize the RandomForestClassifier
model = RandomForestClassifier()
mlflow.set_experiment('SentimentAnalysisExperiment')
with mlflow.start_run() as run:
# Train the model
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    example_input = vectorizer.transform(['This is a sample input'])
    # Predict on the test set
    accuracy = accuracy_score(y_test, y_pred)
    y_pred = model.predict(X_test_vec)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.sklearn.log_model(model, 'tree', input_example=example_input)



'''
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
'''


# Load the model and vectorizer


# Example text for inference

# Transform the example text