import pandas as pd
import nltk
import  random
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
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
params = {
    'n_estimators': random.randint(10, 200),         # Number of trees in the forest  # Maximum depth of the tree
    'min_samples_split': random.randint(2, 10),      # Minimum number of samples required to split an internal node
    'min_samples_leaf': random.randint(1, 10),       # Minimum number of samples required to be at a leaf node
    'max_features': random.choice(['auto', 'sqrt', 'log2', None]),  # Number of features to consider when looking for the best split
    # Whether bootstrap samples are used when building trees
}
# Initialize the RandomForestClassifier
model =  DecisionTreeClassifier()
mlflow.set_experiment('SentimentAnalysisExperiment')
with mlflow.start_run() as run:
# Train the model
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    example_input = vectorizer.transform(['This is a sample input'])
    # Predict on the test set
    signature = infer_signature(X_train_vec, model.predict(X_train_vec))
    accuracy = accuracy_score(y_test, y_pred)
    y_pred = model.predict(X_test_vec)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.sklearn.log_model(model, ' RandomForestClassifier', input_example=example_input)
    mlflow.set_tag("Training Info", "Basic  model for  sentiment ")
    #mlflow.log_artifact(r"C:\Users\HP\Desktop\stock_app\invest\vectorizer.pkl", artifact_path="preprocessing")
    #mlflow.log_artifact(r"C:\Users\HP\Desktop\stock_app\invest\sentiment_model.pkl", artifact_path="model")
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sentiment_model_logistic",
        signature=signature,
        input_example=example_input,
        registered_model_name="logistic  Classifier",
    )


'''
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
'''


