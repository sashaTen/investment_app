import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
  # WordNet data (optional, for improved lemmatization)

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/surge-ai/stock-sentiment/main/sentiment.csv')

# Extract X and y


# Function to preprocess and vectorize the text data






X_train, X_test, y_train, y_test = train_test_split(df['Tweet Text'],  df['Sentiment'], test_size=0.2, random_state=42)
# Initialize the CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=1000)

# Fit and transform the training data
X_train_vec = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_vec = vectorizer.transform(X_test)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_vec, y_train)

# Predict on the test set
y_pred = model.predict(X_test_vec)

# Print accuracy and classification report
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Load the model and vectorizer


# Example text for inference

# Transform the example text