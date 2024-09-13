from zenml import pipeline, step
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@step
def load_data() -> pd.DataFrame:
    """Loads a sample sentiment dataset."""
    data = {
        'text': ["I love this product!", "This is the worst service ever.", "I am so happy with the results.", "This is terrible."],
        'sentiment': [1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    return df

@step
def preprocess_data(data: pd.DataFrame) -> dict:
    """Preprocesses the text data into features."""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['text']).toarray()
    y = data['sentiment']
    return {'features': X, 'labels': y, 'vectorizer': vectorizer}

@step
def train_model(preprocessed_data: dict) -> MultinomialNB:
    """Trains a Naive Bayes classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_data['features'], preprocessed_data['labels'], test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    return model



@step
def evaluate_model(model: MultinomialNB, preprocessed_data: dict) -> None:
    """Evaluates the model and prints the accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_data['features'], preprocessed_data['labels'], test_size=0.2, random_state=42)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

@pipeline
def sentiment_analysis_pipeline():
    """Defines the sentiment analysis pipeline."""
    data = load_data()
    preprocessed_data = preprocess_data(data)
    model = train_model(preprocessed_data)
    evaluate_model(model, preprocessed_data)



if __name__ == "__main__":
    pipeline_run = sentiment_analysis_pipeline()
