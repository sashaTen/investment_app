
import pickle
import mlflow
import mlflow.pyfunc

# Load the vectorizer from the local file
vectorizer_path = r"C:\Users\HP\Desktop\cyber\cyber\vectorizer.pkl"
with open(vectorizer_path, "rb") as f:
    loaded_vectorizer = pickle.load(f)

# Load the model from MLflow
run_id = "6c903748984149eebbf751a37859ef9b"
model_uri = "runs:/fb73ff21ed0548cc874f18e5b2d0e9f5/sentiment_model_logistic"

loaded_model =  mlflow.pyfunc.load_model(model_uri)

# Prepare the input data
example_input = ['This is a sample input']
transformed_input = loaded_vectorizer.transform(example_input)

# Perform inference
prediction = loaded_model.predict(transformed_input)

print("Prediction:", prediction)

