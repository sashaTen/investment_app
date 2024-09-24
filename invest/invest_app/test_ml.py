from zenml.client import Client

vectorizer_artifact = Client().get_artifact_version('ae428915-e9f8-46ad-80a3-f223ebb4e6ce')
vectorizer = vectorizer_artifact.load()

model_artifact = Client().get_artifact_version('6a4a2d3b-8ba9-4248-9c4e-752080717532')
model = model_artifact.load()



text    =   'hello '
sentiment_vectorized = vectorizer.transform([text])
prediction = model.predict(sentiment_vectorized)
print(prediction)

