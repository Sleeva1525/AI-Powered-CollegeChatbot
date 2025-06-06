import json
import random
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the intents JSON
with open("data/intents.json") as file:
    intents = json.load(file)

# Prepare data
inputs = []
labels = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        inputs.append(pattern)
        labels.append(intent["tag"])

# Vectorize input data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(inputs)
y = labels

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the model and vectorizer
os.makedirs("models", exist_ok=True)
with open("models/intent_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved.")
