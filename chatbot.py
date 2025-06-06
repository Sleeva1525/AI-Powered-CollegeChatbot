import json
import random
import pickle
import os

# Define paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, 'data', 'intents.json')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'intent_classifier.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

# Load the intents JSON
with open(INTENTS_PATH, 'r') as file:
    intents = json.load(file)

# Load trained model and vectorizer
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

def get_response(user_input):
    """
    Predict intent from user_input and return a random response.
    """
    # Transform user input using the loaded vectorizer
    user_input_vect = vectorizer.transform([user_input])
    
    # Predict the intent tag
    intent_tag = model.predict(user_input_vect)[0]

    # Find the intent responses
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    
    # Default response if no intent matched
    return "Sorry, I didn't understand that."

# Optional: If you want a CLI chatbot for quick testing
if __name__ == "__main__":
    print("College Chatbot is running! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")
