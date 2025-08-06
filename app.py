from flask import Flask, request, jsonify
import os
import torch
import json
import random
import numpy as np
from nltk_utils import bag_of_words, tokenize
from model import NeuralNet

app = Flask(__name__)

# Load the trained model
FILE = "data.pth"  # Use relative path for deployment
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']
    
    # Tokenize and process the message
    tokens = tokenize(message)
    X = bag_of_words(tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).float()

    # Get prediction from model
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Calculate confidence
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    # Return response if confidence is high enough
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                return jsonify({'response': response})
    else:
        return jsonify({'response': "I'm sorry, I didn't understand that."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
