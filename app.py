from flask import Flask, request, jsonify
import os
import torch
import json
import random
import numpy as np
import nltk
from nltk_utils import bag_of_words, tokenize
from model import NeuralNet

import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Starting application initialization")
    
    # Download NLTK punkt tokenizer if not present
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK punkt tokenizer already installed")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer")
        nltk.download('punkt')
    
    # Load the trained model
    logger.info("Loading model from data.pth")
    FILE = "data.pth"
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
    logger.info("Model loaded successfully")
    
    # Load intents
    logger.info("Loading intents from intents.json")
    with open('intents.json', 'r') as f:
        intents = json.load(f)
    logger.info("Intents loaded successfully")
    
except Exception as e:
    logger.error(f"Fatal error during initialization: {str(e)}", exc_info=True)
    # Rethrow to crash the app - Render will show the error in logs
    raise

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
