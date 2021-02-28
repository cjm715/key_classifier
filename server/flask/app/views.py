from app import app
from flask import Flask, request, jsonify
from flasgger import Swagger
from app.key_classifier_service import KeyClassifierService
import random
import os

PARENT_FOLDER = '/app/'

# log_file = open(PARENT_FOLDER + "temp/log.txt", "a")
# log_file.write(f"logging...\n")

kss = KeyClassifierService()

@app.route('/')
def index():
    return 'Hello from flask'

@app.route('/predict_file', methods = ["POST"])
def predict_file():
    """Let's make a prediction on a file.
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
    responses:
        200:
            description: The output values
    """

    audio_file = request.files.get('file')
    file_name = PARENT_FOLDER + 'temp/'+str(random.randint(0,10000000))
    audio_file.save(file_name)

    probs, predicted_key = kss.predict(file_name)

    os.remove(file_name)

    data = {'key' : predicted_key, 'probabilities': probs}

    return jsonify(data)
