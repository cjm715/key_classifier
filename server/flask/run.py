from flasgger import Swagger
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from key_classifier_service import KeyClassifierService
import random
import os

app = Flask(__name__)
Swagger(app)
PARENT_FOLDER = ''
kss = KeyClassifierService()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_file', methods = ["POST"])
@cross_origin(origin='https://cjm715.github.io/',headers=['Content- Type','Authorization'])
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
    file_name = PARENT_FOLDER +str(random.randint(0,10000000))
    audio_file.save(file_name)
    probs, predicted_key = kss.predict(file_name)
    os.remove(file_name)
    data = {'key' : predicted_key, 'probabilities': probs}

    return jsonify(data)


if __name__ == "__main__":
    app.run(ip='0.0.0.0', port=900, ssl_context='adhoc')