import json
import pickle
import flask
import numpy as np
from flask import Flask, jsonify, request
from data_input import data_in

def load_model():
    try:
        file_name = "models/model.pkl"
        with open(file_name, 'rb') as pickled:
            data = pickle.load(pickled)
            model = data['model']
        return model
    except Exception as e:
        print(e)
        return "Error loading model"

app = Flask(__name__)
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # prepare input features
        request_json = request.get_json()
        x = request_json['input']
        x_in = np.array(x).reshape(1, -1)
        # load model
        model = load_model()
        prediction = model.predict(x_in)[0]
        response = jsonify({'prediction': prediction})
        return response, 200
    except Exception as e:
        print(e)
        return "Error in prediction"

if __name__ == "__main__":
    app.run(debug=True)