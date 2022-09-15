from flask import Flask,request, jsonify
from pycaret.regression import *
import pandas as pd
import numpy as np

app = Flask(__name__)

model = load_model('deployment_14092022')
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen, round=0)
    output = int(prediction.Label[0])
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
