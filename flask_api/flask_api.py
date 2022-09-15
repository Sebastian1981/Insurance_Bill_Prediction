from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    return "<h1>Demo API: Predicting insurance bills!</h1>"  

# import trained model
model = joblib.load("./deployment_14092022.pkl")
#model = load_model('deployment_14092022')
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = model.predict(data_unseen)
    output = int(prediction)
    return jsonify(output)

#if __name__ == '__main__':
#    app.run(debug=True)
app.run(host="0.0.0.0", port=int("5000"), debug=True)
