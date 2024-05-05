import pandas as pd
from flask import Flask, jsonify, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl",'rb'))

@app.route("/predict",methods = ["POST"])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = model.predict(query_df)
    return jsonify({"Prediction " : list(prediction)})

if __name__ == '__main__':
    app.run(debug=True)