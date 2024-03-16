import pickle
from flask import Flask,redirect,url_for,jsonify,render_template,request
import numpy as np
import pandas as pd

app = Flask(__name__)

#model loading
model = pickle.load(open("reg_model.pkl",'rb'))
scaler= pickle.load(open("scaling.pkl",'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Predicted House Price is {}".format(output[0]))

if __name__=="__main__":
    app.run(debug=True)