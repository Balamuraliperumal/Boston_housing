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
    return render_template('index.html')

@app.route('/')
def predict():
    return ""

if __name__=="__main__":
    app.run(debug=True)