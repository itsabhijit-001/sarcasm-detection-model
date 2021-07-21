from __future__ import division, print_function
# coding=utf-8
import sys
import os
import re
import numpy as np
import tensorflow as tf
import neattext.functions as nfx
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
# Flask utils
from flask import Flask, redirect, url_for, request, render_template,flash
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
# app.secret_key = "super secret key"
model=load_model('sarcasm_model.h5')
tokenizer=pickle.load(open('sarcasm_tokenizer.pkl', 'rb'))


@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def sarcastic_predict():
    pred=0
    if request.method == 'POST':
        # check if the post request has the file part
        context=str(request.form.get('context'))
        context=context.lower()
        context=nfx.remove_urls(context)
        context=nfx.remove_special_characters(context)
        context=nfx.remove_stopwords(context)
        text_seq=tokenizer.texts_to_sequences([context])
        text_pad=pad_sequences(text_seq,maxlen=20)
        # Make prediction
        pred=model.predict_classes(text_pad)[0][0]
        
    return render_template('index.html',is_sarcastic=pred)

if __name__ == '__main__':
    app.run(debug=True)