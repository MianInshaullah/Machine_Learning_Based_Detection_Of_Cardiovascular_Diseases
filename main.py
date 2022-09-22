#############################################################################################################################
# DETECTING CARDIOVASCULAR DISEASES USING MACHINE LEARNING AND ARTIFICIAL INTELLIGENCE                                      #
# FYP BY MIAN INSHAULLAH, FAKEHA SAEED, SANA SIKANDAR, JAWERIA WAHEED                                                       #
# SUPERVISED BY DR. NASIR AHMED, ADVISED BY ENGR. NAINA SAID                                                                #
#############################################################################################################################


#LIBRARIES 

#Machine Learning and Data Science Dependencies
import pickle
import sklearn
import numpy as np
import pandas as pd 
import tensorflow as tf
from threading import Timer
from tensorflow.python.keras.models import save_model, load_model

#Web App Dependencies
import webbrowser
from flask import Flask, request, render_template

#Plotting Dependecies
import io
import random
from flask import Flask, Response, request
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG





#Deployment of Machine Learing Model using Flask
app = Flask(__name__)
model = load_model('model.h5',compile=True)
name = ['Normal','Atrial Fibrillation','Ventricular Fibrillation','Right Bundle Branch Block','Left Bundle Branch Block']

def load(filename):
    print(filename)
    test = pd.read_csv(filename)
    print(len(test))
    #test = df.drop(columns=df.columns[360], axis=1)
    test = test.iloc[:,:test.shape[1]-1].values
    test = test.reshape(1,360,1)
    return test

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    print(imagefile)
    imagepath = "" + imagefile.filename
    data = load(imagepath)
    print(data)
    y_pred = model.predict(data)
    return render_template('index.html',prediction=name[np.argmax(y_pred)])
    
def open_browser():
    webbrowser.open_new('http://127.0.0.1:8080/')


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(port=8080)