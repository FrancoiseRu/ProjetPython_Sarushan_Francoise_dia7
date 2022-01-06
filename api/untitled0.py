# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 22:32:14 2022

@author: satha
"""
# Dependencies
import os
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn import preprocessing
import traceback
import pandas as pd
import numpy as np
import pickle


# Your API definition
app = Flask(__name__, template_folder='template')


@app.route('/predict')
def loadPage():
    return render_template("Page.html")


@app.route('/predict', methods=['GET','POST'])
def predict():
    
    if modelGBR:
        try:
            
            inputQuery0= request.form["query0"]
            inputQuery1= request.form["query1"]
            inputQuery2= request.form["query2"]
            inputQuery3= request.form["query3"]
            inputQuery4= request.form["query4"]
            inputQuery5= request.form["query5"]
            inputQuery6= request.form["query6"]
            inputQuery7= request.form["query7"]
            inputQuery8= request.form["query8"]
            inputQuery9= request.form["query9"]
            inputQuery10= request.form["query10"]
            inputQuery11= request.form["query11"]
            inputQuery12= request.form["query12"]
            inputQuery13= request.form["query13"]
            inputQuery14= request.form["query14"]
            
          
            data=[[inputQuery0,inputQuery1,inputQuery2,inputQuery3,inputQuery4,
                   inputQuery5,inputQuery6,inputQuery7,inputQuery8,inputQuery9,
                   inputQuery10,inputQuery11,inputQuery12,inputQuery13,inputQuery14]]
            dataColumns=['Popularity','Nb_views','Daily_interest','Category',
                        'Nb_comments_total_CC1','Nb_comments_last_24h_CC2',
                        'Nb_comments_last_24h_CC2','Nb_comments_first_24_CC4',
                        'Diff_CC2-CC3','Selected_Time','Nb_characters',
                        'Nb_shares','H_Hour','Day_published','Day_Selected_Time']
            new_df=pd.DataFrame(data,columns=dataColumns)

            query=scaler.transform(new_df)
            prediction = modelGBR.predict(query)
            
            output=prediction
            print(prediction)
            
            return render_template("Page.html",output1=output, 
                                   query0=request.form['query0'], 
                                   query1=request.form['query1'], 
                                   query2=request.form['query2'], 
                                   query3=request.form['query3'], 
                                   query4=request.form['query4'], 
                                   query5=request.form['query5'], 
                                   query6=request.form['query6'],
                                   query7=request.form['query7'],
                                   query8=request.form['query8'],
                                   query9=request.form['query9'],
                                   query10=request.form['query10'],
                                   query11=request.form['query11'],
                                   query12=request.form['query12'],
                                   query13=request.form['query13'],
                                   query14=request.form['query14'])
        
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
    # If you don't provide any port the port will be set to 12345
    modelGBR = pickle.load(open('modelGBR.pkl','rb')) # Load "model.pkl"
    print ('Model loaded')
    modelGBR_columns = joblib.load("modelGBR_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    scaler = pickle.load(open("scaler.pkl","rb"))
    print("Model scaler loaded")
    app.run(debug=True)
