from flask import Flask, render_template, request, Response,jsonify
import numpy as np
import pandas as pd
from dateutil import parser
from datetime import date
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib
import xgboost as xgb
import pickle
from IPython.core.display import display
app = Flask(__name__)

@app.route("/",methods = ['GET','POST'])
def func():
    request_type_str = request.method
    if request_type_str == 'GET':    
        return render_template('index.html',href = "static/Rossman.jpg")    
    
    else:
        startDate = request.form['startDate']   
        endDate = request.form['endDate']    
        storeList = request.form['storeList']  
        storeList = storeList.split(',') # Convert str to list
        storeList = [int(store) for store in storeList] # Convert str in list to int
        if endDate=='':
            endDate=startDate 
        
        xgbR = joblib.load('xgbR_first')
        df_test = pd.read_pickle('test_file')
        StHol_cat_map = pickle.load(open("StHol_cat.pickle", "rb"))   
        
        features_x = ['Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'StateHoliday', 'Day', 'Week', 'Month', 'Year', 'DayOfYear']
        startDate = np.datetime64(parser.parse(startDate),'ns')
        endDate = np.datetime64(parser.parse(endDate),'ns')
        range_=pd.date_range(startDate,endDate,freq='D')
        dateList = []
        for date_ in range_.values:
            date_ = date_.astype('M8[D]').astype('O')
            dateList.append(date_)
        
        test_set = df_test[(df_test.Date.isin(dateList)) & (df_test.Store.isin(storeList))].copy()
        test_set.sort_values('Date',inplace=True)  
        test_set['StateHoliday'] = test_set['StateHoliday'].map(StHol_cat_map)
        test_set['Day'] = pd.Index(test_set['Date']).day
        test_set['Week'] = pd.Index(test_set['Date']).week
        test_set['Month'] = pd.Index(test_set['Date']).month
        test_set['Year'] = pd.Index(test_set['Date']).year
        test_set['DayOfYear'] = pd.Index(test_set['Date']).dayofyear 
        dsubmit = xgb.DMatrix(test_set[features_x])
        ypred_bst = xgbR.predict(dsubmit) # Test predictions
        ypred_bst = (np.exp(ypred_bst) - 1) * 0.985
        for ind,val in enumerate(test_set.Open.values):
            if val==0:
                ypred_bst[ind]=0
        test_set['Sales'] = ypred_bst    
        return render_template('results.html',dateList = dateList, storeList = storeList, df = test_set, href = "static/Rossman.jpg")