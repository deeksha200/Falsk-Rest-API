# -*- coding: utf-8 -*-

import numpy as np
import pickle
import prophet
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from flask import Flask, request, render_template


# Create application
app = Flask(__name__)

# Bind home function to URL
@app.route('/')
def show():
    return render_template('predictorform.html')

#error metric
def smape(y_true, y_pred):
    
    # CONVERT TO NUMPY
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # WHEN BOTH EQUAL ZERO, METRIC IS ZERO
    both = np.abs(y_true) + np.abs(y_pred)
    idx = np.where(both==0)[0]
    y_true[idx]=1; y_pred[idx]=1
    
    # return 100/len(y_true) * np.sum(2*np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    return np.sum(2*np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))/len(y_true)

# Bind predict function to URL
@app.route('/predict', methods =['POST'])

#predictions
def predict():
    # Load ML model
    # dic=defaultdict(list)
    smape_simple,smape_holt,smape_holt_expo,smape_holt_damp=[],[],[],[]
    smape_winter_1,smape_winter_3,smape_prophet=[],[],[]
    data=pd.read_csv("C://Users//olw01//OneDrive - ORMAE//Desktop//rp_flask_api//train.csv//train.csv")
    cfi=list(set(data['cfips']))
    cfi.sort()
    form=request.form
    data['microbusiness_density'].replace({0:0.0000001}, inplace=True)
    data['y'], lam = boxcox(data['microbusiness_density'])
    BASE= "C://Users//olw01//OneDrive - ORMAE//Desktop//rp_flask_api//"
    if request.method=='POST':
        for i in cfi:
            model_1 = pickle.load(open(f'results//simple_forecast_{i}.pckl', 'rb')) 
            model_2 = pickle.load(open(f'holt//holt_{i}.pckl', 'rb'))
            model_3 = pickle.load(open(f'holt_expo//holt_expo_{i}.pckl', 'rb'))
            model_4 = pickle.load(open(f'holt_damp//holt_damp_{i}.pckl', 'rb'))
            model_5 = pickle.load(open(f'winter_1//winter_1_{i}.pckl', 'rb'))
            model_6 = pickle.load(open(f'winter_3//winter_3_{i}.pckl', 'rb'))
            model_7 = pickle.load(open(f'prophet_data//prophet_{i}.pckl', 'rb'))

            output_1 = model_1.forecast(13)
            output_2 = model_2.forecast(13)
            output_3 = model_3.forecast(13)
            output_4 = model_4.forecast(13)
            output_5 = model_5.forecast(13)
            output_6 = model_6.forecast(13)
            future = model_7.make_future_dataframe(periods=13, freq='MS')    
            forecasts = model_7.predict(future)
            acc_forecasts = inv_boxcox(forecasts['yhat'].tail(13), lam) 
            output_7 = acc_forecasts


            smape_simple.append(smape(data['microbusiness_density'].loc[data['cfips']==i][34:39],output_1[0:5]))
            smape_holt.append(smape(data['microbusiness_density'].loc[data['cfips']==i][34:39],output_2[0:5]))
            smape_holt_expo.append(smape(data['microbusiness_density'].loc[data['cfips']==i][34:39],output_3[0:5]))
            smape_holt_damp.append(smape(data['microbusiness_density'].loc[data['cfips']==i][34:39],output_4[0:5]))
            smape_winter_1.append(smape(data['microbusiness_density'].loc[data['cfips']==i][34:39],output_5[0:5]))
            smape_winter_3.append(smape(data['microbusiness_density'].loc[data['cfips']==i][34:39],output_6[0:5]))
            smape_prophet.append(smape(data['y'].loc[data['cfips']==i][34:39],output_7[0:5]))
        # Check the output values and retrive the result with html tag based on the value
    return render_template('resultsform.html',
                               result = [np.round(np.mean(smape_simple),3),np.round(np.mean(smape_holt_expo),3),np.round(np.mean(smape_holt),3),
                                         np.round(np.mean(smape_holt_damp),3),np.round(np.mean(smape_winter_1),3),np.round(np.mean(smape_winter_3),3),
                                         np.round(np.mean(smape_prophet),3)])
    
if __name__ == '__main__':
#Run the application
    app.run(debug=True)
    
    