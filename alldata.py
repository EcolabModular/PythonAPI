import pandas as pd
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

class ML:
    def GetOriginalData(name):
        if(name=="temp"):
            def parser(x):
                return datetime.strptime(x,'%m/%d/%Y')
            series = pd.read_csv('/Users/olartem/Documents/2020A/IA/DatasetTemperaturas.csv',index_col=0,parse_dates=[0],date_parser=parser)
            X = series.values
            train, test = X[1:len(X)-50], X[len(X)-50:]
            
            return test.tolist()
        if(name=="mant"):
            def parser(x):
                return datetime.strptime(x,'%m/%d/%Y')
            data = pd.read_csv('/Users/olartem/Documents/2020A/IA/DatasetMantenimiento.csv',index_col=0,parse_dates=[0],date_parser=parser)

            X = data.values
            train, test  = X[1:len(X)-22], X[len(X)-22:]
            return test.tolist()
        
    def GetArimaPredictions(name):
        if(name=="temp"):
            def parser(x):
                return datetime.strptime(x,'%m/%d/%Y')
            data = pd.read_csv('/Users/olartem/Documents/2020A/IA/DatasetTemperaturas.csv',index_col=0,parse_dates=[0],date_parser=parser)
            X = data.values
            train, test  = X[1:len(X)-50], X[len(X)-50:]
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(2,1,0))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
            error = mean_squared_error(test, predictions)
            predictions.append([error])
            res = np.asarray(predictions)
            
            r = res.tolist()
            return r          
        if(name=="mant"):
            def parser(x):
                return datetime.strptime(x,'%m/%d/%Y')
            data = pd.read_csv('/Users/olartem/Documents/2020A/IA/DatasetMantenimiento.csv',index_col=0,parse_dates=[0],date_parser=parser)
            X = data.values
            train, test  = X[1:len(X)-22], X[len(X)-22:]
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(10,1,0))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)                
            error = mean_squared_error(test, predictions)
            predictions.append([error])
            res = np.asarray(predictions)
            
            r = res.tolist()
            return r               
        
    def GetARPredictions(name):
        if(name=="temp"):
            
            def parser(x):
                return datetime.strptime(x,'%m/%d/%Y')
            series = pd.read_csv('/Users/olartem/Documents/2020A/IA/DatasetTemperaturas.csv',index_col=0,parse_dates=[0],date_parser=parser)
            
            X = series.values
            train, test = X[1:len(X)-50], X[len(X)-50:]
            
            window = 25
            model = AR(train)
            model_fit = model.fit()
            coef = model_fit.params
            # walk forward over time steps in test
            history = train[len(train)-window:]
            history = [history[i] for i in range(len(history))]
            predictions = list()
            for t in range(len(test)):
                length = len(history)
                lag = [history[i] for i in range(length-window,length)]
                yhat = coef[0]
                for d in range(window):
                    yhat += coef[d+1] * lag[window-d-1]
                obs = test[t]
                predictions.append(yhat)
                history.append(obs)
            rmse = sqrt(mean_squared_error(test, predictions))
            predictions.append([rmse])
            res = np.asarray(predictions)
            
            r = res.tolist()
            return r                      
            
        if(name=="mant"):
            def parser(x):
                return datetime.strptime(x,'%m/%d/%Y')
            series = pd.read_csv('/Users/olartem/Documents/2020A/IA/DatasetMantenimiento.csv',index_col=0,parse_dates=[0],date_parser=parser)
            X = series.values
            train, test = X[1:len(X)-22], X[len(X)-22:]
            window = 18
            model = AR(train)
            model_fit = model.fit()
            coef = model_fit.params
            # walk forward over time steps in test
            history = train[len(train)-window:]
            history = [history[i] for i in range(len(history))]
            predictions = list()
            for t in range(len(test)):
                length = len(history)
                lag = [history[i] for i in range(length-window,length)]
                yhat = coef[0]
                for d in range(window):
                    yhat += coef[d+1] * lag[window-d-1]
                obs = test[t]
                predictions.append(yhat)
                history.append(obs)
            rmse = sqrt(mean_squared_error(test, predictions))
            predictions.append([rmse])
            res = np.asarray(predictions)
            
            r = res.tolist()
            return r         