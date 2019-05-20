# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 17:32:38 2018

@author: tanveer
"""

from keras import backend as K
import os
from importlib import reload
import csv

def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

set_keras_backend("cntk")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
#from sklearn.cross_validation import  train_test_split
from sklearn.model_selection  import  train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math


class lstm:
    def prediction(csv_file_import):

        np.random.seed(7)

        #load the dataset
        #msft_dataset = pd.read_csv('csv_file_import')
        #print(msft_dataset.head())
        msft_dataset = csv_file_import

        #print(msft_dataset.dtypes)


        msft_dataset['X_coordinates'] = pd.to_datetime(msft_dataset['X_coordinates'])
        #msft_dataset['Y_coordinates'] = pd.to_numeric(msft_dataset['Y_coordinates'], downcast='float')
        msft_dataset['Distance_of _sensitive_location'] = pd.to_numeric(msft_dataset['Distance_of _sensitive_location'], downcast='float')
        msft_dataset['Speed'] = pd.to_numeric(msft_dataset['Speed'], downcast='float')

        msft_dataset.set_index('X_coordinates',inplace=True)
        #msft_dataset.info()

        #print(msft_dataset)

        #sorting of dataset is done here
        msft_dataset.sort_index(inplace=True)



        #extract just close prices as that is what we want to predict
        #date and close
        #msft_close = msft_dataset['Y_coordinates']
        msft_close = msft_dataset['Distance_of _sensitive_location']
        
        #print(msft_close)

        #only close and that in sorted manner
        msft_close = msft_close.values.reshape(len(msft_close), 1)

        #print(msft_close)
        plt.plot(msft_close)
        plt.show()




        scaler = MinMaxScaler(feature_range=(0,1))

        #print(scaler)

        msft_close = scaler.fit_transform(msft_close)
        msft_close

        #print(msft_close)

        #split data into train and test
        train_size = int(len(msft_close)* 0.7)
        test_size = len(msft_close) - train_size

        msft_train, msft_test = msft_close[0:train_size, :], msft_close[train_size:len(msft_close), :]

        print('Split data into train and test: ', len(msft_train), len(msft_test))




        #need to now convert the data into time series looking back over a period of days...e.g. use last 7 days to predict price


        def create_ts(ds, series):
            X, Y =[], []
            
            for i in range(len(ds)-series - 1):
                #print(ds)
                item = ds[i:(i+series), 0]
                X.append(item)
                Y.append(ds[i+series, 0])
            return np.array(X), np.array(Y)

        series = 9


        #create_ts function called for past 7 days
        trainX, trainY = create_ts(msft_train, series)
        testX, testY = create_ts(msft_test, series)

        #original form
        #print(trainX)

        #reshaping is done
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        '''
        #after reshaping
        print(trainX)
        #inverse of reshaping is done to bring in original format
        trainX = np.reshape(trainX, (1,trainX.shape[1], trainX.shape[0]))

        #inverse of reshaping to original form
        print(trainX)

        '''
        #build the model
        model = Sequential()
        model.add(LSTM(4, input_shape=(series, 1)))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        #fit the model
        model.fit(trainX, trainY, epochs=10, batch_size=32)


        #test this model out
        trainPredictions = model.predict(trainX)
        testPredictions = model.predict(testX)
        #unscale predictions

        trainPredictions = scaler.inverse_transform(trainPredictions)
        testPredictions = scaler.inverse_transform(testPredictions)
        trainY = scaler.inverse_transform([trainY])
        testY = scaler.inverse_transform([testY])

        print(trainY)
        print(trainPredictions)
        '''
        predict=[]
        for i in ((trainPredictions)):
            predict.append(i)

        print(predict)
        '''

        #lets calculate the root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredictions[:, 0]))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredictions[:, 0]))
        print('Train score: %.2f rmse', trainScore)
        print('Test score: %.2f rmse', testScore)



        #lets plot the predictions on a graph and see how well it did
        train_plot = np.empty_like(msft_close)
        train_plot[:,:] = np.nan
        train_plot[series:len(trainPredictions)+series, :] = trainPredictions

        test_plot = np.empty_like(msft_close)
        test_plot[:,:] = np.nan
        test_plot[len(trainPredictions)+(series*2)+1:len(msft_close)-1, :] = testPredictions

        #plot on graph
        plt.plot(scaler.inverse_transform(msft_close))


        plt.plot(train_plot)
        plt.plot(test_plot)
        plt.show()
        
        
        predict=[]
        for i in ((train_plot)):
            predict.append(i)
            
        
        print(predict)
        
        myFile = open('test_frames_main6f_csv_predict_file.csv', 'w')
        
        with myFile:   
            
        #writer = csv.writer(myFile)
            myFields = ['X_coordinates_predict','Time_taken'] 
        
            writer = csv.DictWriter(myFile, fieldnames=myFields, lineterminator='\n')    
            writer.writeheader()  #used for writing the header file
        
            for i in predict:     
                
                time=i/30
                writer.writerow({'X_coordinates_predict' : i,'Time_taken':time})
            
            

        

        #print('train_plot',train_plot)
        #print('test_plot',test_plot)

