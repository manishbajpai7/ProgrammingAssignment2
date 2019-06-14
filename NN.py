# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:35:55 2019

@author: ND68005
"""

# organize imports
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd
os.chdir(r"C:/Users/ND68005/Desktop/Projects")
os.getcwd()

# seed for reproducing same results
seed = 9
np.random.seed(seed)

# load pima indians dataset
#dataset = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)
 
dataset = pd.read_csv("nn.csv")
dataset.head()

# split into input and output variables
X = dataset.drop(['Lag1Change','Area'],axis=1)
Y = dataset[['Lag1Change']]
X.head()
Y.head()

# split the data into training (67%) and testing (33%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)

# create the model
model = Sequential()
model.add(Dense(6, input_dim=6, init='uniform', activation='relu'))
model.add(Dense(6, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=200, batch_size=5, verbose=0)

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))
