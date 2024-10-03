# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:02:45 2022

@author: toshiba
"""
from PIL import Image
from matplotlib import pyplot as plt
from skimage.util import img_as_ubyte
import pickle
import numpy 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
import pandas
import numpy
import pickle
import os
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt


data=pandas.read_csv("age_gender.csv")

data2=data.drop(['age','ethnicity','img_name'], axis=1)
#if there is null
data2.isnull().sum()

#Reshaping PICS
PICs=pandas.Series(data2['pixels'])
PICs_splited=PICs.apply(lambda PICs:PICs.split(" "))
PICs_splited2=PICs_splited.apply(lambda PICs_splited: numpy.array(list(map(lambda z: numpy.int(z),PICs_splited))))
PICs_splited3=numpy.stack(numpy.array(PICs_splited2),axis=0)
x=numpy.reshape(PICs_splited3,(-1,48,48))

Labels= numpy.array(data2['gender'])
Y_Class = np_utils.to_categorical(Labels, 2)
X_train,X_test,Y_train,y_test = train_test_split(x, Y_Class,test_size=.25, shuffle=True, random_state=3)
X_val, X_test1, y_val, y_test1 = train_test_split(x, Y_Class, test_size=0.5, shuffle=True, random_state=10)

#Model Olusturma
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(48,48,1)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(2, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()
# training the model for 10 epochs
history_cnn300150=model.fit(X_train, Y_train, epochs=60,batch_size=30,  validation_data=(X_val, y_val))


#saving Model and model's histort
model.save('face_gender_classificationfinal.h5')
pickle.dump(history_cnn300150.history,open("model_history1final.p","wb"))

#Accuracy
y_pred = model.predict(X_test)
y_pred1 = numpy.argmax(y_pred, axis = 1)
y_test1 = numpy.argmax(y_test, axis = 1)
acct=y_pred1==y_test1
acct=acct*1
acc=(sum(acct)/len(acct))*100
acc


