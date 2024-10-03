#

 ## ############################ For unbalanced dataset ####################

import numpy
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

Rawdara = []


data=pandas.read_csv("bank-full.csv",sep=';')
data.head(5)

#display columns
#data.columns
data.isnull().sum()

categ = ['job','marital','education','default','balance','housing','loan','contact','month','poutcome','y']
le = LabelEncoder()
data[categ] = data[categ].apply(le.fit_transform)
data.head(10)
#data Spiltting 

## For unbalanced dataset
data=data.to_numpy()
X, y = data[:, :16], data[:, 16:]
Y_Class = np_utils.to_categorical(y, 2)
X_train, X_test, y_train, y_test = train_test_split(X, Y_Class, test_size=0.25, random_state=1, stratify=y)
X_val, X_test1, y_val, y_test1 = train_test_split(X, Y_Class, test_size=0.5, shuffle=True, random_state=10)


model = Sequential()
model.add(Dense(8, activation='relu', input_dim=16))
model.add(Dropout(0.4))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

history=model.fit(X_train, y_train, batch_size=64, epochs=200, validation_data=(X_val, y_val))


y_pred = model.predict(X_test)
y_pred1 = numpy.argmax(y_pred, axis = 1)
y_test1 = numpy.argmax(y_test, axis = 1)
acct=y_pred1==y_test1
acct=acct*1
acc=(sum(acct)/len(acct))*100
acc

class_names = ['Class-0', 'Class-1']
print("#"*40)
print("\nclassifier DNN performance on test dataset\n")
print(classification_report(y_test1, y_pred1, target_names=class_names))
print("#"*40 + "\n")


# plot confusion matrix
matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
matrix
sns.heatmap(matrix, annot=True,fmt='.4g')









# ## ############################ For balanced dataset ####################

import numpy
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

Rawdara = []


data=pandas.read_csv("bank-full.csv",sep=';')
data.head(5)

#display columns
#data.columns
data.isnull().sum()

categ = ['job','marital','education','default','balance','housing','loan','contact','month','poutcome','y']
le = LabelEncoder()
data[categ] = data[categ].apply(le.fit_transform)
data.head(10)

data=data.to_numpy()
j=1
for i in range(len(data)-1):
    if ((data[i,16] == 0) and (j < 5289)):
        Rawdara.append(data[i])
        j = j+1
    elif (data[i,16] == 1):
        Rawdara.append(data[i])


Rawdara = numpy.array(Rawdara)
X, y = Rawdara[:, :16], Rawdara[:, 16:] 
Y_Class = np_utils.to_categorical(y, 2)
X_train, X_test, y_train, y_test = train_test_split(X, Y_Class, test_size=0.25, random_state=1, stratify=y)
X_val, X_test1, y_val, y_test1 = train_test_split(X, Y_Class, test_size=0.5, shuffle=True, random_state=10)


model = Sequential()
model.add(Dense(8, activation='relu', input_dim=16))
model.add(Dropout(0.4))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

history=model.fit(X_train, y_train, batch_size=64, epochs=200, validation_data=(X_val, y_val))


y_pred = model.predict(X_test)
y_pred1 = numpy.argmax(y_pred, axis = 1)
y_test1 = numpy.argmax(y_test, axis = 1)
acct=y_pred1==y_test1
acct=acct*1
acc=(sum(acct)/len(acct))*100
acc

class_names = ['Class-0', 'Class-1']
print("#"*40)
print("\nclassifier DNN performance on test dataset\n")
print(classification_report(y_test1, y_pred1, target_names=class_names))
print("#"*40 + "\n")


# plot confusion matrix
matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
matrix
sns.heatmap(matrix, annot=True,fmt='.4g')

