from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import pandas
import numpy
import pickle
import matplotlib.pyplot as plt
import tensorflow



#####Loading dataset and execute data processing
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

### Loading training model
loaded_model = tensorflow.keras.models.load_model('face_gender_classificationfinal.h5')

##ACCURACY Calculation

y_pred = loaded_model.predict(X_test)
y_pred1 = numpy.argmax(y_pred, axis = 1)
y_test1 = numpy.argmax(y_test, axis = 1)
acct=y_pred1==y_test1
acct=acct*1
acc=(sum(acct)/len(acct))*100
acc


#########################plotting Conf. Matrix
import sklearn.metrics as metrics
matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
matrix
import seaborn as sns
sns.heatmap(matrix, annot=True,fmt='.4g')



#plotting the accuracy and loss for train and validation
# summarize history for accuracy
#loading training history file
train_drawing = pickle.load(open("model_history1final.p", "rb"))
plt.plot(train_drawing['accuracy'])
plt.plot(train_drawing['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(train_drawing['loss'])
plt.plot(train_drawing['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()