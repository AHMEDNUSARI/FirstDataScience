import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def get_x_y(data, N, offset):
    """
    Split data into x (features) and y (target)
    """
    x, y = [], []
    for i in range(offset, len(data)):
        x.append(data[i-N:i])
        y.append(data[i])
    x = numpy.array(x)
    y = numpy.array(y)
    
    return x, y
#first data
data=pandas.read_csv("AXISBANK.csv")
data.set_index("Date", drop=False, inplace=True)
data2=data.loc[:,['Date', 'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close']]

train_data = data2[:int(.75*len(data2))][['Date', 'Close']]
train_test = data2[int(.75*len(data2)):][['Date', 'Close']]
#train and test data shape:
print("train_data.shape = " + str(train_data.shape))
print("train_test.shape = " + str(train_test.shape))

scaler = MinMaxScaler(feature_range=(0, 1))

train_data_scaled = scaler.fit_transform(numpy.array(train_data['Close']).reshape(-1,1))
x_train, y_train = get_x_y(train_data_scaled, 9, 9)

train_test_scaled = scaler.fit_transform(numpy.array(train_test['Close']).reshape(-1,1))
x_test, y_test = get_x_y(train_test_scaled, 9, 9)
print("x_train.shape = " + str(x_train.shape))
print("y_train.shape = " + str(y_train.shape))
print("x_test.shape = " + str(x_test.shape))
print("y_test.shape = " + str(y_test.shape))




# Create the LSTM network
model = Sequential()
model.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.3))

##Hidden
model.add(LSTM(units=20))
model.add(Dropout(0.3))
model.add(Dense(1))

## Compiling and training network
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=30, batch_size=1)
model.save('Stock_predicting.h')


# predicting and reverse transform preidected values
Y_predicted= model.predict(x_test)
Y_predicted_inversed=scaler.inverse_transform(Y_predicted)

#real value reverse transform
Y_true=scaler.inverse_transform(y_test)


plt.figure(figsize=(16,8))
plt.plot(Y_true)
plt.plot(Y_predicted_inversed, color='red')