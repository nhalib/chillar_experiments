# first neural network with keras make predictions
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
# load the dataset

# split into input (X) and output (y) variables

x = []
y = []
for i in range(700):

    x1 = random.randint(0,49)
    x2 = random.randint(0,49)

    x.append([x1/100,x2/100])
    y.append((x1 + x2)/100)

x = x
stat = True
if stat:
    # define the keras model
    model = Sequential()
    model.add(Dense(4, input_shape=(2,), activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(x, y, epochs=150, batch_size=10, verbose=0)
    # make class predictions with the model
    predictions = model.predict(x)
    # summarize the first 5 cases
    error = 0
    for i in range(len(x)):
        error += abs(predictions[i] - y[i])/y[i]

    print(error/len(x))