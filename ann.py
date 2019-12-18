from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset = np.loadtxt("C:\\Users\\Prabha\\Desktop\\My codes\\cnn\\diabetes.csv", delimiter=",", skiprows=1)
X = dataset[:,0:8]
Y = dataset[:,8]


model = Sequential()
model.add(Dense(12, activation='relu', input_dim=8))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer= 'adam', loss='squared_hinge', metrics=['accuracy'])
history = model.fit(X, Y, epochs=100, batch_size= 10)
loss, accuracy = model.evaluate(X, Y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

probabilities = model.predict(X)
predictions = [float(np.round(x)) for x in probabilities]
accuracy = np.mean(predictions == Y)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))
