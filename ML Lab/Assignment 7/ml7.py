import numpy as np

df = np.loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
                , delimiter = ',')

df

df.shape

# input data
x = df[:,:8]

# output data
y = df[:,8]

x

y

# For counting 1:(+ve) and 0:(-ve) for diabeties
from collections import Counter
Counter(y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

# input layer
model.add(Dense(12, input_dim=8, activation='relu'))

# hidden layer
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))

# outout layer
# sigmoid is for binary classification
# softmax is for multi-class classification
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(x, y, epochs=100, batch_size=10)

# Evaluate the model
model.evaluate(x,y)

model.summary()









