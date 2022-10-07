import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")

import numpy
import tensorflow.python 
from tensorflow import keras
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10
import pandas as pd
import matplotlib.pyplot as plt

#specifying which variables we want to load the data into (normally need to do preprocessing)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#In this case, the input values are the pixels in the image, which have a value between 0 to 255. to normalize(0 to 1) the data we can simply divide the image values by 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

#he images can't be used by the network as they are, they need to be encoded first and one-hot (000001 only ever one 1 in string of 0s) encoding is best used when doing binary classification. The Numpy command to_categorical() is used to one-hot encode.

# One-hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]


model = keras.Sequential()
###         Convolutional layers, activation, dropout, pooling       ##########
#convolutional layer of 32 filters, 3x3px in size. This also adds the relu Activation layer. The dropout layer prevents overfitting
model.add(keras.layers.Conv2D(32, 3, input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

#Repeat with a larger filter to find patterns in greater detail - The exact number of pooling layers you should use will vary depending on the task you are doing, and it's something you'll get a feel for over time. Since the images are so small here already we won't pool more than twice.
model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())
    
model.add(keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

###         Flattening      ##########

model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))

###     Densely Connected Layers    ###### Since we've got fairly small images condensed into fairly small feature maps - there's no need to have multiple dense layers. A single, simple, 32-neuron layer should be quite enough
#if we had three dense layers (128, 64 and 32), the number of trainable parameters would skyrocket at 2.3M, as opposed to the 400k in this model. The larger model actually even had lower accuracy, besides the longer training times in our tests.

model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.BatchNormalization())

#In the final layer, we pass in the number of classes for the number of neurons
#class_num = 10
model.add(keras.layers.Dense(class_num, activation='softmax'))

### Compile & Optimise ####
# The optimizer is what will tune the weights in your network to approach the point of lowest loss. The Adaptive Moment Estimation (Adam) algorithm is a very commonly used optimizer, and a very sensible default optimizer to try out

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())

### Training the model ####### (using seed for reproducibility) Epoch: an arbitrary cutoff, generally defined as "one pass over the entire dataset"

seed = 21
numpy.random.seed(seed)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=64)

# Model evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

pd.DataFrame(history.history).plot()
plt.show()