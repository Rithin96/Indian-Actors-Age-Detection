import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from imageio import imread
from skimage.transform import resize
from keras.utils.np_utils import to_categorical
# Import dataset
Train = pd.read_csv("train.csv")
Y = Train.iloc[:, 1:2].values
Test = pd.read_csv("test.csv")

#Categorical encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(Train['Class'])
y_train =to_categorical(y_train,3)

root_dir = os.path.abspath('.')
data_dir = ''

#Convert all the Train set images to the one size
train_list =[]
for i in Train.ID:
    image_path = os.path.join(data_dir,'Train1',i)
    image = imread(image_path)
    image = resize(image, (64,64)) # Size - 64 X 64
    image = image.astype('float32')
    train_list.append(image);

#Stack all the resized train images into a single list
X_Train = np.stack(train_list)

#Repeat the same resizing with the test set images
test_list =[]
for i in Test.ID:
    image_path = os.path.join(data_dir,'Test1',i)
    image1 = imread(image_path)
    image1 = resize(image1,(64,64))
    image1 = image1.astype('float32')
    test_list.append(image1);

X_test = np.stack(test_list)

#Build the Model Architecture
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
#Initialize CNN
classifier = Sequential()

#Build first convolutional layer
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation = 'relu'))

#Apply Max polling from the built covolution layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Add second convolutional layer to the pooled layer 
#This is done after checking the accuracy with single convolutional layer for improving the accuracy
classifier.add(Convolution2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

#Flatten the pooled features or converting each feature into single vectors
classifier.add(Flatten())

#Build Full connection between the neurons or inputs and the flatten layers 
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.25))

#Develope the output layer
classifier.add(Dense(units =3, activation = 'softmax'))

#Compile the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'] )
classifier.fit(X_Train,y_train,batch_size=64,epochs=15)

#Predicting the classes of the test set images   
Class_pred = classifier.predict_classes(X_test)

#Decode the encoded classes to its original class names and add to the test set
Class_pred = encoder.inverse_transform(Class_pred)

Test['class'] = Class_pred
  
        
      
        
        