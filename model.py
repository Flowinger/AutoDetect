# Imports
# Basics
from __future__ import print_function, division
import pandas as pd
import numpy as np
import random
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import shutil
import glob
import os
import cv2
from scipy.misc import imread
# sklearn
from sklearn.metrics import f1_score, accuracy_score

import heapq
from pprint import pprint
import coremltools
from IPython.display import SVG

# keras
from keras.layers import Dense, Flatten, Embedding, Reshape, Activation, SimpleRNN, GRU, LSTM, GlobalAveragePooling1D,GlobalAveragePooling2D, Convolution1D,Convolution2D, MaxPooling1D,MaxPooling2D, Merge, Dropout
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.utils.visualize_util import model_to_dot, plot
from keras.datasets import imdb, reuters
from keras.preprocessing import sequence, image
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape

# logging (set to INFO)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def predict_img(filepath):
    '''Function to predict car model for single image.
    INPUT: File path
    OUTPUT: Five car models with highest probabilities
    '''
    # preprocess image
    img = image.load_img(filepath, target_size=(224,224))
    
    # image array -> (3,155,155)
    img = image.img_to_array(img)/255
    
    # expand by 1 dimension to -> (1,3,155,155)
    img = np.expand_dims(img, axis=0)
    
    x = model.predict(img)[0]
    
    return predict_top5(x,all_classes)

def predict_top5(pred_array, all_classes):
    '''Print out the five car models with highest probabilities
    INPUT: Array with predicted probabilities / output from model.predict()
    OUTPUT: Zipped list with car model name and probability
    '''
    # Take index of five highest values
    top5 = heapq.nlargest(5, enumerate(pred_array), key=lambda x: x[1])
    
    names = [all_classes[model[0]] for model in top5]
    zipped = list(zip(names, top5))

    return zipped


#folder: 'AD/AutoDetect/*'
def get_all_classes(folder_path):
    '''Get all car model names in a folder.'''
    all_models = []
    
    for filename in glob.iglob(folder_path):
        all_models.append(filename.split('/')[2])
        
    return dict(zip(list(range(468)),sorted(all_models)))

def scores_report(final_model):
    '''Function to return model accuracy & loss
    INPUT: model
    OUTPUT: epoch history'''
    
    history = final_model.history
    print('Highest accuracy{:.2%}'.format(max(history['acc'].values())*100))
    print('Lowest loss'+min(history['loss'].values()))
    print('Highest validation accuracy{:.2%}'.format(max(history['val_acc'].values())*100))
    print('Lowest validation loss'+min(history['val_loss'].values()))
    
    return history

# Create the pre-trained model.
base_model = InceptionV3(weights='imagenet', include_top=False)#, input_shape=(224,224,3))

x = base_model.output

# Global spatial average pooling layer
x = GlobalAveragePooling2D()(x)

# Fully-connected layer
x = Dense(468, activation='relu')(x)

# Removed Dropout layer and replaced it with a batch normalization layer since model was underfitting
#x = Dropout(0.2)(x)
x = BatchNormalization()(x)

# Logistic layer
predictions = Dense(468, activation='softmax')(x)

model = Model(base_model.input,predictions)

# Train only the top layers (randomly selected)
# -> freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model after freezing layers
adam = Adam(lr=0.0001, beta_1=0.9)
rms = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the image data
batch_size = 64
epochs = 100
img_height = 224
img_width = 224
train_samples = 127647
val_samples = 27865

# Image Augmentation: rescale, zoom, zca whitening, horizontal flip
train_datagen = ImageDataGenerator(featurewise_center=True,
        rescale=1./255,
        #shear_range=0.2,
        zoom_range=0.2,
        zca_whitening=True,
        #rotation_range=0.5,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Process images directly from the folder using generators
train_generator = train_datagen.flow_from_directory(
        'AD/AutoDetect/',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'AD/validation/',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

# Fine-tune the model
print('First fitting:')
model.fit_generator(
        train_generator,
        samples_per_epoch=train_samples // batch_size,
        nb_epoch=epochs,
        validation_data=validation_generator,
        nb_val_samples=val_samples // batch_size)

# Now, the top layers are well trained and we can start fine-tuning convolutional layers from inception V3.
# -> freeze bottom layers and train the rest of the layers.

# Look at all layers to determine how many layers we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we will freeze the first 158 layers and unfreeze/train the rest:
# 28,44,60,70,92,114,136,158,172,191

for layer in model.layers[:158]:
    layer.trainable = False
for layer in model.layers[158:]:
    layer.trainable = True

print('Second fitting:')
# Recompile the model before training it again.
model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])

# fine-tune the model
final_model = model.fit_generator(
        train_generator,
        samples_per_epoch=train_samples // batch_size,
        nb_epoch=epochs,
        validation_data=validation_generator,
        nb_val_samples=val_samples // batch_size)


scores_report(final_model)

print('Saving model...')
model.save('model_v1.h5')
print('Saving model weights...')
model.save_weights('model_v1_weights.h5')
print('Done.')


