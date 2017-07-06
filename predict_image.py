from keras.models import load_model
import numpy as np
import heapq
import glob
from pprint import pprint
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
    
    names = [all_classes[car_model[0]] for car_model in top5]
    zipped = list(zip(names, top5))

    print('Top 5 predictions:')
    return zipped


#folder: 'AD/AutoDetect/*'
def get_all_classes(folder_path):
    '''Get all car model names in a folder.'''
    all_models = []
    
    for filename in glob.iglob(folder_path):
        all_models.append(filename.split('/')[2])
        
    return dict(zip(list(range(468)),sorted(all_models)))

all_classes = get_all_classes('/Users/flowinger/AD/AutoDetect/*')

#print(all_classes)

print('Filepath model:')
model_input = input('>')
print('Loading model...')
model = load_model(model_input)
print('Filepath image:')
image_input = input('>')
pprint(predict_img(image_input))

