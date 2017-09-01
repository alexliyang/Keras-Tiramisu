import os
import glob
import random
import json
import pickle

import numpy as np
import cv2

from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.layers import concatenate
from keras.layers.core import Lambda
from keras.models import Model
from keras.utils.data_utils import Sequence
from keras import backend as K

import tensorflow as tf

class WeightedCrossentropy:
    def __init__(self):
        self.class_weights = pickle.load(open('class_weights.p', 'rb'))

    def loss(self, y_true, y_pred):
        loss = K.categorical_crossentropy(y_true, y_pred)
        labels = np.argmax(y_true)
        return loss * self.class_weights[labels]
        
class MapillaryGenerator(Sequence):
    def __init__(self, folder='datasets/mapillary', mode='training', n_classes=66, batch_size=3, crop_shape=(224, 224), resize_shape=(640, 360), horizontal_flip=True):
        self.image_path_list = sorted(glob.glob(os.path.join(folder, mode, 'images/*')))
        self.label_path_list = sorted(glob.glob(os.path.join(folder, mode, 'instances/*')))
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.horizontal_flip = horizontal_flip
        
        # Preallocate memory
        if crop_shape is not None:
            self.X = np.zeros((batch_size, crop_shape[1], crop_shape[0], 3), dtype='float32')
            self.Y = np.zeros((batch_size, crop_shape[1], crop_shape[0], self.n_classes), dtype='float32')
        else:
            self.X = np.zeros((batch_size, resize_shape[1], resize_shape[0], 3), dtype='float32')
            self.Y = np.zeros((batch_size, resize_shape[1], resize_shape[0], self.n_classes), dtype='float32')
        
    def __len__(self):
        return len(self.image_path_list) // self.batch_size
        
    def __getitem__(self, i):         
        images = [cv2.resize(cv2.imread(image_path, 1), self.resize_shape) for image_path in self.image_path_list[i*self.batch_size:(i+1)*self.batch_size]]
        labels = [cv2.resize(cv2.imread(label_path, 0), self.resize_shape) for label_path in self.label_path_list[i*self.batch_size:(i+1)*self.batch_size]]
        
        n = 0
        for image, label in zip(images, labels):
            if self.crop_shape is not None:
                image, label = _random_crop(image, label, self.crop_shape)
                
            if self.horizontal_flip and (random.random() < 0.5):
                image = cv2.flip(image, 1)
                label = cv2.flip(label, 1)
                
            self.X[n] = image
            self.Y[n] = to_categorical(label, self.n_classes).reshape((label.shape[0], label.shape[1], -1))            
            n += 1
        
        return self.X, self.Y
        
    def on_epoch_end(self):
        # Shuffle dataset for next epoch
        c = list(zip(self.image_path_list, self.label_path_list))
        random.shuffle(c)
        self.image_path_list, self.label_path_list = zip(*c)
                
class Visualization(Callback):
    def __init__(self, resize_shape=(640, 360), batch_steps=10, n_gpu=1, **kwargs):
        super(Visualization, self).__init__(**kwargs)
        self.resize_shape = resize_shape
        self.batch_steps = batch_steps
        self.n_gpu = n_gpu
        self.counter = 0

        # TODO: Remove this lazy hardcoded paths
        self.test_images_list = glob.glob('datasets/mapillary/testing/images/*')
        with open('datasets/mapillary/config.json') as config_file:
            config = json.load(config_file)
        self.labels = config['labels']
        
        
    def on_batch_end(self, batch, logs={}):
        self.counter += 1
        
        if self.counter == self.batch_steps:
            self.counter = 0
            
            test_image = cv2.resize(cv2.imread(random.choice(self.test_images_list), 1), self.resize_shape)
            
            inputs = [test_image]*self.n_gpu          
            output = self.model.predict(np.array(inputs), batch_size=self.n_gpu)[0]
        
            cv2.imshow('input', test_image)
            cv2.waitKey(1)
            cv2.imshow('output', _apply_color_map(np.argmax(output, axis=-1), self.labels))
            cv2.waitKey(1)
            
class ExpDecay:
    def __init__(self, initial_lr, decay):
        self.initial_lr = initial_lr
        self.decay = decay
    
    def scheduler(self, epoch):
        return self.initial_lr * np.exp(-self.decay*epoch)

# Taken from https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
def make_parallel(model, gpu_count):
    if gpu_count < 2:
        return model
        
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(concatenate(outputs, axis=0))
            
    return Model(inputs=model.inputs, outputs=merged)
    
# Taken from Mappillary Vistas demo.py
def _apply_color_map(image_array, labels):
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    for label_id, label in enumerate(labels):
        # set all pixels with the current label to the color of the current label
        color_array[image_array == label_id] = label["color"]

    return color_array
    
def _random_crop(image, label, crop_shape):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
        x = random.randrange(image.shape[1]-crop_shape[0])
        y = random.randrange(image.shape[0]-crop_shape[1])
        
        return image[y:y+crop_shape[1], x:x+crop_shape[0], :], label[y:y+crop_shape[1], x:x+crop_shape[0]]
    else:
        raise Exception('Crop shape exceeds image dimensions!')
        
