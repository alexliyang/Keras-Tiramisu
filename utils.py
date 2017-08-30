import os
import glob
import random
import json

import numpy as np
import cv2
from keras.utils import to_categorical
from keras.callbacks import Callback

def apply_color_map(image_array, labels):
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    for label_id, label in enumerate(labels):
        # set all pixels with the current label to the color of the current label
        color_array[image_array == label_id] = label["color"]

    return color_array
    
def random_crop(image, label, crop_shape):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
        x = random.randrange(image.shape[1]-crop_shape[0])
        y = random.randrange(image.shape[0]-crop_shape[1])
        
        return image[y:y+crop_shape[1], x:x+crop_shape[0], :], label[y:y+crop_shape[1], x:x+crop_shape[0]]
    else:
        raise Exception('Crop shape exceeds image dimensions!')
        

def data_generator(folder='datasets/mapillary', mode='training', batch_size=32, crop_shape=(224, 224), resize_shape=(640, 360), horizontal_flip=True):
    # read in config file
    with open(os.path.join(folder, 'config.json')) as config_file:
        config = json.load(config_file)
        
    # get lists of files
    images_list = glob.glob(os.path.join(folder, mode, 'images/*'))
    labels_list = glob.glob(os.path.join(folder, mode, 'labels/*'))
    instances_list = glob.glob(os.path.join(folder, mode, 'instances/*'))
    
    # do some checks
    if (len(images_list) == 0) or (len(labels_list) == 0) or (len(instances_list) == 0):
        raise Exception('No dataset samples found!')
        
    if (len(images_list) != len(labels_list)) or (len(images_list) != len(instances_list)):
        raise Exception('Dataset is corrupted! Missing or unexpected samples')
    
    # allocate batch memory space
    if crop_shape is not None:
        X = np.zeros((batch_size, crop_shape[1], crop_shape[0], 3), dtype='float32')
        Y = np.zeros((batch_size, crop_shape[1], crop_shape[0], len(config['labels'])), dtype='float32')
    else:
        X = np.zeros((batch_size, resize_shape[1], resize_shape[0], 3), dtype='float32')
        Y = np.zeros((batch_size, resize_shape[1], resize_shape[0], len(config['labels'])), dtype='float32')
    
    while True:
        # Shuffle
        c = list(zip(images_list, labels_list, instances_list))
        random.shuffle(c)
        images_list, labels_list, instances_list = zip(*c)
        
        sample_idx = 0
        for image_path, label_path, instance_path in zip(images_list, labels_list, instances_list):
        
            # Read sample
            image = cv2.resize(cv2.imread(image_path, 1), resize_shape)
            label = cv2.resize(cv2.imread(instance_path, 0), resize_shape)
            
            # Random crop if required
            if crop_shape is not None:
                image, label = random_crop(image, label, crop_shape) 
            
            # Random horizontal flip
            if horizontal_flip and (random.random() < 0.5):
                image = cv2.flip(image, 1)
                label = cv2.flip(label, 1)

            # Add to batch
            X[sample_idx] = image
            Y[sample_idx] = to_categorical(label, len(config['labels'])).reshape((label.shape[0], label.shape[1], -1))
            sample_idx += 1
            
            # Done with batch
            if sample_idx == batch_size:
                sample_idx = 0
                yield X,Y
                
class Visualization(Callback):
    def __init__(self, resize_shape=(640, 360), batch_steps=10, name='', **kwargs):
        super(Visualization, self).__init__(**kwargs)
        self.resize_shape = resize_shape
        self.batch_steps = batch_steps
        self.name = name
        self.counter = 0

        self.test_images_list = glob.glob('datasets/mapillary/testing/images/*')
        with open('datasets/mapillary/config.json') as config_file:
            config = json.load(config_file)
        self.labels = config['labels']
        
        
    def on_batch_end(self, batch, logs={}):
        self.counter += 1
        
        if self.counter == self.batch_steps:
            self.counter = 0
            
            test_image = cv2.resize(cv2.imread(random.choice(self.test_images_list), 1), self.resize_shape)                
            output = self.model.predict(np.array([test_image]), batch_size=1)[0]
        
            cv2.imshow(self.name + ': input', test_image)
            cv2.waitKey(1)
            cv2.imshow(self.name + ': output', apply_color_map(np.argmax(output, axis=-1), self.labels))
            cv2.waitKey(1)
        
