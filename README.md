# Keras-Tiramisu
A Keras implementation of [The One Hundred Layers Tiramisu](https://arxiv.org/abs/1611.09326)

## Requirements

* Python (tested with 2.7, 3.x should work too)
* OpenCV 3.2.0
* Keras 2.0.6
* Tensorflow 1.0.1
* Numpy 1.13.1

## Usage

The ```train``` script will start a training session with the [Mapillary Vistas dataset](https://www.mapillary.com/dataset/vistas)

Remember to set the execution privileges: ```sudo chmod +x ./train```

Then, run by issuing: ```./train```

For more training options see ```./train --help```

## Outro

Note that the Mapillary Vistas dataset is expected to be under ```datasets/mapillary```

Soon I will add the weights file and a test script
