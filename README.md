# TensorFlow Examples: simple deep learning applications

## Introduction

In this repository examples of deep learning with tensor flow can be found. At the moment image classification using the 
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)  dataset is the only example but more will come.

The original idea is to show some clear, complete and simple example to introduce people the concepts of deep learning
and how it can be applied to real problems.

Each example is placed in its own folder isolated from every other examples, and the only dependency out of its own
folder is the common utils packages which can be found in the folder *deeplearning/utils*.


##CIFAR-10 dataset and image classification

This project builds a tensorflow model to clasify images of ten different classes using the
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The idea of the project is not new and a similar example
can be found within the []tensorflow github repository] (https://github.com/tensorflow/models), but to be honest
it is difficult to me understand the implementation available in that repository, and furthermore, my implementation
approach is different in the sense that in this repository the tensorflow APIs are used exclusively to build the 
neural network model, but everything else is purely python and numpy (like image pre-processing, dataset loading).

Apart from the model, main python scripts are provided to load an arbitrary image and classify it after execute the 
pre-processing to reduce the image dimension.  


### Project structure

The project root folder is **deeplearning/image_classification/cifar10**, and the project structure is detailed below:

- dataloader
    
    code to load the dataset, by default the *cifar-10-binary.tar.gz* needs to be located under the directory
    **resources/datasets**.
    
- networking
    - difar10_vgg16_model.py
    
        This file implements the neural network model based on vgg16, but not equal due to the small dimension of the
        images from the CIFAR10 dataset.
        
    - cifar10_train.py
    
        Main python script to train the model, and store the tensorflow model under the directory **resources/model**
        every 20 epochs.
    
- resources
    - datasets
    
        This directory should contain the binary version cifar10 dataset (cifar-10-binary.tar.gz). When invoked the
        train logic the dataset will be loaded from here. 
        
    - model
    
        It contains the pre-trained model, which will be used by the training logic to not have to start training
        the neural network from scratch, or by the inference logic to classify images.
     
- bin

    bash scripts to execute the python scripts for training, inference and to execute the test cases as well.

### Execute inference using pre-trained model to classify an input image.


### Execute training
 
 The main script to execute the training is **cifar10/bin/run_train.sh**, and it is no needed any input parameters.
 
```bash
    sh run_train.sh
```
 
 The file ***cifar10/constans.py*** can be changed to modfy some of the default parameters.
 
### References

To develop the source code in this repository the following sources have been used to learn about tensor flow and to
get inspiration.

- CIFAR-10 dataset

    https://www.cs.toronto.edu/~kriz/cifar.html
    
- TensorFlow source code examples in github 
    
    https://github.com/tensorflow/models
    
- TensorFlow official documentation
    
    https://www.tensorflow.org/tutorials/
    
- Horea Muresan's github repository with a tensorflow model to classify fruit images.
    
    https://github.com/Horea94/Fruit-Images-Dataset
    
- Yugandhar Nanda, VGG neural network explanation in Quora

    https://www.quora.com/What-is-the-VGG-neural-network

## Future work

At the moment this repository only contains one example about image classification, the plan is continue growing it, and
the current plan is adding examples of object detection using deep learning and tensorflow, apart of continue
improving the image classification example applying different techniques to improve the accuracy in the evaluation 
dataset.

## License

MIT License

Copyright (c) 2018/2019 Esteban Collado

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.