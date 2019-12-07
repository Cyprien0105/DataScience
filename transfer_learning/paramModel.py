#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:28:16 2019

@author: cyprien
"""


import matplotlib.pyplot as plt
import os
from datetime import datetime
#import random as rn
#import numpy as np
#import tensorflow as tf


from efficientnet.keras import EfficientNetB3
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


class ModelRetrainer:
    
    
    def __init__(self):
        self.__model_type = "EfficientNetB3"
        self.__image_size = 32
        self.__batch_size = 64
        self.__epoch_number = 50
        self.__learning_rate = 0.0001
        self.__model_name = "default_model"
        self.__data_directory = ""
        self.__image_generator = ImageDataGenerator()
        
    
    def setModelType(self, type_name="EfficientNetB3" ):
        self.__model_type = type_name
    
    def setImageGenerator(self, image_generator= ImageDataGenerator(rescale=1/255,
                                                                    validation_split=0.10)):
        self.__image_generator = image_generator
        
    def setDataDirectory(self, pathToData):
        self.__data_directory = pathToData
        
    def setTrainConfig(self, image_size=32,
                       batch_size=64,
                       epoch_number=50,
                       lr = 0.0001,
                       model_name="default_model"):
        
        self.__image_size = image_size
        self.__batch_size = batch_size
        self.__epoch_number = epoch_number
        self.__model_name = model_name
        self.__learning_rate = lr
    
    def trainModel(self):
        
        print(os.listdir(self.__data_directory))
        class_nb = len(os.listdir(self.__data_directory))
        

        train_generator = self.__image_generator.flow_from_directory(
            directory = self.__data_directory,
            target_size=(self.__image_size,self.__image_size),
            subset="training",
            batch_size=self.__batch_size,
            shuffle=True,
            class_mode= "categorical",
            seed=42
        )
        
        val_generator = self.__image_generator.flow_from_directory(
            directory = self.__data_directory,
            target_size=(self.__image_size,self.__image_size),
            subset="validation",
            batch_size=self.__batch_size,
            shuffle=True,
            class_mode= "categorical",
            seed=42
        )
        
        if self.__model_type== "EfficientNetB3":
            pretrainedModel = EfficientNetB3(
                weights='imagenet',
                input_shape=(self.__image_size, self.__image_size, 3),
                include_top=False,
                pooling='max'
            )
        else :
            raise Exception('{} model is not a member of our available pretrained models'.format(self.__model_type)) 
        
        
        
        model = Sequential()
        model.add(pretrainedModel)
        model.add(Dense(units = 120, activation='relu'))
        model.add(Dense(units = 120, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units = class_nb, activation='softmax'))
        model.summary()
        
        model.compile(
                optimizer=Adam(lr=0.0001, clipnorm=1, clipvalue=5), 
                loss='categorical_crossentropy', 
                metrics=['accuracy']
                )
        
        logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")+ "_" + self.__model_name
        callbacks = [
                #EarlyStopping(monitor='val_loss', patience= 10),
                     ModelCheckpoint(filepath= self.__model_name + '_best_model.h5', monitor='val_loss', save_best_only=True),
                     TensorBoard(log_dir=logdir)]
        
        history = model.fit_generator(
            train_generator,
            epochs = self.__epoch_number,
            #steps_per_epoch = 20,
            validation_data = val_generator,
            #validation_steps = 5,
            callbacks = callbacks
        )
        
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        epochs = range(1,len(acc) + 1)
        
        fig=plt.figure(figsize=(15, 5))
        fig.add_subplot(1, 2, 1)
        plt.plot(epochs,acc,'g',label = 'Training Accuracy')
        plt.plot(epochs,val_acc,'r',label = 'Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        fig.add_subplot(1, 2, 2)
        plt.plot(epochs,loss,'g',label = 'Training loss')
        plt.plot(epochs,val_loss,'r',label = 'Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.show()
        
        # serialize weights to HDF5
        model.save(self.__model_name +'.h5')
        print("Saved final model : " + self.__model_name +'.h5')
        print("Saved best model considering val_loss : " + self.__model_name + '_best_model.h5')
    
