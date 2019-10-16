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
        self.__model_type = "efficientNetB3"
        self.__image_size = 32
        self.__batch_size = 64
        self.__epoch_number = 50
        self.__model_name = "default_model"
        self.__data_directory = ""
        
        
#    np.random.seed(42)
#    rn.seed(42)
#    tf.set_random_seed(42)
#    sess = tf.Session(graph=tf.get_default_graph())
#    K.set_session(sess)
#    
    
    def setModelType(self, type_name="efficientNetB3" ):
        self.__model_type
    
        
    def setDataDirectory(self, pathToData):
        
        self.__data_directory = pathToData
        
    def setTrainConfig(self, image_size=32,
                       batch_size=64,
                       epoch_number=50,
                       model_name="default_model"):
        
        self.__image_size = image_size
        self.__batch_size = batch_size
        self.__epoch_number = epoch_number
        self.__model_name = model_name
    
    def trainModel(self):
        
        print(os.listdir(self.__data_directory))
        class_nb = len(os.listdir(self.__data_directory))

        train_datagen = ImageDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True, 
                samplewise_std_normalization=True,
                brightness_range=[0.5,1.5],
            rescale=1/255,
            validation_split=0.10,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        

        train_generator = train_datagen.flow_from_directory(
            directory = self.__data_directory,
            target_size=(self.__image_size,self.__image_size),
            subset="training",
            batch_size=self.__batch_size,
            shuffle=True,
            class_mode= "categorical",
            seed=42
        )
        
        val_generator = train_datagen.flow_from_directory(
            directory = self.__data_directory,
            target_size=(self.__image_size,self.__image_size),
            subset="validation",
            batch_size=self.__batch_size,
            shuffle=True,
            class_mode= "categorical",
            seed=42
        )
        
        if self.__model_type== "efficientNetB3":
            pretrainedModel = EfficientNetB3(
                weights='imagenet',
                input_shape=(self.__image_size, self.__image_size, 3),
                include_top=False,
                pooling='max'
            )
        
        
        
        model = Sequential()
        model.add(pretrainedModel)
        #model.add(resnet)
        # mark loaded layers as not trainable
        #for layer in model.layers:
        #	layer.trainable = False
        model.summary()
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
                     ModelCheckpoint(filepath= self.__model_name + 'best_model.h5', monitor='val_loss', save_best_only=True),
                     TensorBoard(log_dir=logdir)]
        
        history = model.fit_generator(
            train_generator,
            epochs = 50,
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
        
        plt.plot(epochs,acc,'g',label = 'Training Accuracy')
        plt.plot(epochs,val_acc,'r',label = 'Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.figure()
        
        plt.plot(epochs,loss,'g',label = 'Training loss')
        plt.plot(epochs,val_loss,'r',label = 'Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.show()
        
        # serialize weights to HDF5
        model.save_weights(self.__model_name +'.h5')
        print("Saved model to disk : " + self.__model_name)
    
