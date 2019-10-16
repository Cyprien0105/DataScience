#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:28:16 2019

@author: cyprien
"""


import pandas as pd 
import matplotlib.pyplot as plt
import os
from datetime import datetime
import random as rn
import numpy as np
import tensorflow as tf
import cv2
import io

import argparse

from efficientnet.keras import EfficientNetB3
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import backend as K
import keras



np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)


path = '/home/cyprien/REI-Dataset/REI'
print(os.listdir(path))

# generate a pandas dataframe with the column id and label, 
# the id is the name of the image whilst the label is the category of that image

df = pd.DataFrame(columns=['id','label'])

# open up all the folders within the folder REI
# - REI
#   |- bedroom
#   |- backyard
#   |- bathroom
#   |- frontyard
#   |- livingRoom
#   |- kitchen

for directory in os.listdir(path):
    for roomdirectory in os.listdir(os.path.join(path,directory)):
        current_df = pd.DataFrame({'id':[directory + "/" + roomdirectory],
                                  'label': [directory]})
        df = df.append(current_df, ignore_index=True)
    
# create a training and validation dataset
# allPictures folder contains all the REI images in one folder called img


#train_dir = '/home/cyprien/REI-Dataset/allPictures/img'

# create keras ImageDataGenerator for image preprocessing
# - REI
#   |- allPictures
#        |- img

parser = argparse.ArgumentParser(description= 'Room recognition model')
parser.add_argument('-bs', '--batch_size',type=int, default=32 , help='size of the batches')
parser.add_argument('-is', '--image_size',type=int, default=256 , help='size of the image')
parser.add_argument('-u', '--unit_number',type=int, default=120 , help='number of neurons per layer')
parser.add_argument('-dr', '--dropout_rate',type=float, default=0.5 , help='dropout rate befor the last layer')
parser.add_argument('-lr', '--learning_rate',type=float, default=0.0001 , help='learning rate for the optimizer')
parser.add_argument('-en', '--epoch_number',type=float, default=50 , help='number of epochs')
parser.add_argument('-spe', '--step_per_epoch',type=float, default=20 , help='number of steps per epoch')
args = parser.parse_args()

model_name = ("model_" +
 "batch_" + str(args.batch_size) +
 "_imSize_" + str(args.image_size) +
                   "_uNumber_" + str(args.unit_number) +
                   "_droupout_" + str(args.dropout_rate) +
                   "_lr_" + str(args.learning_rate) +
                   "_weignts")

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


# dataframe: Pandas dataframe 
# directory: string, path to the directory to read images from. 
# x_col: string, column in dataframe that contains the filenames 
# y_col: string or list, column/s in dataframe that has the target data.
# target_size: (height, width), default: (256, 256). 
# classes: optional list of classes 
# class_mode: one of "categorical", "binary", "sparse", "input", "other" or None. Default: "categorical". 
# batch_size: size of the batches of data (default: 32).
# shuffle: whether to shuffle the data (default: True)

train_generator = train_datagen.flow_from_dataframe(
    dataframe = df,
    directory = path,
    x_col="id",
    y_col="label",
    target_size=(args.image_size,args.image_size),
    subset="training",
    batch_size=args.batch_size,
    shuffle=True,
    classes = ['backyard', 'bathroom', 'bedroom', 'frontyard', 'kitchen', 'livingRoom'],
    class_mode= "categorical",
    seed=42
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe = df,
    directory = path,
    x_col="id",
    y_col="label",
    target_size=(args.image_size,args.image_size),
    subset="validation",
    batch_size=args.batch_size,
    shuffle=True,
    classes = ['backyard', 'bathroom', 'bedroom', 'frontyard', 'kitchen', 'livingRoom'],
    class_mode= "categorical",
    seed=42
)

efficient_net = EfficientNetB3(
    weights='imagenet',
    input_shape=(args.image_size,args.image_size,3),
    include_top=False,
    pooling='max'
)



model = Sequential()
model.add(efficient_net)
#model.add(resnet)
# mark loaded layers as not trainable
#for layer in model.layers:
#	layer.trainable = False
model.summary()
model.add(Dense(units = args.unit_number, activation='relu'))
model.add(Dense(units = args.unit_number, activation='relu'))
model.add(Dropout(args.dropout_rate))
model.add(Dense(units = 6, activation='softmax'))
model.summary()

model.compile(
        optimizer=Adam(lr=args.learning_rate, clipnorm=1, clipvalue=5), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
        )

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + model_name
callbacks = [
        #EarlyStopping(monitor='val_loss', patience= 10),
             ModelCheckpoint(filepath= model_name + 'best_model.h5', monitor='val_loss', save_best_only=True),
             TensorBoard(log_dir=logdir)]

history = model.fit_generator(
    train_generator,
    epochs = args.epoch_number,
    steps_per_epoch = args.step_per_epoch,
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
model.save_weights(model_name +'.h5')
print("Saved model to disk : " + model_name)

#model.load_weights(model_name + '.h5')
name = 'model_batch_32_imSize_256_uNumber_120_droupout_0.5_lr_0.0001_weignts.h5'
model.load_weights(name)

# summarize model.
model.summary()

test_dir = '/home/cyprien/REI-Dataset/immovlanpictures/'

test_datagen = ImageDataGenerator(
    rescale=1/255
)

test_generator = test_datagen.flow_from_directory(
    directory = test_dir,
    target_size=(args.image_size,args.image_size),
    batch_size=1,
    shuffle=False,
    class_mode= None,
)

# very important to use test_generator.reset() each time you evoke test_generator
#test_generator.reset()

preds = model.predict_generator(
    test_generator,
    steps=len(test_generator.filenames),
    verbose = 1

)

# Look at the predictions

#test_generator.reset()
nbPlot = len(test_generator.filenames)
    
fig=plt.figure(figsize=(15, nbPlot*5))

for i in range(nbPlot):
    fig.add_subplot(nbPlot, 2, i*2+1)
    objects = ['backyard', 'bathroom', 'bedroom', 'frontyard', 'kitchen', 'livingRoom']
    y_pos = np.arange(len(objects))
    performance = preds[i]
    plt.bar(y_pos, performance, align='center', alpha=0.75)
    plt.xticks(y_pos, objects)
    plt.ylabel('Probability')
    plt.title('Room type')


    fig.add_subplot(nbPlot, 2, i*2+2)
    path = test_dir + test_generator.filenames[i]
    img = cv2.imread(path)
    plt.imshow(img)
    plt.title(test_generator.filenames[i])

buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image = tf.image.decode_png(buf.getvalue(), channels=4)
image = tf.expand_dims(image, 0)
summary_op = tf.summary.image("plot", image)

with tf.Session() as sess:
    # Run
    summary = sess.run(summary_op)
    # Write summary
    writer = tf.summary.FileWriter(logdir)
    writer.add_summary(summary)
    writer.close()
#


for i in range(200, 400):
    fig=plt.figure(figsize=(15, 5))
    fig.add_subplot(1, 2, 1)
    objects = ['backyard', 'bathroom', 'bedroom', 'frontyard', 'kitchen', 'livingRoom']
    y_pos = np.arange(len(objects))
    performance = preds[i]
    plt.bar(y_pos, performance, align='center', alpha=0.75)
    plt.xticks(y_pos, objects)
    plt.ylabel('Probability')
    plt.title('Room type')


    fig.add_subplot(1, 2, 2)
    path = test_dir + test_generator.filenames[i]
    img = cv2.imread(path)
    plt.imshow(img)
    plt.title(test_generator.filenames[i])

