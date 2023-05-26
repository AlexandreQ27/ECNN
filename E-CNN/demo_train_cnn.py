import tensorflow as tf
import sys
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from keras.utils import np_utils
#from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LambdaCallback
print(tf.__version__)
print("python版本:%s"% sys.version)


sys.path.append('/home/utseus/QYB_test/E-CNN-classifier-main/libs')
import ds_layer #Dempster-Shafer layer
import utility_layer_train #Utility layer for training
import utility_layer_test #Utility layer for training
import AU_imprecision #Metric average utility for set-valued classification

from scipy.optimize import minimize
import math
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train=x_train.astype("float32") / 255.0
x_test=x_test.astype("float32") / 255.0
y_train_label = y_train
y_test_label = y_test
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
    
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 3
num_class=10
inputs_pixels = IMG_WIDTH * IMG_HEIGHT
prototypes=200

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

c1_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_1)
c1_3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_2)
c1_4 = tf.keras.layers.Conv2D(48, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_3)
c1_5 = tf.keras.layers.Conv2D(48, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_4)
bt1 = tf.keras.layers.BatchNormalization()(c1_5)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(bt1)
dr1 = tf.keras.layers.Dropout(0.5)(p1)


c2_1 = tf.keras.layers.Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(dr1)
c2_2 = tf.keras.layers.Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_1)
c2_3 = tf.keras.layers.Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_2)
c2_4 = tf.keras.layers.Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_3)
c2_5 = tf.keras.layers.Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_4)
bt2 = tf.keras.layers.BatchNormalization()(c2_5)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(bt2)
dr2 = tf.keras.layers.Dropout(0.5)(p2)

c3_1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(dr2)
c3_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_1)
c3_3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_2)
c3_4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_3)
c3_5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_4)
bt3 = tf.keras.layers.BatchNormalization()(c3_5)
p3 = tf.keras.layers.MaxPooling2D((8, 8))(bt3)
dr3 = tf.keras.layers.Dropout(0.5)(p3)

flatten1=tf.keras.layers.Flatten()(dr3)

outputs = tf.keras.layers.Dense(num_class, activation='softmax')(flatten1)

model_p = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model_p.compile(optimizer=keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004), #lr=0.005
              loss='CategoricalCrossentropy',
              metrics=['accuracy'])
model_p.summary()

filepath = '/home/utseus/QYB_test/E-CNN-classifier-main/weight_qyb/cnn_model.h5'#please define our own filepath to save the weights of the probabilistic FitNet-4 classifier
os.makedirs(os.path.dirname(filepath),exist_ok=True)
checkpoint_callback = ModelCheckpoint(
    filepath=filepath, monitor='val_accuracy', verbose=1,
    save_best_only=True, save_weights_only=True,save_frequency=1)
#def save_weights(epoch,logs):
#	if epoch%1==0:
#		K.model.save_weight(fliepath)
#lambda_callback=LambdaCallback(on_epoch_end=save_weights)
model_p.fit(x_train, y_train, batch_size=25, epochs=200, verbose=1, callbacks=[checkpoint_callback], validation_data=(x_test, #y_test), shuffle=True)

model_p.load_weights('/home/utseus/QYB_test/E-CNN-classifier-main/weight_qyb/cnn_model.h5')# replace the path with your own
model_p.evaluate(x_train, y_train, batch_size=25, verbose=1)
model_p.evaluate(x_test, y_test, batch_size=25, verbose=1)






