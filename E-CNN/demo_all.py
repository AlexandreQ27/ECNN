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

#filepath = '/home/utseus/QYB_test/E-CNN-classifier-main/weight_qyb/cnn_model.h5'#please define our own filepath to save the weights of the probabilistic FitNet-4 classifier
#os.makedirs(os.path.dirname(filepath),exist_ok=True)
#checkpoint_callback = ModelCheckpoint(
#    filepath=filepath, monitor='val_accuracy', verbose=1,
#    save_best_only=True, save_weights_only=True,save_frequency=1)
#def save_weights(epoch,logs):
#	if epoch%1==0:
#		K.model.save_weight(fliepath)
#lambda_callback=LambdaCallback(on_epoch_end=save_weights)
#model_p.fit(x_train, y_train, batch_size=25, epochs=200, verbose=1, callbacks=[checkpoint_callback], validation_data=(x_test, #y_test), shuffle=True)

#model_p.load_weights('/home/utseus/QYB_test/E-CNN-classifier-main/weight_qyb/cnn_model.h5')# replace the path with your own
#model_p.evaluate(x_train, y_train, batch_size=25, verbose=1)
#model_p.evaluate(x_test, y_test, batch_size=25, verbose=1)


inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

#convolution stages
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

#DS layer
ED = ds_layer.DS1(prototypes,128)(flatten1)
ED_ac = ds_layer.DS1_activate(prototypes)(ED)
mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)

#Utility layer for training
outputs = utility_layer_train.DM(0.9, num_class)(mass_Dempster_normalize)


model_e = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model_e.compile(optimizer=keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004), 
              loss='CategoricalCrossentropy',
              metrics=['accuracy'])
model_e.summary()



#get the features using the probabilistic classifier
model_e.load_weights('/home/utseus/QYB_test/E-CNN-classifier-main/weight_qyb/cnn_model.h5',by_name = False) #please give our own filepath to save the weights of the probabilistic FitNet-4 classifier
feature = tf.keras.Model(inputs=[inputs], outputs=[flatten1])
x_train_feature = feature.predict(x_train)
x_test_feature = feature.predict(x_test)

#Use the features to train DS layer
inputs = tf.keras.layers.Input(128)
ED = ds_layer.DS1(prototypes,128)(inputs)
ED_ac = ds_layer.DS1_activate(prototypes)(ED)
mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)
outputs = utility_layer_train.DM(0.9, num_class)(mass_Dempster_normalize)
model_mid = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model_mid.compile(optimizer=keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004), #0.001
              loss='CategoricalCrossentropy',
              metrics=['accuracy'])
model_mid.fit(x_train_feature, y_train, batch_size=25,  epochs=2, verbose=1, validation_data=(x_test_feature, y_test), shuffle=True)

#give the trained paramters to the evidential model
model_e.load_weights('/home/utseus/QYB_test/E-CNN-classifier-main/weight_qyb/cnn_model.h5',by_name = False)#please give our own filepath to save the weights of the probabilistic FitNet-4 classifier
DSLAYER_DS1_W = tf.reshape(model_mid.layers[1].get_weights()[0], [1, 200, 128])
DSLAYER_DS1_activate_W = model_mid.layers[2].get_weights()
DSLAYER_DS2_W = model_mid.layers[3].get_weights()
model_e.layers[26].set_weights(DSLAYER_DS1_W)
model_e.layers[27].set_weights(DSLAYER_DS1_activate_W)
model_e.layers[28].set_weights(DSLAYER_DS2_W)

#fine-tune the golable weights in the evidential CNN classifier and evaluate the classifier
filepath = '/home/utseus/QYB_test/E-CNN-classifier-main/weight_qyb/ecnn_model.h5'#define our own path to save the weights of the evidential FitNet-4 classifier
checkpoint_callback = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1,
    save_best_only=True, save_weights_only=True,
    save_frequency=1)
model_e.fit(x_train, y_train, batch_size=25,  epochs=3, verbose=1, callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)
     
# aim func: cross entropy
def func(x):
  fun=0
  for i in range(len(x)):
    fun += x[i] * math.log10(x[i])
  return fun

#constraint 1: the sum of weights is 1
def cons1(x):
  return sum(x)

#constraint 2: define tolerance to imprecision
def cons2(x):
  tol = 0
  for i in range(len(x)):
    tol += (len(x) -(i+1)) * x[i] / (len(x) - 1)
  return tol

#compute the weights g for ordered weighted average aggreagtion
num_class = 10
for j in range(2,(num_class+1)):
  num_weights = j
  ini_weights = np.asarray(np.random.rand(num_weights))

  name='weight'+str(j)
  locals()['weight'+str(j)]= np.zeros([5, j])

  for i in range(5):
    tol = 0.5 + i * 0.1

    cons = ({'type': 'eq', 'fun' : lambda x: cons1(x)-1},
          {'type': 'eq', 'fun' : lambda x: cons2(x)-tol},
          {'type': 'ineq', 'fun' : lambda x: x-0.00000001}
        )
  
    res = minimize(func, ini_weights, method='SLSQP', options={'disp': True}, constraints=cons)
    locals()['weight'+str(j)][i] = res.x
    print (res.x)

#function for power set
def PowerSetsBinary(items):  
    #generate all combination of N items  
    N = len(items)  
    #enumerate the 2**N possible combinations  
    set_all=[]
    for i in range(2**N):
        combo = []  
        for j in range(N):  
            if(i >> j ) % 2 == 1:  
                combo.append(items[j]) 
        set_all.append(combo)
    return set_all



class_set=list(range(num_class))
act_set= PowerSetsBinary(class_set)
act_set.remove(act_set[0])#emptyset is not needed
act_set=sorted(act_set)
print(act_set)
print(len(act_set))
#label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
#              5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
     
utility_matrix = np.zeros([len(act_set), len(class_set)])
tol_i = 3 
#tol_i = 0 with tol=0.5, tol_i = 1 with tol=0.6, tol_i = 2 with tol=0.7, tol_i = 3 with tol=0.8, tol_i = 4 with tol=0.9
for i in range(len(act_set)):
  intersec = class_set and act_set[i]
  if len(intersec) == 1:
    utility_matrix[i, intersec] = 1
  
  else:
    for j in range(len(intersec)):
      utility_matrix[i, intersec[j]] = locals()['weight'+str(len(intersec))][tol_i, 0]
print (utility_matrix)

number_act_set = len(act_set)
inputs_pixels = IMG_WIDTH * IMG_HEIGHT
prototypes=200

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

#convolution stages
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

#DS layer
ED = ds_layer.DS1(prototypes,128)(flatten1)
ED_ac = ds_layer.DS1_activate(prototypes)(ED)
mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)

#Utility layer for testing
outputs = utility_layer_test.DM_test(num_class, number_act_set, 0.9)(mass_Dempster_normalize)


model_e_imprecise = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model_e_imprecise.compile(optimizer=keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004), 
              loss='CategoricalCrossentropy',
              metrics=['accuracy'])
model_e_imprecise.summary()

model_e_imprecise.layers[-1].set_weights(tf.reshape(utility_matrix, [1, 1023, 10]))
model_e_imprecise.load_weights('/home/utseus/QYB_test/E-CNN-classifier-main/weight_qyb/ecnn_model.h5')

resutls = tf.argmax(model_e_imprecise.predict(x_test),-1)
imprecise_results =[]
for i in range(len(resutls)):
  act_local = resutls[i]
  set_valued_results = act_set[act_local]
  imprecise_results.append(set_valued_results)
print (imprecise_results)
average_utility_imprecision = AU_imprecision.average_utility(utility_matrix, resutls, y_test_label, act_set)
print (average_utility_imprecision)

