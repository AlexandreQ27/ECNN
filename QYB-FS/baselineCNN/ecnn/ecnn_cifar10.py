from __future__ import division, absolute_import, print_function
from common.util import *
from setup_paths import *
# import tensorflow as tf
import sys
import os
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.datasets import cifar10
# from keras.utils import np_utils
# from keras.callbacks import ModelCheckpoint
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
print("python版本:%s"% sys.version)


sys.path.append('C:/Users/14471/Desktop/QYB-FS/QiuYiBo_DS_CNN/E-CNN/libs')
import ds_layer #Dempster-Shafer layer
import utility_layer_train #Utility layer for training
import utility_layer_test #Utility layer for training
import AU_imprecision #Metric average utility for set-valued classification

from scipy.optimize import minimize
import math
import numpy as np

class CIFAR10ECNN:
    def __init__(self, mode='train', filename="ecnn_cifar10.h5", norm_mean=False, epochs=100, batch_size=32):
        self.mode = mode #train or load
        self.filename = filename
        self.norm_mean = norm_mean
        self.epochs = epochs
        self.batch_size = batch_size

        #====================== load data ========================
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cifar10_data()
        if self.norm_mean:
            self.x_train, self.x_test = normalize_mean(self.x_train, self.x_test)
        else: # linear 0-1
            self.x_train, self.x_test = normalize_linear(self.x_train, self.x_test)

        #convert labels to one_hot
        self.y_test_labels = self.y_test
        self.y_train_labels = self.y_train
        self.y_train, self.y_test = toCat_onehot(self.y_train, self.y_test, self.num_classes)
        #print(self.x_train[:5])
        #print(self.y_train[:5])
        #====================== Model =============================
        self.input_shape = self.x_train.shape[1:]
        self.model = self.build_model()

        if mode=='train':
            self.model = self.train(self.model)
        elif mode=='load':
            self.model.load_weights("{}{}".format(checkpoints_dir, self.filename),by_name=True)
        else:
            raise Exception("Sorry, select the right mode option (train/load)")

    def build_model(self):
        #================= Settings =========================
        weight_decay = 0.0005
        basic_dropout_rate = 0.1
        prototypes=200
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
        for j in range(2,(self.num_classes+1)):
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
        #print (res.x)

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



        class_set=list(range(self.num_classes))
        act_set= PowerSetsBinary(class_set)
        act_set.remove(act_set[0])#emptyset is not needed
        act_set=sorted(act_set)
     
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
        #print (utility_matrix)

        number_act_set = len(act_set)
        #================= Input ============================
        inputs = Input(shape=self.input_shape)

        #================= CONV ============================
        #convolution stages
        c1_1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_1)
        c1_3 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_2)
        c1_4 = Conv2D(48, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_3)
        c1_5 = Conv2D(48, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_4)
        bt1 = BatchNormalization()(c1_5)
        p1 = MaxPooling2D((2, 2))(bt1)
        dr1 = Dropout(0.5)(p1)
        
        c2_1 = Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(dr1)
        c2_2 = Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_1)
        c2_3 = Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_2)
        c2_4 = Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_3)
        c2_5 = Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_4)
        bt2 = BatchNormalization()(c2_5)
        p2 = MaxPooling2D((2, 2))(bt2)
        dr2 = Dropout(0.5)(p2)
        
        c3_1 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(dr2)
        c3_2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_1)
        c3_3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_2)
        c3_4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_3)
        c3_5 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_4)
        bt3 = BatchNormalization()(c3_5)
        p3 = MaxPooling2D((8, 8))(bt3)
        dr3 = Dropout(0.5)(p3)
        flatten1=Flatten()(dr3)
        
        #================= DS layer============================
        ED = ds_layer.DS1(prototypes,128)(flatten1)
        ED_ac = ds_layer.DS1_activate(prototypes)(ED)
        mass_prototypes = ds_layer.DS2(prototypes, self.num_classes)(ED_ac)
        mass_prototypes_omega = ds_layer.DS2_omega(prototypes, self.num_classes)(mass_prototypes)
        mass_Dempster = ds_layer.DS3_Dempster(prototypes, self.num_classes)(mass_prototypes_omega)
        mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)
        
        #task0 = Dropout(basic_dropout_rate + 0.3, name='l_32')(task0)
        
        #================= Output - classification head ============================
        #Utility layer for testing
        outputs = utility_layer_test.DM_test(self.num_classes, number_act_set, 0.9)(mass_Dempster_normalize)

        #================= The final model ============================
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def train(self, model):
        #================= Settings =========================
        learning_rate = 0.01
        lr_decay = 1e-6
        lr_drop = 30
        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = LearningRateScheduler(lr_scheduler)
        weights_file = "{}{}".format(checkpoints_dir, self.filename)
        model_checkpoint = ModelCheckpoint(weights_file, monitor='val_accuracy', save_best_only=True, verbose=1)
        callbacks=[reduce_lr, model_checkpoint]

        #================= Data augmentation =========================
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(self.x_train)

        #================= Train =========================
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

        historytemp = model.fit_generator(datagen.flow(self.x_train, y=self.y_train, batch_size=self.batch_size),
                                          epochs=self.epochs, callbacks=callbacks,
                                          validation_data=(self.x_test, self.y_test))
        
        #================= Save model and history =========================
        with open("{}{}_history.pkl".format(checkpoints_dir, self.filename[:-3]), 'wb') as handle:
            pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # # model.save_weights(weights_file)

        return model