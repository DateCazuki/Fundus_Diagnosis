import tensorflow as tf
import keras
from keras.preprocessing.image import load_img, img_to_array, array_to_img
#from keras.preprocessing.image import random_rotation, random_shift, random_zoom
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils

import glob         #ファイル読み込みに使用
import sys
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import cv2
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import mymodule
from mymodule import BatchGenerator
from mymodule import Make_Raw_List2
from mymodule import plot_loss_accuracy_graph

# Hyper Parameter
BATCH_SIZE   = 32
NUM_CLASSES  = 8
EPOCHS       = 3
class_name   = ["AMD","RYO","Gla","MH","RD","RP","DM","Normal"]

#imgsize
IMG_ROWS     = 768
IMG_COLS     = 1024
IMG_CHANNELS = 3

#Input File
csv_filename = "data.csv"
image_folder = "./img"

#Output File
best_model_path = 'best_model_path.h5'
final_model_path = 'final_model_path.h5'
output_file = 'model.summary.txt'


img_path_array,teacher_array = Make_Raw_List2(image_folder,csv_filename)

#with open("image+teacher.csv","w") as fp:
#    if teacher_array.shape[0] == img_path_array.shape[0]:
#        print(teacher_array.shape[0])
#    for i in range(teacher_array.shape[0]):
#        fp.write(img_path_array[i]+','+str(teacher_array[i])+'\n')


train_img_path,test_img_path,train_teacher,test_teacher = train_test_split(img_path_array,teacher_array,
                                                                test_size=0.2,shuffle=True,random_state=0)

train_img_path,val_img_path,train_teacher,val_teacher = train_test_split(train_img_path,train_teacher,
                                                                test_size=0.25,shuffle=True,random_state=0)



if keras.backend.image_data_format == "channels_first":
    IMG_SHAPE   = (IMG_CHANNELS,IMG_ROWS,IMG_COLS)
else:
    IMG_SHAPE   = (IMG_ROWS,IMG_COLS,IMG_CHANNELS)

print("debugpoint2")

train_batch_generator = BatchGenerator(train_img_path,train_teacher,IMG_SHAPE,BATCH_SIZE)
val_batch_generator = BatchGenerator(val_img_path,val_teacher,IMG_SHAPE,BATCH_SIZE)
test_batch_generator = BatchGenerator(test_img_path,test_teacher,IMG_SHAPE,BATCH_SIZE)

print("debugpoint3")

#CNN model 構築
model = Sequential()
model.add(MaxPooling2D(pool_size=(2,2), input_shape=IMG_SHAPE))
model.add(Conv2D(32,kernel_size=(8,8),activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Conv2D(64,kernel_size=(16,16),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,kernel_size=(3,3),activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
#model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32,activation='relu'))
model.add(Dense(NUM_CLASSES,activation='softmax'))

model.summary()

print("debugpoint4")


with open(output_file,"w") as fp:
    model.summary(print_fn=lambda x:fp.write(x + "\r\n"))

model.compile(optimizer=keras.optimizers.Nadam(),
            loss=keras.losses.categorical_crossentropy,
            metrics=['accuracy'])

chk_point = keras.callbacks.ModelCheckpoint(filepath=best_model_path,monitor='val_accuracy',
                                            verbose=1,save_best_only=True, save_weights_only=False,
                                            mode='max',period=1)

print("debugpoint5")

fit_record = model.fit_generator(train_batch_generator,epochs=EPOCHS,
                                steps_per_epoch=train_batch_generator.batches_per_epoch,
                                verbose=1,validation_data=val_batch_generator,
                                validation_steps=test_batch_generator.batches_per_epoch,
                                shuffle=False,callbacks=[chk_point])

print("debugpoint6")

model.save(final_model_path)

## 学習したモデルのテスト
test_model = keras.models.load_model(best_model_path)

test_result=test_model.predict_generator(test_batch_generator, steps=None, max_queue_size=10,
                                        workers=1,use_multiprocessing=False, verbose=1)

predict = np.argmax(test_result,axis=1)
true_label = np.argmax(test_teacher,axis=1)

print("debugpoint7")

print("test_result")
print(test_result)
print("test_result.shape={}".format(test_result.shape))

print("predict")
print(predict)
print("predict.shape={}".format(predict.shape))

print("true_label")
print(true_label)
print("true_label.shape={}".format(true_label.shape))

print(classification_report(true_label,predict,target_names=class_name))

plot_loss_accuracy_graph(fit_record)
