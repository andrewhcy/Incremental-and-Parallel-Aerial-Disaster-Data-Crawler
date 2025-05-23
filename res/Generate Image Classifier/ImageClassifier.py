import tensorflow as tf
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
#from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential, load_model, save_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.metrics import Precision, Recall, BinaryAccuracy
from keras.constraints import MaxNorm

def checkGPUorCPU():
    # Avoid OOM errors by setting GPU Memory Consumption Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True) #Limit GPU Memory Consumption Growth to prevent out of memory errors
    print(tf.config.list_physical_devices('GPU'))

def trainModel():
    data = tf.keras.utils.image_dataset_from_directory('Data') #Read data from specified folder

    data = data.map(lambda x,y: (x/255, y))
    data.as_numpy_iterator().next()

    #Initialise ratio of data for the specified uses 
    train_size = int(len(data)*.7) 
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)

    #Split the data according to the specified ratios
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)

    #Generate CNN model architecture
    model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3), kernel_constraint=MaxNorm(3)))
    model.add(MaxPooling2D())
    #Overfitting management
    BatchNormalization()
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3,3), 1, activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(MaxPooling2D())
    BatchNormalization()
    model.add(Dropout(0.5))

    model.add(Conv2D(16, (3,3), 1, activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(MaxPooling2D())
    BatchNormalization()
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    #model.summary()

    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
    mcp_save = ModelCheckpoint('models/bestModel' + str(counter) + '.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min')

    hist = model.fit(train, epochs=100, validation_data=val, callbacks=[tensorboard_callback, mcp_save, reduce_lr_loss, earlyStopping])
    return hist, model, test

def tester(model):
    img = cv2.imread('test.jpg') #Read test image
    resize = tf.image.resize(img, (256,256)) #Resize test image
    yhat = model.predict(np.expand_dims(resize/255, 0))
    print(yhat)

    if yhat > 0.5: 
        print('Predicted class is Not a Disaster')
    else:
        print('Predicted class is a Disaster')

    model.save(os.path.join('models','disasterOrNot' + str(counter) + '.h5'))

    new_model = load_model('models/disasterOrNot' + str(counter) + '.h5')

    yhatnew = new_model.predict(np.expand_dims(resize/255, 0))

    if yhatnew > 0.5: 
        print('Predicted class is Not a Disaster')
    else:
        print('Predicted class is a Disaster')

if __name__ == "__main__":
    counter = 0
    for i in range(50):
        checkGPUorCPU()
        hist, model, test = trainModel()
        #tester(model)
        counter += 1
