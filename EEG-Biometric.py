import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD

np.random.seed(0)

# Hyperparameters
training_epochs = 60   # Total number of training epochs
learning_rate = 0.01   # Initial learning rate

# Função para reduzir o learning rate conforme epochs
def scheduler(current_epoch, learning_rate):
    if current_epoch < 2:
        learning_rate = 0.01
        return learning_rate
    elif current_epoch < 37:
        learning_rate = 0.001
        return learning_rate
    else:
        learning_rate = 0.0001
        return learning_rate

# Optimizador
opt = SGD(learning_rate=LearningRateScheduler(scheduler, verbose=0), momentum=0.9)

# create a model
def create_model():
    model = Sequential()

    # Conv1
    model.add(Conv1D(96, (11), input_shape=(1920, 64), activation='relu'))
    model.add(BatchNormalization())
    # Pool1
    model.add(MaxPooling1D(strides=4))
    # Conv2
    model.add(Conv1D(128, (9), activation='relu'))
    model.add(BatchNormalization())
    # Pool2
    model.add(MaxPooling1D(strides=2))
    # Conv3
    model.add(Conv1D(256, (9), activation='relu')) 
    model.add(BatchNormalization())
    # Pool3
    model.add(MaxPooling1D(strides=2))
    # FC1
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    # FC2
    model.add(Dense(4096, activation='relu'))
    # FC3
    model.add(Dense(256))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))
    # FC4
    model.add(Dense(109, activation='softmax'))

    return model

model = create_model()
model.summary()

# model.compile(opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])