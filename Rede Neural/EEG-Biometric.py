import numpy as np
import pyedflib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from scipy.signal import butter, sosfilt

# import pandas as pd
# from sklearn import metrics

np.random.seed()

# Tasks:
# 1: Baseline, eyes open                                -> train
# 2: Baseline, eyes closed                              -> test
# 3: Task 1 (open and close left or right fist) - Run 1 -> train
# 7: Task 1 (open and close left or right fist) - Run 2 -> test

# Hyperparameters
batch_size = 100               # Batch Size
training_epochs = 60           # Total number of training epochs - Definitivo: 60
initial_learning_rate = 0.01   # Initial learning rate

# Pre-processing Parameters
band_pass_1 = [1, 50]          # First filter
band_pass_2 = [10, 30]         # Second filter
band_pass_3 = [30, 50]         # Third filter

# Parameters used in process_signals() and load_data_EOEC()
window_size = 1920
offset = 480
distribution = 0.9             # 90% | 10%

# Other Parameters
num_classes = 10               # Total number of classes
num_channels = 64              # Number of channels in an EEG signal

def read_EDF(path, channels=None):
    """
    Reads data from an EDF file and returns it in a numpy array format.

    Parameters:
        - path: path of the file that will be read.
    
    Optional Parameters:
        - channels: number of channels that will be read. By default, this function reads all channels.
    """

    reader = pyedflib.EdfReader(path)

    if channels:
        signals = []
        signal_labels = reader.getSignalLabels()
        for c in channels:
            index = signal_labels.index(c)
            signals.append(reader.readSignal(index))
        signals = np.array(signals)
    else:
        n = reader.signals_in_file
        signals = np.zeros((n, reader.getNSamples()[0]))
        for i in np.arange(n):
            signals[i, :] = reader.readSignal(i)

    reader._close()
    del reader
    return signals

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Auxiliar function for butter_bandpass_filter().

    Parameters:
        - lowcut: lowcut of the filter;
        - highcut: highcut of the filter;
        - fs: frequency of the data (sample).

    Optional Parameters:
        - order: order of the signal. This parameter is equal to 5 by default.
    """
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Band-pass filters some data and returns it.

    Parameters:
        - data: data that will be band-pass filtered;
        - lowcut: lowcut of the filter;
        - highcut: highcut of the filter;
        - fs: frequency of the data (sample).
    
    Optional Parameters:
        - order: order of the signal. This parameter is equal to 5 by default.
    """

    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def pre_processing(content):
    """
    Pre-processess the signals of each channel of an EEG signal. The signal will be band-pass filtered in 3
    frequency bands. These bands are defined

    Parameters:
        - content: the EEG signal that will be pre-processed.
    """

    channels = content.shape[0]
    c = 0

    # First band: 1~50Hz
    while c < channels:
        signal = content[c, :]
        content[c] = butter_bandpass_filter(signal, band_pass_1[0], band_pass_1[1], content.shape[1])
        c += 1
    c = 0

    # Second band: 10~30Hz
    while c < channels:
        signal = content[c, :]
        content[c] = butter_bandpass_filter(signal, band_pass_2[0], band_pass_2[1], content.shape[1])
        c += 1
    c = 0

    # Third band: 30~50Hz
    while c < channels:
        signal = content[c, :]
        content[c] = butter_bandpass_filter(signal, band_pass_3[0], band_pass_3[1], content.shape[1])
        c += 1
    c = 0

    return content

def signal_cropping(x_data, y_data, content, window_size, offset, num_subject, num_classes, distribution=1.0, x_data_2=0, y_data_2=0):
    """
    Crops a content (EEG signal) and returns the processed signal and its' respective label using a sliding
    window.

    Considering that the format of an EEG signal is (s1,s2):
        - s1 is the number of channels in the signals (electrodes used);
        - s2 is the number of samples.

    Parameters:
        - x_data: list that stores the processed signals;
        - y_data: list that stores the processed labels;
        - content: EEG signal that will be processed;
        - window_size: size of the sliding window. Considering all channels of the EEG signal will be used,
        this number corresponds to s2;
        - offset: amount of samples the window will slide in each iteration;
        - num_subject: class of the subject;
        - num_classes: total number of classes.
    
    Optional Parameters:
        - distribution: a number in the interval (0,1]. (distribution * 100)% of the processed signals will be
        stored in x_data and y_data, and [100 - (distribution * 100)]% will be stored in x_data_2 and y_data_2.
        This number is 1.0 by default, corresponding to 100% of the data being stored in x_data and y_data, and
        x_data_2 and y_data_2 not being used nor returned; 
        - x_data_2: list that stores the processed signals;
        - y_data_2: list that stores the processed labels;
    """

    num_subject -= 1 ## Subject: 1~109 / Array Positions: 0~108

    # Checking the offset parameter
    if offset < 0:
        print('ERROR: The offset parameter can\'t be negative.')
        return x_data, y_data
    elif offset == 0:
        print('ERROR: An offset equals to 0 would result in "infinite" equal windows.')
        return x_data, y_data
    # Checking the distribution parameter
    elif distribution <= 0 or distribution > 1:
        print('ERROR: The distribution parameter needs to be in the interval (0,1].')
        return x_data, y_data
    else:
        i = window_size
        while i <= content.shape[1] * distribution:
            arr = content[: , (i-window_size):i]
            x_data.append(arr)

            arr2 = np.zeros((1,num_classes))
            arr2[0, num_subject] = 1
            y_data.append(arr2)

            i += offset

        if distribution == 1.0:
            return x_data, y_data
        
        while i <= content.shape[1]:
            arr = content[: , (i-window_size):i]
            x_data_2.append(arr)

            arr2 = np.zeros((1,num_classes))
            arr2[0, num_subject] = 1
            y_data_2.append(arr2)

            i += offset

        return x_data, y_data, x_data_2, y_data_2

def load_data_EOEC():
    """
    Returns the processed signals and labels for training (x_train and y_train), validation (x_val and y_val) and
    testing (x_test and y_test), using "Eyes Open" and "Eyes Closed" data and the following distribution:
        - 90% of "Eyes Open" data (Task 1) will be used for training;
        - 10% of "Eyes Open" data (Task 1) will be used for validation;
        - 100% of "Eyes Closed" data (Task 2) will be used for testing.

    The return of this function is in the format: x_train, x_val, x_test, y_train, y_val, y_test.
    """

    # Processing x_train, y_train, x_val e y_val
    x_trainL = list()
    x_valL = list()
    y_trainL = list()
    y_valL = list()

    for i in range(1, num_classes + 1):
        content_EO = read_EDF('./dataset/S{:03d}R01.edf'.format(i))
        content_EO = pre_processing(content_EO)
        x_trainL, y_trainL, x_valL, y_valL = signal_cropping(x_trainL, y_trainL, content_EO, window_size, offset, i, num_classes, distribution, x_valL, y_valL)
    
    x_train = np.asarray(x_trainL, dtype = object).astype('float32')
    x_val = np.asarray(x_valL, dtype = object).astype('float32')
    y_train = np.asarray(y_trainL, dtype = object).astype('float32')
    y_val = np.asarray(y_valL, dtype = object).astype('float32')

    # Processing x_test e y_test
    x_testL = list()
    y_testL = list()

    for i in range(1, num_classes + 1):
        content_EC = read_EDF('./dataset/S{:03d}R01.edf'.format(i))
        content_EC = pre_processing(content_EC)
        x_testL, y_testL = signal_cropping(x_testL, y_testL, content_EC, window_size, window_size, i, num_classes)

    x_test = np.asarray(x_testL, dtype = object).astype('float32')
    y_test = np.asarray(y_testL, dtype = object).astype('float32')

    # The initial format of a "x_data" (EEG signal) will be "a x 64 x 1920", but the input shape of the CNN is
    # "a x 1920 x 64".
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1])
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[2], x_val.shape[1])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[1])

    # The initial format of a "y_data" (label) will be "a x 1 x b". The correct format is "a x b".
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])
    y_val = y_val.reshape(y_val.shape[0], y_val.shape[2])
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])

    return x_train, x_val, x_test, y_train, y_val, y_test

def scheduler(current_epoch, learning_rate):
    """
    Lowers the learning rate hyperparameter relative to the number of epochs.
    """
    if current_epoch < 2:
        learning_rate = 0.01
        return learning_rate
    elif current_epoch < 37:
        learning_rate = 0.001
        return learning_rate
    else:
        learning_rate = 0.0001
        return learning_rate

def create_model():
    """
    Create and returns the CNN model.
    """
    model = Sequential()

    # Conv1
    model.add(Conv1D(96, (11), input_shape=(window_size, num_channels), activation='relu'))
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
    model.add(Dense(num_classes, activation='softmax'))

    return model

model = create_model()
model.summary()

# Loading the data
x_train, x_val, x_test, y_train, y_val, y_test = load_data_EOEC()

# Printing data formats
print(f'x_train: {x_train.shape}')
print(f'x_val: {x_val.shape}')
print(f'x_test: {x_test.shape}')
print(f'y_train: {y_train.shape}')
print(f'y_val: {y_val.shape}')
print(f'y_test: {y_test.shape}')

# Defining the optimizer, compiling, defining the LearningRateScheduler and training the model
opt = SGD(learning_rate = initial_learning_rate, momentum = 0.9)
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
callback = LearningRateScheduler(scheduler, verbose=0)
results = model.fit(x_train,
                    y_train,
                    batch_size = batch_size,
                    epochs = training_epochs,
                    callbacks = [callback],
                    validation_data = (x_val, y_val)
                    )

# Test the model
# prediction_values = model.predict_classes(x_test)
# prediction_values = (model.predict(x) > 0.5).astype("int32") - binary classification
prediction_values = np.argmax(model.predict(x_test), axis=-1)

# Evaluate the model to see the accuracy
print("Evaluating on training set...")
(loss, accuracy) = model.evaluate(x_train,y_train, verbose=0)
print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

print("Evaluating on testing set...")
(loss, accuracy) = model.evaluate(x_test, y_test, verbose=0)
print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

# Summarize history for accuracy and loss
# summarize history for accuracy
plt.subplot(211)
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# summarize history for loss
plt.subplot(212)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.tight_layout()
plt.show()

max_loss = np.max(results.history['loss'])
min_loss = np.min(results.history['loss'])
print("Maximum Loss : {:.4f}".format(max_loss))
print("Minimum Loss : {:.4f}".format(min_loss))
print("Loss difference : {:.4f}".format((max_loss - min_loss)))


# model = model[0:-2] # modificar

# calculate the loss on the test set
# X_pred = model.predict(X_test)
# X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
# X_pred = pd.DataFrame(X_pred, columns=test.columns)
# X_pred.index = test.index
# scored = pd.DataFrame(index=test.index)
# Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
# scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)

# Function to compute EER based on FAR and FRR values
# def find_EER(far, frr):
#    far_optimum = 0
#    frr_optimum = 0
#    x = np.absolute((np.array(far) - np.array(frr)))
#    print("diff_far_frr=", x)
#    print("min_diff_far_frr=", min(x))
#    y = np.nanargmin(x)
#    print("index of min difference=", y)
#    far_optimum = far[y]
#    frr_optimum = frr[y]
#    return far_optimum, frr_optimum

# Compute FPR, TPR, and Thresholds using ROC_curve from sci-kit learn
# fpr, tpr,  threshold = metrics.roc_curve(Y_test, scored, pos_label=1)
# fnr = 1 - tpr # get FNR , however FPR is same as FAR
# far_optimum, frr_optimum = find_EER(fpr, fnr)
# print("far_optimum = ", far_optimum)
# print("frr_optimum = ", frr_optimum)
# EER = max(far_optimum, frr_optimum)
# EER_scores.append(EER*100)
# HTER = 0.5 * (far_optimum + frr_optimum)
# HTER_scores.append(HTER*100)
# print("EER_scores: maximum of the FAR OR FRR when |FAR -FRR| is minimized ", EER_scores)
# print("HTER_scores (0.5 * (FAR+FRR)*100: ", HTER_scores)