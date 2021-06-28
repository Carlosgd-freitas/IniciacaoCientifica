import numpy as np
import pyedflib
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Concatenate, GlobalAveragePooling1D, Reshape, Activation, Permute, Multiply
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from scipy.signal import butter, sosfilt
from sklearn.metrics.pairwise import euclidean_distances

np.random.seed()

# Hyperparameters
batch_size = 100               # Batch Size
training_epochs = 40           # Total number of training epochs
initial_learning_rate = 0.01   # Initial learning rate

# Pre-processing Parameters
frequency = 160                # Frequency of the sampling
band_pass_1 = [1, 50]          # First filter option, 1~50Hz
band_pass_2 = [10, 30]         # Second filter option, 10~30Hz
band_pass_3 = [30, 50]         # Third filter option, 30~50Hz

# Parameters used in load_data()
train = [3, 11]                # Tasks used for training and validation
test = [7]                     # Tasks used for testing
window_size = 1920
offset = 200
distribution = 0.9             # 90% for training | 10% for validation

# Channels for some lobes of the brain
frontal_lobe   = [22, 23, 24, 25, 26, 27, 28, 29, 30,
                  31, 32, 33, 34, 35, 36, 37, 38, 39,
                  40]
motor_cortex   = [8, 9, 10, 11, 12, 13, 14]
occipital_lobe = [56, 57, 58, 59, 60, 61, 62, 63]

# 9 channels present in Yang et al. article
frontal_lobe_yang = [26, 27, 28]
motor_cortex_yang = [10, 11, 12]
occipital_lobe_yang = [61, 62, 63]
all_channels_yang = [10, 11, 12, 26, 27, 28, 61, 62, 63]

# Other Parameters
num_classes = 109              # Total number of classes
num_channels = 64              # Number of channels in an EEG signal

# Tasks:
# Task 1 - EO
# Task 2 - EC
# Task 3 - T1R1
# Task 4 - T2R1
# Task 5 - T3R1
# Task 6 - T4R1
# Task 7 - T1R2
# Task 8 - T2R2
# Task 9 - T3R2
# Task 10 - T4R2
# Task 11 - T1R3
# Task 12 - T2R3
# Task 13 - T3R3
# Task 14 - T4R3

def read_EDF(path, channels=None):
    """
    Reads data from an EDF file and returns it in a numpy array format.

    Parameters:
        - path: path of the file that will be read.
    
    Optional Parameters:
        - channels: list of channels that will be read. By default, this function reads all channels.
    """
    
    file_folder = os.path.dirname(os.path.abspath(__file__))
    new_path = os.path.join(file_folder, path)
    reader = pyedflib.EdfReader(new_path)

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

def butter_bandpass(lowcut, highcut, fs, order):
    """
    Auxiliar function for butter_bandpass_filter().

    Parameters:
        - lowcut: lowcut of the filter;
        - highcut: highcut of the filter;
        - fs: frequency of the data (sample).
        - order: order of the signal.
    """
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=12):
    """
    Band-pass filters some data and returns it.

    Parameters:
        - data: data that will be band-pass filtered;
        - lowcut: lowcut of the filter;
        - highcut: highcut of the filter;
        - fs: frequency of the data (sampling).
    
    Optional Parameters:
        - order: order of the signal. This parameter is equal to 12 by default.
    """

    sos = butter_bandpass(lowcut, highcut, fs, order)
    y = sosfilt(sos, data)
    return y

def pre_processing(content, lowcut, highcut, frequency):
    """
    Pre-processess each channel of an EEG signal using band-pass filters.

    Parameters:
        - content: the EEG signal that will be band-pass filtered;
        - lowcut: lowcut of the filter;
        - highcut: highcut of the filter;
        - frequency: frequency of the data (sampling).
    """

    channels = content.shape[0]
    c = 0

    while c < channels:
        signal = content[c, :]
        content[c] = butter_bandpass_filter(signal, lowcut, highcut, frequency)
        c += 1

    return content

def normalize_signal(content):
    """
    Normalizes each channel of an EEG signal.

    Parameters:
        - content: the EEG signal that will be normalized.
    """

    channels = content.shape[0]
    c = 0
    
    while c < channels:
        content[c] -= np.mean(content[c])
        content[c] += np.absolute(np.amin(content[c]))
        content[c] /= np.std(content[c])
        content[c] /= np.amax(content[c])
        c += 1

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

    num_subject -= 1 # Subject: 1~109 / Array Positions: 0~108

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

def load_data(folder_path, train_tasks, test_tasks, verbose=0):
    """
    Returns the processed signals and labels for training (x_train and y_train), validation (x_val and y_val) and
    testing (x_test and y_test).

    The return of this function is in the format: x_train, x_val, x_test, y_train, y_val, y_test.

    Parameters:
        - folder_path: path of the folder in which the the EDF files are stored.
        E.g. if this python script is in the same folder as the sub-folder used to store the EDF files, and this
        sub-folder is called "Dataset", then this parameter should be: './Dataset/';
        - train_tasks: list that contains the numbers of the experimental runs that will be used to create train
        and validation data;
        - test_tasks: list that contains the numbers of the experimental runs that will be used to create testing
        data.
    
    Optional Parameters:
        - verbose: if set to 1, prints what type of data (training/validation or testing) is currently being
        processed. Default value is 0.
    """

    # Processing x_train, y_train, x_val and y_val
    if(verbose):
        print('Training and Validation data is being processed...')

    x_trainL = list()
    x_valL = list()
    y_trainL = list()
    y_valL = list()

    for train_task in train_tasks:
        for i in range(1, num_classes + 1):
            train_content = read_EDF(folder_path+'S{:03d}/S{:03d}R{:02d}.edf'.format(i, i, train_task), frontal_lobe_yang)
            train_content = pre_processing(train_content, band_pass_2[0], band_pass_2[1], frequency)
            train_content = normalize_signal(train_content)
            x_trainL, y_trainL, x_valL, y_valL = signal_cropping(x_trainL, y_trainL, train_content, window_size, offset, i, num_classes, distribution, x_valL, y_valL)
    
    x_train = np.asarray(x_trainL, dtype = object).astype('float32')
    x_val = np.asarray(x_valL, dtype = object).astype('float32')
    y_train = np.asarray(y_trainL, dtype = object).astype('float32')
    y_val = np.asarray(y_valL, dtype = object).astype('float32')

    # Processing x_test and y_test
    if(verbose):
        print('Testing data is being processed...')

    x_testL = list()
    y_testL = list()

    for test_task in test_tasks:
        for i in range(1, num_classes + 1):
            test_content = read_EDF(folder_path+'S{:03d}/S{:03d}R{:02d}.edf'.format(i, i, test_task), frontal_lobe_yang)
            test_content = pre_processing(test_content, band_pass_2[0], band_pass_2[1], frequency)
            test_content = normalize_signal(test_content)
            x_testL, y_testL = signal_cropping(x_testL, y_testL, test_content, window_size, window_size, i, num_classes)

    x_test = np.asarray(x_testL, dtype = object).astype('float32')
    y_test = np.asarray(y_testL, dtype = object).astype('float32')

    # The initial format of a "x_data" (EEG signal) will be "a x num_channels x window_size", but the 
    # input shape of the CNN is "a x window_size x num_channels".
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1])
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[2], x_val.shape[1])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[1])

    # The initial format of a "y_data" (label) will be "a x 1 x num_classes", but the correct format
    # is "a x num_classes".
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

def InceptionBlock(input_img, block_index, block_type='basic', filters_sizes=(64, 96, 128, 16, 32, 128, 32), factor=1):
    """
    Creates and returns an inception block for a CNN.

    Parameters:
        - input_img: input data for the inception block;
        - block_index: index of the inception block;
    
    Optional Parameters:
        - block_type: what type of inception block will be generated. Default value is 'basic';
        - filters_sizes: tuple of filter sizes for each of the 7 convolution layers of this inception block. Default
        tuple is (64, 96, 128, 16, 32, 128, 32);
        - factor: used to multiply the number of filters used in each convolution layer simultaneously. Default
        value is 1:
    """
    result = -1

    if(block_type == 'basic' or block_type == 'flat'):
        conv1_1_1 = Conv1D(int(filters_sizes[0] * factor), 1, padding='same', activation='relu', name=f'conv1_{block_index}_1_f{factor}')(input_img)
        conv2_1_1 = Conv1D(int(filters_sizes[1] * factor), 1, padding='same', activation='relu', name=f'conv2_{block_index}_1_f{factor}')(input_img)
        conv2_1_2 = Conv1D(int(filters_sizes[2] * factor), 5, padding='same', activation='relu', name=f'conv2_{block_index}_2_f{factor}')(conv2_1_1)
        conv3_1_1 = Conv1D(int(filters_sizes[3] * factor), 1, padding='same', activation='relu', name=f'conv3_{block_index}_1_f{factor}')(input_img)
        conv3_1_2 = Conv1D(int(filters_sizes[4] * factor), 3, padding='same', activation='relu', name=f'conv3_{block_index}_2_f{factor}')(conv3_1_1)
        conv4_1_1 = Conv1D(int(filters_sizes[5] * factor), 2, padding='same', activation='relu', name=f'conv4_{block_index}_1_f{factor}')(input_img)
        maxP_3_1 = MaxPooling1D(pool_size=3, strides=1, padding="same", name=f'maxP_3_{block_index}_f{factor}')(conv4_1_1)
        conv4_1_2 = Conv1D(int(filters_sizes[6] * factor), 1, padding='same', activation='relu', name=f'conv4_{block_index}_2_f{factor}')(maxP_3_1)

        result = Concatenate(axis=2)([conv1_1_1, conv2_1_2, conv3_1_2, conv4_1_2])

        # Generated Inception Block will have a flat output
        if(block_type == 'flat'):
            result = Flatten()(result)
    else:
        print('ERROR: Invalid Inception Block type.\n')

    return result

def SEBlock(input, block_type='basic', se_ratio = 16, activation = "relu", data_format = 'channels_last', ki = "he_normal"):
    '''
    Creates and returns a squeeze & excitation block for a CNN.

    Parameters:
        - input: input data for the squeeze & excitation block;
    Optional Parameters:
        - block_type: what type of squeeze & excitation block will be generated. Default value is 'basic';
        - se_ratio : ratio for reducing the number of filters in the first dense layer of the block. Default
        value is 16;
        - activation : activation function of the first dense layer. Default value is "relu";
        - data_format : if channel axis is the first dimension of the input, this parameter should be
        'channels_first', and if it's the last dimension, this parameter should be 'channels_last'. Default
        value is 'channels_last';
        - ki : kernel initializer. Default value is "he_normal".
    '''
    x = -1

    if(block_type == 'basic' or block_type == 'flat'):
        channel_axis = -1 if data_format == 'channels_last' else 1
        input_channels = input.shape[channel_axis]

        reduced_channels = input_channels // se_ratio

        # Squeeze operation
        x = GlobalAveragePooling1D()(input)
        x = Reshape(1,1,input_channels)(x) if data_format == 'channels_first' else x
        x = Dense(reduced_channels, kernel_initializer= ki)(x)
        x = Activation(activation)(x)

        # Excitation operation
        x = Dense(input_channels, kernel_initializer=ki, activation='sigmoid')(x)
        x = Permute(dims=(3,1,2))(x) if data_format == 'channels_first' else x
        x = Multiply()([input, x])

        # Generated Squeeze and Excitation Block will have a flat output
        if(block_type == 'flat'):
            x = Flatten()(x)
    else:
        print('ERROR: Invalid Squeeze and Excitation Block type.\n')

    return x

def create_model():
    """
    Creates and returns the CNN model.
    """
    model = Sequential(name='Biometric_for_Identification')

    # Conv1
    model.add(Conv1D(96, (11), input_shape=(window_size, num_channels), activation='relu', name='Conv1'))
    model.add(BatchNormalization(name='Norm1'))
    # Pool1
    model.add(MaxPooling1D(strides=4, name='Pool1'))
    # Conv2
    model.add(Conv1D(128, (9), activation='relu', name='Conv2'))
    model.add(BatchNormalization(name='Norm2'))
    # Pool2
    model.add(MaxPooling1D(strides=2, name='Pool2'))
    # Conv3
    model.add(Conv1D(256, (9), activation='relu', name='Conv3')) 
    model.add(BatchNormalization(name='Norm3'))
    # Pool3
    model.add(MaxPooling1D(strides=2, name='Pool3'))
    # FC1
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='FC1'))
    # FC2
    model.add(Dense(4096, activation='relu', name='FC2'))
    # FC3
    model.add(Dense(256, name='FC3'))
    model.add(BatchNormalization(name='Norm4'))
    # Dropout
    model.add(Dropout(0.1, name='Drop'))
    # FC4
    model.add(Dense(num_classes, activation='softmax', name='FC4'))

    return model

def create_model_with_inception(remove_last_layer=False):
    """
    Creates and returns the CNN model using inception blocks.

    Optional Parameters:
        - remove_last_layer: if True, the model created won't have the fully connected block at the end with a
        softmax activation function.
    """

    inputs = Input(shape=(window_size, num_channels))
    block_1 = InceptionBlock(inputs, 1)
    block_2 = InceptionBlock(block_1, 2, 'flat')
    fc_1 = Dense(256, name='FC1')(block_2)
    
    # Model used for Identification
    if(remove_last_layer == False):
        fc_2 = Dense(num_classes, activation='softmax', name='FC2')(fc_1)
        model = Model(inputs=inputs, outputs=fc_2, name='Biometric_for_Identification')
        
    # Model used for Verification
    else:
        model = Model(inputs=inputs, outputs=fc_1, name='Biometric_for_Verification')

    return model

def create_model_with_SE(remove_last_layer=False):
    """
    Creates and returns the CNN model using squeeze & excitation blocks.

    Optional Parameters:
        - remove_last_layer: if True, the model created won't have the fully connected block at the end with a
        softmax activation function.
    """

    inputs = Input(shape=(window_size, num_channels))
    block_1 = SEBlock(inputs)
    block_2 = SEBlock(block_1)
    block_3 = SEBlock(block_2)
    block_4 = SEBlock(block_3)
    block_5 = SEBlock(block_4, 'flat')
    fc_1 = Dense(256, name='FC1')(block_5)
    
    # Model used for Identification
    if(remove_last_layer == False):
        fc_2 = Dense(num_classes, activation='softmax', name='FC2')(fc_1)
        model = Model(inputs=inputs, outputs=fc_2, name='Biometric_for_Identification')
        
    # Model used for Verification
    else:
        model = Model(inputs=inputs, outputs=fc_1, name='Biometric_for_Verification')

    return model

# model = create_model()
model = create_model_with_inception()
# model = create_model_with_SE()
model.summary()

# Loading the data
# x_train, x_val, x_test, y_train, y_val, y_test = load_data('./Dataset/', train, test, 1)
x_train, x_val, x_test, y_train, y_val, y_test = load_data('/media/work/carlosfreitas/IniciacaoCientifica/RedeNeural/Dataset/', train, test)

# Printing data formats
print('\nData formats:')
print(f'x_train: {x_train.shape}')
print(f'x_val: {x_val.shape}')
print(f'x_test: {x_test.shape}')
print(f'y_train: {y_train.shape}')
print(f'y_val: {y_val.shape}')
print(f'y_test: {y_test.shape}\n')

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

# Saving model weights
model.save('model_weights.h5')

# Evaluate the model to see the accuracy
print('\nEvaluating on training set...')
(loss, accuracy) = model.evaluate(x_train, y_train, verbose = 0)
print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

print('Evaluating on validation set...')
(loss, accuracy) = model.evaluate(x_val, y_val, verbose = 0)
print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

print('Evaluating on testing set...')
(loss, accuracy) = model.evaluate(x_test, y_test, verbose = 0)
print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

# Summarize history for accuracy
plt.subplot(211)
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])

# Summarize history for loss
plt.subplot(212)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.tight_layout()
plt.savefig(r'accuracy-loss.png', format='png')
plt.show()

max_loss = np.max(results.history['loss'])
min_loss = np.min(results.history['loss'])
print("Maximum Loss : {:.4f}".format(max_loss))
print("Minimum Loss : {:.4f}".format(min_loss))
print("Loss difference : {:.4f}\n".format((max_loss - min_loss)))

# Removing the last 2 layers of the model and getting the features array
# model_for_verification = Sequential(name='Biometric_for_Verification')
# for layer in model.layers[:-2]:
#     model_for_verification.add(layer)
# model_for_verification.summary()
# model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model_for_verification.load_weights('model_weights.h5', by_name=True)
# x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

# Removing the last layer of the model with inception blocks and getting the features array
model_for_verification = create_model_with_inception(True)
model_for_verification.summary()
model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
model_for_verification.load_weights('model_weights.h5', by_name=True)
x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

# Removing the last layer of the model with squeeze & excitation blocks and getting the features array
# model_for_verification = create_model_with_SE(True)
# model_for_verification.summary()
# model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model_for_verification.load_weights('model_weights.h5', by_name=True)
# x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

def one_hot_encoding_to_classes(y_data):
    """
    Takes a 2D numpy array that contains one-hot encoded labels and returns a 1D numpy array that contains
    the classes.

    Parameters:
        - y_data: 2D numpy array in the format (number of samples, number of classes).
    """

    i = 0
    j = 0
    num_samples = y_data.shape[0]
    arr = np.zeros(shape=(num_samples, 1))

    while i < num_samples:
        while y_data[i, j] != 1:
            j += 1
        arr[i] = j+1
        i += 1
    
    return arr

def calc_metrics(feature1, label1, feature2, label2, plot_det=True, path=None):
    """
    Calculates Decidability, Equal Error Rate (EER) and returns them, as well as the respective thresholds.

    Parameters:
        - feature1: one of the feature vectors;
        - label1: labels of the feature1 vector;
        - feature2: one of the feature vectors;
        - label2: labels of the feature2 vector.
    
    Optional Parameters:
        - plot_det: if set to True, plots the Detection Error Trade-Off (DET) graph. True by default;
        - path: file path that will store the Detection Error Trade-Off (DET) graph in a png file. No file path 
        is selected by default.
    """

    resolu = 5000

    feature1 = feature1.T
    xmax = np.amax(feature1,axis=0)
    xmin = np.amin(feature1,axis=0)
    x = feature1
    feature1 = (x - xmin)/(xmax - xmin)
    feature1 = feature1.T

    feature2 = feature2.T
    xmax = np.amax(feature2, axis=0)
    xmin = np.amin(feature2, axis=0)
    x = feature2
    feature2 = (x - xmin) / (xmax - xmin)
    feature2 = feature2.T

    # All against all euclidean distance
    dist = euclidean_distances(feature1, feature2)

    # Separating distances from genuine pairs and impostor pairs
    same_list = []
    dif_list = []
    for row in range(len(label1)):
        for col in range(row+1, len(label2)):
            if (label1[row] == label2[col]):
                same_list.append(dist[row, col])
            else:
                dif_list.append(dist[row, col])

    same = np.array(same_list)
    dif = np.array(dif_list)

    # Mean and standard deviation of both vectors
    mean_same = np.mean(same)
    mean_dif = np.mean(dif)
    std_same = np.std(same)
    std_dif = np.std(dif)

    # Decidability
    d = abs(mean_dif - mean_same) / np.sqrt(0.5 * (std_same ** 2 + std_dif ** 2))

    dmin = np.amin(same)
    dmax = np.amax(dif)

    # Calculate False Match Rate and False NonMatch Rate for different thresholds
    FMR = np.zeros(resolu)
    FNMR = np.zeros(resolu)
    t = np.linspace(dmin, dmax, resolu)

    for t_val in range(resolu):
        fm = np.sum(dif <= t[t_val])
        FMR[t_val] = fm / len(dif)

    for t_val in range(resolu):
        fnm = np.sum(same > t[t_val])
        FNMR[t_val] = fnm / len(same)

    # DET graph (FMR x FNMR)
    plt.plot(FMR, FNMR, color='darkorange', label='DET curve')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Match Rate')
    plt.ylabel('False NonMatch Rate')
    plt.title('Detection Error Trade-Off')
    plt.legend(loc="lower right")

    # If plot_det = True, plots FMR x FNMR
    if plot_det == True:
        plt.show()

    # If path != None, saves FMR x FNMR to a file
    if path != None:
        plt.savefig(path + r'EER.png', format='png')

    # Equal Error Rate (EER)
    abs_diffs = np.abs(FMR - FNMR)
    min_index = np.argmin(abs_diffs)
    eer = (FMR[min_index] + FNMR[min_index])/2
    thresholds = t[min_index]

    return d, eer, thresholds

# Calculating EER and Decidability
y_test_classes = one_hot_encoding_to_classes(y_test)
d, eer, thresholds = calc_metrics(x_pred, y_test_classes, x_pred, y_test_classes)
print(f'EER: {eer*100.0} %')
print(f'Decidability: {d}')
