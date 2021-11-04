import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
# from pyedflib import EdfReader
from numpy import savetxt, loadtxt
from scipy.signal import butter, sosfilt, firwin, filtfilt
from sklearn.metrics.pairwise import euclidean_distances
import tensorflow.keras as keras

# def read_EDF(path, channels = None):
#     """
#     Reads data from an EDF file and returns it in a numpy array format.

#     Parameters:
#         - path: path of the file that will be read.
    
#     Optional Parameters:
#         - channels: list of channel codes that will be read. By default, this function reads all channels.
#         The list containing all channel codes is: ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
#         'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.',
#         'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..',
#         'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..',
#         'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..',
#         'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']
#     """
    
#     file_folder = os.path.dirname(os.path.abspath(__file__))
#     new_path = os.path.join(file_folder, path)
#     reader = EdfReader(new_path)

#     if channels:
#         signals = []
#         signal_labels = reader.getSignalLabels()
#         for c in channels:
#             index = signal_labels.index(c)
#             signals.append(reader.readSignal(index))
#         signals = np.array(signals)
#     else:
#         n = reader.signals_in_file
#         signals = np.zeros((n, reader.getNSamples()[0]))
#         for i in np.arange(n):
#             signals[i, :] = reader.readSignal(i)

#     reader._close()
#     del reader
#     return signals

# def create_csv_database_from_edf(edf_folder_path, csv_folder_path, num_classes, channels = None):
#     """
#     Creates a database with CSV files from the original Physionet database, that contains EEG Signals stored
#     in EDF files.

#     Parameters:
#         - edf_folder_path: path of the folder in which the the EDF files are stored;
#         - csv_folder_path: path of the folder in which the the CSV files will be stored;
#         - num_classes: total number of classes (individuals).
    
#     Optional Parameters:
#         - channels: list of channel codes that will be read from the edf files. By default,
#         this function reads all channels. The list containing all channel codes is:
#         ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..',
#         'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
#         'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..',
#         'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
#         'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
#         'O1..', 'Oz..', 'O2..', 'Iz..']
#     """
#     if(os.path.exists(csv_folder_path) == False):
#         os.mkdir(csv_folder_path)

#     subject = 1
#     while(subject <= num_classes):
#         if(os.path.exists(csv_folder_path+'/S{:03d}'.format(subject)) == False):
#             os.mkdir(csv_folder_path+'/S{:03d}'.format(subject))

#         task = 1
#         while(task <= 14):
#             data = read_EDF(edf_folder_path+'S{:03d}/S{:03d}R{:02d}.edf'.format(subject, subject, task), channels)
#             savetxt(csv_folder_path+'/S{:03d}/S{:03d}R{:02d}.csv'.format(subject, subject, task), data,
#                     fmt='%d', delimiter=',')
#             task += 1

#         subject += 1

def load_data(folder_path, train_tasks, test_tasks, file_type, num_classes, channels = None, verbose = 0):
    """
    Loads and returns lists containing raw signals used for training (train_content) and testing (test_content).

    The return of this function is in the format: train_content, test_content.

    Parameters:
        - folder_path: path of the folder in which the the EDF files are stored.
        E.g. if this python script is in the same folder as the sub-folder used to store the EDF files, and this
        sub-folder is called "Dataset", then this parameter should be: './Dataset/';
        - train_tasks: list that contains the numbers of the experimental runs that will be used to create train
        and validation data;
        - test_tasks: list that contains the numbers of the experimental runs that will be used to create testing
        data;
        - file_type: extension of the files that contains the EEG signals. Valid extensions are 'edf' and 'csv';
        - num_classes: total number of classes (individuals).
    
    Optional Parameters:
        - channels: list of channel codes that will be read. By default, this function reads all channels.
        The list containing all channel codes is: ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
        'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.',
        'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..',
        'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..',
        'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..',
        'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']
        - verbose: if set to 1, prints what type of data (training/validation or testing) is currently being
        loaded. Default value is 0.
    """

    # Processing x_train, y_train, x_val and y_val
    if(verbose):
        print('Training and Validation data are being loaded...')

    train_content = list()

    for train_task in train_tasks:
        if(verbose):
            print(f'* Using task {train_task}:')

        for i in range(1, num_classes + 1):
            if(verbose):
                print(f'  > Loading data from subject {i}.')

            # if(file_type == 'edf'):
                # train_content.append(read_EDF(folder_path+'S{:03d}/S{:03d}R{:02d}.edf'.format(i, i, train_task), channels))
            if(file_type == 'csv'):
                train_content.append(loadtxt(folder_path+'S{:03d}/S{:03d}R{:02d}.csv'.format(i, i, train_task), delimiter=','))
            else:
                print('ERROR: Invalid file_type parameter. Data will not be loaded.')

    # Processing x_test and y_test
    if(verbose):
        print('\nTesting data are being loaded...')

    test_content = list()

    for test_task in test_tasks:
        if(verbose):
            print(f'* Using task {test_task}:')

        for i in range(1, num_classes + 1):
            if(verbose):
                print(f'  > Loading data from subject {i}.')

            # if(file_type == 'edf'):
                # test_content.append(read_EDF(folder_path+'S{:03d}/S{:03d}R{:02d}.edf'.format(i, i, test_task), channels))
            if(file_type == 'csv'):
                test_content.append(loadtxt(folder_path+'S{:03d}/S{:03d}R{:02d}.csv'.format(i, i, test_task), delimiter=','))
            else:
                print('ERROR: Invalid file_type parameter. Data will not be loaded.')

    return train_content, test_content

def bandpass_filter(signal, lowcut, highcut, fs, filter_order, filter_type):
    """
    Band-pass filters a signal and returns it.

    Parameters:
        - signal: signal that will be band-pass filtered;
        - lowcut: lowcut of the filter;
        - highcut: highcut of the filter;
        - fs: frequency of the signal;
        - filter_order: order of the filter;
        - filter_type: how the signal will be filtered:
            * 'sosfilt': using the sosfilt() function from the scipy library;
            * 'filtfilt': using the firwin() and filtfilt() functions from the scipy library.
    """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if(filter_type == 'sosfilt'):
        sos = butter(filter_order, [low, high], btype='band', output='sos')
        y = sosfilt(sos, signal)
    elif(filter_type == 'filtfilt'):
        fir_coeff = firwin(filter_order+1,[low,high], pass_zero=False)
        y = filtfilt(fir_coeff, 1.0, signal)

    return y

def pre_processing(content, lowcut, highcut, frequency, filter_order, filter_type):
    """
    Pre-processess each channel of an EEG signal using band-pass filters.

    Parameters:
        - signal: signal that will be band-pass filtered;
        - lowcut: lowcut of the filter;
        - highcut: highcut of the filter;
        - fs: frequency of the signal;
        - filter_order: order of the filter;
        - filter_type: type of the filter used:
            * 'sosfilt': using the sosfilt() function from the scipy library.
            * 'filtfilt': using the firwin() and filtfilt() functions from the scipy library.
    """

    channels = content.shape[0]
    c = 0

    if(filter_type != 'sosfilt' and filter_type != 'filtfilt'):
        print('ERROR: Invalid filter_type parameter. Signal will not be filtered.')
        return content

    while c < channels:
        signal = content[c, :]
        content[c] = bandpass_filter(signal, lowcut, highcut, frequency, filter_order, filter_type)
        c += 1

    return content

def verbose_each_10_percent(count, data_amount, flag):
    """
    Auxiliar function for optional verbose on other functions. Returns the flag, possibly modified.

    Parameters:
        - count: current data index that was processed;
        - data_amount: length of the list of data;
        - flag: current state of the flag.
    """
    if count == data_amount and flag < 10:
        print('100%')
        flag = 10
    elif count >= data_amount * 0.9 and flag < 9:
        print('90%...',end='')
        flag = 9
    elif count >= data_amount * 0.8 and flag < 8:
        print('80%...',end='')
        flag = 8
    elif count >= data_amount * 0.7 and flag < 7:
        print('70%...',end='')
        flag = 7
    elif count >= data_amount * 0.6 and flag < 6:
        print('60%...',end='')
        flag = 6
    elif count >= data_amount * 0.5 and flag < 5:
        print('50%...',end='')
        flag = 5
    elif count >= data_amount * 0.4 and flag < 4:
        print('40%...',end='')
        flag = 4
    elif count >= data_amount * 0.3 and flag < 3:
        print('30%...',end='')
        flag = 3
    elif count >= data_amount * 0.2 and flag < 2:
        print('20%...',end='')
        flag = 2
    elif count >= data_amount * 0.1 and flag < 1:
        print('10%...',end='')
        flag = 1
    
    return flag

def filter_data(data, filter, sample_frequency, filter_order, filter_type, verbose = 0):
    """
    Takes a list of raw signals as input, applies a band-pass filter on each of them and outputs them as a list.

    The return of this function is in the format: filtered_data.

    Parameters:
        - data: list of signals that will be band-pass filtered;
        - filter: a list with length 2, where the first value is the lowcut of the band-pass filter used in
        pre-processing, and the second value is the highcut;
        - sample_frequency: frequency of the sampling;
        - filter_order: order of the filter;
        - filter_type: type of the filter used:
            * 'sosfilt': using the sosfilt() function from the scipy library.
            * 'filtfilt': using the firwin() and filtfilt() functions from the scipy library.
    
    Optional Parameters:
        - verbose: if set to 1, prints how many % of data is currently filtered (for each interval of 10%).
        Default value is 0.
    """

    filtered_data = list()

    if verbose == 1:
        count = 0
        flag = 0
        print('Data is being filtered: 0%...',end='')

    for signal in data:
        filtered_data.append(pre_processing(signal, filter[0], filter[1], sample_frequency, filter_order, filter_type))

        if verbose == 1:
            count += 1
            flag = verbose_each_10_percent(count, len(data), flag)
    
    return filtered_data

def normalize_signal(content, normalize_type):
    """
    Normalizes an EEG signal.

    Parameters:
        - content: the EEG signal that will be normalized.
        - normalize_type: type of normalization used:
            * 'each_channel': each channel of the EEG signal will be used to compute the mean, standard deviation,
            minimum and maximum values, which will be applied only to themselves in order to normalize them.
            * 'all_channels': all channels of the EEG signal will be used to compute the mean, standard deviation,
            minimum and maximum values, which will be applied to each signal in order to normalize them.
    """

    channels = content.shape[0]
    c = 0
    
    if(normalize_type == 'each_channel'):
        while c < channels:
            content[c] -= np.mean(content[c])
            content[c] += np.absolute(np.amin(content[c]))
            content[c] /= np.std(content[c])
            content[c] /= np.amax(content[c])

            c += 1
    elif(normalize_type == 'all_channels'):
        content -= np.mean(content)

        min_value = np.amin(content)
        while c < channels:
            content[c] += np.absolute(min_value)
            c += 1
        c = 0

        standard_deviation = np.std(content)
        while c < channels:
            content[c] /= standard_deviation
            c += 1
        c = 0

        max_value = np.amax(content)
        while c < channels:
            content[c] /= max_value
            c += 1
        c = 0
    elif(normalize_type == 'sun'):
        while c < channels:
            mean = np.mean(content[c])
            std = np.std(content[c])

            content[c] -= mean
            content[c] /= std

            c += 1
    else:
        print('ERROR: Invalid normalize_type parameter.')

    return content

def normalize_data(data, normalize_type, verbose = 0):
    """
    Takes a list of signals as input, normalizes and outputs them as a list.

    The return of this function is in the format: normalized_data.

    Parameters:
        - data: list of signals that will be normalized;
        - normalize_type: type of normalization used:
            * 'each_channel': each channel of the EEG signal will be used to compute the mean, standard deviation,
            minimum and maximum values, which will be applied only to themselves in order to normalize them.
            * 'all_channels': all channels of the EEG signal will be used to compute the mean, standard deviation,
            minimum and maximum values, which will be applied to each signal in order to normalize them.
    
    Optional Parameters:
        - verbose: if set to 1, prints how many % of data is currently filtered (for each interval of 10%).
        Default value is 0.
    """

    normalized_data = list()

    if verbose == 1:
        count = 0
        flag = 0
        print('Data is being normalized: 0%...',end='')

    for signal in data:
        normalized_data.append(normalize_signal(signal, normalize_type))

        if verbose == 1:
            count += 1
            flag = verbose_each_10_percent(count, len(data), flag)

    return normalized_data

def signal_cropping(x_data, y_data, content, window_size, offset, num_subject, num_classes, split_ratio=1.0, x_data_2=0, y_data_2=0, mode=None):
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
        - split_ratio: a number in the interval (0,1]. (split_ratio * 100)% of the processed signals will be
        stored in x_data and y_data, and [100 - (split_ratio * 100)]% will be stored in x_data_2 and y_data_2.
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
        print('ERROR: An offset equal to 0 would result in "infinite" equal windows.')
        return x_data, y_data
    # Checking the split_ratio parameter
    elif split_ratio <= 0 or split_ratio > 1:
        print('ERROR: The split_ratio parameter needs to be in the interval (0,1].')
        return x_data, y_data
    else:
        i = window_size
        while i <= content.shape[1] * split_ratio:
            if(mode != 'labels_only'):
                arr = content[: , (i-window_size):i]
                x_data.append(arr)

            arr2 = np.zeros((1,num_classes))
            arr2[0, num_subject] = 1
            y_data.append(arr2)

            i += offset

        if split_ratio == 1.0:
            return x_data, y_data
        
        while i <= content.shape[1]:
            if(mode != 'labels_only'):
                arr = content[: , (i-window_size):i]
                x_data_2.append(arr)

            arr2 = np.zeros((1,num_classes))
            arr2[0, num_subject] = 1
            y_data_2.append(arr2)

            i += offset

        return x_data, y_data, x_data_2, y_data_2

def crop_data(data, data_tasks, num_classes, window_size, offset, split_ratio=1.0, reshape='classic', mode=None, verbose=0):
    """
    Applies a sliding window cropping for data augmentation of the signals recieved as input and outputs them
    as numpy arrays.

    The default return of this function is in the format: x_data, y_data.

    Parameters:
        - data: list of signals that will be processed;
        - data_tasks: list containing the numbers of the experimental runs that were used to compose the data
        in load_data();
        - num_classes: total number of classes (individuals);
        - window_size: sliding window size;
        - offset: sliding window offset (deslocation);
    
    Optional Paramters:
        - split_ratio: if set to a value in the interval (0,1), then the data will be splited into 2 subsets and
        the return of the function will change its' format to: x_data, y_data, x_data_2, y_data_2. Default value
        is 1.0.
        - verbose: if set to 1, prints how many % of data is currently cropped (for each interval of 10%).
        Default value is 0.
    """

    x_dataL = list()
    x_dataL_2 = list()
    y_dataL = list()
    y_dataL_2 = list()

    if verbose == 1:
        count = 0
        flag = 0
        data_amount = len(data_tasks) * num_classes
        print('Data is being cropped: 0%...',end='')

    # Checking the split_ratio parameter
    if split_ratio <= 0 or split_ratio > 1:
        print('ERROR: The split_ratio parameter needs to be in the interval (0,1].')
        return None
    elif split_ratio == 1:
        for task in range(0, len(data_tasks)):
            for i in range(1, num_classes + 1):
                x_dataL, y_dataL = signal_cropping(x_dataL, y_dataL, data[ (task * num_classes) + i - 1],
                                                   window_size, offset, i, num_classes, mode=mode)

                if verbose == 1:
                    count += 1
                    flag = verbose_each_10_percent(count, data_amount, flag)

        if verbose == 1:
            print('Data is being transformed to an numpy array and being reshaped.')

        x_data = np.asarray(x_dataL, dtype = object).astype('float32')
        y_data = np.asarray(y_dataL, dtype = object).astype('float32')

        # The initial format of a "x_data" (EEG signal) is "a x num_channels x window_size", but the 
        # input shape of the CNN is "a x window_size x num_channels".
        if reshape == 'classic':
            x_data = x_data.reshape(x_data.shape[0], x_data.shape[2], x_data.shape[1])
        elif reshape == 'data_generator':
            temp = np.empty((x_data.shape[0], x_data.shape[2], x_data.shape[1]))

            i = 0
            while(i < x_data.shape[0]):
                temp[i] = x_data[i].reshape(x_data.shape[2], x_data.shape[1])
                i += 1

            x_data = temp
        elif reshape != 'no_reshape':
            print('ERROR: Invalid reshape parameter.')

        # The initial format of a "y_data" (label) is "a x 1 x num_classes", but the correct format
        # is "a x num_classes".
        if reshape == 'classic':
            y_data = y_data.reshape(y_data.shape[0], y_data.shape[2])

        return x_data, y_data
    else:
        for task in range(0, len(data_tasks)):
            for i in range(1, num_classes + 1):
                x_dataL, y_dataL, x_dataL_2, y_dataL_2 = signal_cropping(x_dataL, y_dataL, data[ (task * num_classes) + i - 1],
                                                                         window_size, offset, i, num_classes,
                                                                         split_ratio, x_dataL_2, y_dataL_2, mode)
                
                if verbose == 1:
                    count += 1
                    flag = verbose_each_10_percent(count, data_amount, flag)

        if verbose == 1:
            print('Data is being transformed to an numpy array and being reshaped.')

        x_data = np.asarray(x_dataL, dtype = object).astype('float32')
        x_data_2 = np.asarray(x_dataL_2, dtype = object).astype('float32')
        y_data = np.asarray(y_dataL, dtype = object).astype('float32')
        y_data_2 = np.asarray(y_dataL_2, dtype = object).astype('float32')

        # The initial format of a "x_data" (EEG signal) is "a x num_channels x window_size", but the 
        # input shape of the CNN is "a x window_size x num_channels".
        if reshape == 'classic':
            x_data = x_data.reshape(x_data.shape[0], x_data.shape[2], x_data.shape[1])
            x_data_2 = x_data_2.reshape(x_data_2.shape[0], x_data_2.shape[2], x_data_2.shape[1])
        elif reshape == 'data_generator':
            temp = np.empty((x_data.shape[0], x_data.shape[2], x_data.shape[1]))
            i = 0
            while(i < x_data.shape[0]):
                temp[i] = x_data[i].reshape(x_data.shape[2], x_data.shape[1])
                i += 1
            x_data = temp
            
            temp = np.empty((x_data_2.shape[0], x_data_2.shape[2], x_data_2.shape[1]))
            i = 0
            while(i < x_data_2.shape[0]):
                temp[i] = x_data_2[i].reshape(x_data_2.shape[2], x_data_2.shape[1])
                i += 1
            x_data_2 = temp
        elif reshape != 'no_reshape':
            print('ERROR: Invalid reshape parameter.')

        # The initial format of a "y_data" (label) is "a x 1 x num_classes", but the correct format
        # is "a x num_classes".
        if reshape == 'classic':
            y_data = y_data.reshape(y_data.shape[0], y_data.shape[2])
            y_data_2 = y_data_2.reshape(y_data_2.shape[0], y_data_2.shape[2])

        return x_data, y_data, x_data_2, y_data_2

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
    print(f'len(label1) = {len(label1)}')
    print(f'len(feature1) = {len(feature1)}')

    lesser1 = len(label1)
    if(len(feature1) < len(label1)):
        lesser1 = len(feature1)
    
    lesser2 = len(label2)
    if(len(feature2) < len(label2)):
        lesser2 = len(feature2)

    same_list = []
    dif_list = []
    for row in range(lesser1):
        for col in range(row+1, lesser2):
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

def n_samples_with_sliding_window(full_signal_size, window_size, offset):
    """
    Returns the number of samples in a signal, generated after applying a sliding window.

    Parameters:
        - full_signal_size: full size of the signal;
        - window_size: size of the sliding window;
        - offset: amount of samples the window will slide in each iteration.
    """
    n_samples = 1
    i = window_size

    if(offset == 0):
        print('ERROR: An offset equal to 0 would result in "infinite" equal windows.')
        return 0

    while(i < full_signal_size):
        n_samples += 1
        i += offset

    return n_samples

# def stack_arrays(array_A, array_B):
#     """
#     Auxiliar function for the __data_generation() function on the DataGenerator Class. Stacks two arrays
#     vertically: array_A on top of array_B. array_type is either 'data' or 'labels'.
#     """
#     n_dim = array_A.ndim

#     if(n_dim == 2):
#         array_C = np.empty((array_A.shape[0] + array_B.shape[0], array_A.shape[1]))
#     else:
#         array_C = np.empty((array_A.shape[0] + array_B.shape[0], array_A.shape[1], array_A.shape[2]))

#     for i in range(0, array_A.shape[0]):
#         array_C[i] = array_A[i]
    
#     for i in range(0, array_B.shape[0]):
#         array_C[i + array_A.shape[0]] = array_B[i]

#     return array_C

class DataGenerator(keras.utils.Sequence):
    """
    Generates data for the model on the fly.
    """
    def __init__(self, data_generator_type, batch_size, dim, offset, full_signal_size, n_channels,
                n_classes, tasks, dataset_type, split_ratio, list_IDs=None, shuffle=False):
        """
        Initialization function of the class.
        
        Parameters:
            - data_generator_type: what kind of input does the data generator expects, and what it will do before
            feeding the data into the model;
                * 'process_data': the data generator will load, filter, normalize and crop the data, using a
                sliding window in the last method.
                * 'crop_only': the data generator will take preprocessed data, stored in csv files, and crop
                them using a sliding window.
            - batch_size: while training, the processed data will be split into groups of shape
            (batch_size, dim, n_channels), which will be fed into the model;
            - dim: size of the sliding window;
            - offset: amount of samples the window will slide in each iteration;
            - full_signal_size: full size of the signals being processed;
            - n_channels: number of channels in each signal being processed;
            - n_classes: total number of classes (individuals);
            - tasks: list that contains the numbers of the experimental runs that will be used;
            - dataset_type: which type of dataset will be created by the data generator. Valid types are 'train',
            'validation' and 'test';
            - split_ratio: a number in the interval (0,1]. (split_ratio * 100)% of the processed signals will be
            stored separetly from the other [100 - (split_ratio * 100)]%.
        
        Optional Parameters:
            - list_IDs: used if the data_generator_type is 'crop_only'. A list of csv file names, in which the
            preprocessed data are stored. Default value is None;
            - shuffle: if the data being fed into the model will be shuffled or not at each epoch. Default value is
            False.
        """
        self.data_generator_type = data_generator_type
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.dim = dim
        self.offset = offset
        self.full_signal_size = full_signal_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.tasks = tasks
        self.dataset_type = dataset_type
        self.split_ratio = split_ratio
        self.shuffle = shuffle

        # self.lag_counter = 0

        aux = math.floor(n_samples_with_sliding_window(self.full_signal_size, self.dim, self.offset))
        aux_2 = math.floor(n_samples_with_sliding_window(self.full_signal_size * self.split_ratio, self.dim, self.offset))

        if(self.dataset_type == 'train'):
            self.samples_per_file = aux_2
        elif(self.dataset_type == 'validation'):
            self.samples_per_file = aux - aux_2
        elif(self.dataset_type == 'test'):
            self.samples_per_file = aux

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        n_samples = self.samples_per_file * len(self.tasks) * self.n_classes
        aux = math.floor(n_samples / self.batch_size) * self.batch_size

        return math.floor((aux / self.batch_size) - 1)
        # return math.ceil(n_samples / self.batch_size)

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """

        # First three batches generated are ignored by model.fit
        # if(self.lag_counter < 3):
        #     x = np.zeros((self.batch_size, self.dim, self.n_channels))
        #     y = np.zeros((self.batch_size, self.n_classes))
        #     self.lag_counter += 1
        #     return (x, y)

        # Generate indexes of the batch
        indexes = self.indexes[self.first_index:self.first_index + self.batch_size]

        # print(f'__getitem__ : index = {index}')
        # print(f'__getitem__ : self.batch_size = {self.batch_size}')
        # print(f'__getitem__ : self.indexes = {self.indexes}')
        print(f'\nindexes da vez = {indexes}')

        # excess já tem uma batch pronta ?
        if(self.excess_x is not None):
            if(self.excess_x.shape[0] >= self.batch_size):
                # print(f'\npor excesso: excess_x era {self.excess_x.shape} - ', end='') #####

                x = np.empty((self.batch_size, self.dim, self.n_channels))
                y = np.empty((self.batch_size, self.n_classes))

                aux_x = None
                aux_y = None

                if(self.excess_x.shape[0] != self.batch_size):
                    aux_x = np.empty((self.excess_x.shape[0] - self.batch_size, self.dim, self.n_channels))
                    aux_y = np.empty((self.excess_x.shape[0] - self.batch_size, self.n_classes))

                for i in range(0, self.excess_x.shape[0]):
                    # "transportando" a batch pronta que tá no excess
                    if(i < self.batch_size):
                        x[i] = self.excess_x[i]
                        y[i] = self.excess_y[i]

                    # restante do excess armazenado em aux
                    else:
                        aux_x[i - self.batch_size] = self.excess_x[i]
                        aux_y[i - self.batch_size] = self.excess_y[i]
                
                self.excess_x = aux_x
                self.excess_y = aux_y
                self.first_index += self.batch_size # + 1

                ###################################
                # if(self.excess_x is not None):
                #     print(f'excess_x ficou {self.excess_x.shape} - x tem tamanho {x.shape}\n') 
                # else:
                #     print(f'sem excesso - x tem tamanho {x.shape}\n')
                ###################################

                return (x, y)

        # quais arquivos é pra ler nessa batch?

        n_files = int(math.ceil(self.batch_size / self.samples_per_file))

        first_file = 0
        i = 0
        while(i < self.first_index):
            i += self.samples_per_file
            first_file += 1

        list_temp = []
        for i in range(0, n_files):
            # k = indexes[first_file]
            list_temp.append(self.list_IDs[first_file])
            first_file += 1
    
        if self.shuffle == True:
            np.random.shuffle(list_temp)

        # print(f'por arquivo: arquivo eh {list_temp} - ', end='') #####

        (x, y) = self.__data_generation(list_temp)

        #########################################################
        # print(f'x tem tamanho {x.shape} - ', end='') #
        # if(self.excess_x is not None):
        #     print(f'excess_x tem tamanho {self.excess_x.shape}')
        # else:
        #     print('sem excesso.\n')
        #########################################################

        return (x, y)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """

        self.excess_x = None         # list that will store data that exceeds batch_size
        self.excess_y = None         # list that will store labels that exceeds batch_size
        self.first_index = 0         # first index avaliable

        n_samples = self.samples_per_file * len(self.tasks) * self.n_classes

        aux = math.floor(n_samples / self.batch_size)

        # self.indexes = np.arange((aux * self.batch_size) - self.batch_size)
        self.indexes = np.arange(aux * self.batch_size)

        print(f'\nself.samples_per_file = {self.samples_per_file}')
        print(f'len(self.tasks) = {len(self.tasks)}')
        print(f'self.n_classes = {self.n_classes}')
        print(f'n_samples = {n_samples}')
        print(f'aux = {aux}')
        print(f'self.indexes = {self.indexes}')

        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples.
        """

        # Initialization
        temp_x = []
        subjects = []

        # Loading data from .csv files ('crop_only' DataGenerator)
        for i, ID in enumerate(list_IDs_temp):
            if(self.dataset_type == 'train' or self.dataset_type == 'validation'):
                file_x = np.loadtxt('processed_train_data/' + list_IDs_temp[i], delimiter=';', usecols=range(self.n_channels))
                string = 'processed_train_data/' + list_IDs_temp[i]

            elif(self.dataset_type == 'test'):
                file_x = np.loadtxt('processed_test_data/' + list_IDs_temp[i], delimiter=';', usecols=range(self.n_channels))
                string = 'processed_test_data/' + list_IDs_temp[i]

            file_x = np.asarray(file_x, dtype = object).astype('float32')
            file_x = file_x.T
            temp_x.append(file_x)

            string = string.split("_subject_")[1]      # 'X.csv'
            subject = int(string.split(".csv")[0])     # X
            subjects.append(subject)

        # Cropping Data
        x_dataL = list()
        x_dataL_2 = list()
        y_dataL = list()
        y_dataL_2 = list()

        if(self.dataset_type == 'train' or self.dataset_type == 'validation'):
            pos = 0
            for data in temp_x:
                x_dataL, y_dataL, x_dataL_2, y_dataL_2 = signal_cropping(x_dataL, y_dataL, data, self.dim,
                                                        self.offset, subjects[pos], self.n_classes,
                                                        self.split_ratio, x_dataL_2, y_dataL_2)
                pos += 1

        elif(self.dataset_type == 'test'):
            pos = 0
            for data in temp_x:
                x_dataL, y_dataL = signal_cropping(x_dataL, y_dataL, data, self.dim, self.offset, subjects[pos],
                                    self.n_classes)
                pos += 1
        
        x = None
        y = None

        if(self.dataset_type == 'train' or self.dataset_type == 'test'):
            x_data = np.asarray(x_dataL, dtype = object).astype('float32')
            y_data = np.asarray(y_dataL, dtype = object).astype('float32')

            if(x_data.shape[0] != 0):
                x_data = x_data.reshape(x_data.shape[0], x_data.shape[2], x_data.shape[1])
                y_data = y_data.reshape(y_data.shape[0], y_data.shape[2])

            x = x_data
            y = y_data

        elif(self.dataset_type == 'validation'):
            x_data_2 = np.asarray(x_dataL_2, dtype = object).astype('float32')
            y_data_2 = np.asarray(y_dataL_2, dtype = object).astype('float32')

            if(x_data_2.shape[0] != 0):
                x_data_2 = x_data_2.reshape(x_data_2.shape[0], x_data_2.shape[2], x_data_2.shape[1])
                y_data_2 = y_data_2.reshape(y_data_2.shape[0], y_data_2.shape[2])

            x = x_data_2
            y = y_data_2
        
        # print(f'samples produzidas no cropping {x.shape} - ', end='') #####

        # Is there any excess from the previous batch? If so, merge it first
        if(self.excess_x is not None):
            x = np.vstack((self.excess_x, x))
            y = np.vstack((self.excess_y, y))

        # Only (batch_size, dim, n_channels) data and (batch_size, num_classes) labels are returned
        if(x.shape[0] > self.batch_size):
            self.excess_x = x[self.batch_size:]
            self.excess_y = y[self.batch_size:]

        x = x[:self.batch_size]
        y = y[:self.batch_size]

        # Updating first index avaliable
        self.first_index += self.batch_size

        return (x, y)
