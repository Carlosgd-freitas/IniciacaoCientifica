import os
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, firwin, filtfilt
from sklearn.metrics.pairwise import euclidean_distances

def read_EDF(path, channels = None):
    """
    Reads data from an EDF file and returns it in a numpy array format.

    Parameters:
        - path: path of the file that will be read.
    
    Optional Parameters:
        - channels: list of channel codes that will be read. By default, this function reads all channels.
        The list containing all channel codes is: ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
        'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.',
        'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..',
        'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..',
        'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..',
        'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']
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

def load_data(folder_path, train_tasks, test_tasks, num_classes, channels = None, verbose = 0):
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

            train_content.append(read_EDF(folder_path+'S{:03d}/S{:03d}R{:02d}.edf'.format(i, i, train_task), channels))

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

            test_content.append(read_EDF(folder_path+'S{:03d}/S{:03d}R{:02d}.edf'.format(i, i, test_task), channels))

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
        print('Data is being filtered: 0%...',end='')

    for signal in data:
        filtered_data.append(pre_processing(signal, filter[0], filter[1], sample_frequency, filter_order, filter_type))

        if verbose == 1:
            count += 1
            if count == len(data):
                print('100%')
            elif count >= len(data) * 0.9:
                print('90%...',end='')
            elif count >= len(data) * 0.8:
                print('80%...',end='')
            elif count >= len(data) * 0.7:
                print('70%...',end='')
            elif count >= len(data) * 0.6:
                print('60%...',end='')
            elif count >= len(data) * 0.5:
                print('50%...',end='')
            elif count >= len(data) * 0.4:
                print('40%...',end='')
            elif count >= len(data) * 0.3:
                print('30%...',end='')
            elif count >= len(data) * 0.2:
                print('20%...',end='')
            elif count >= len(data) * 0.1:
                print('10%...',end='')
    
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
        print('Data is being normalized: 0%...',end='')

    for signal in data:
        normalized_data.append(normalize_signal(signal, normalize_type))

        if verbose == 1:
            count += 1
            if count == len(data):
                print('100%')
            elif count >= len(data) * 0.9:
                print('90%...',end='')
            elif count >= len(data) * 0.8:
                print('80%...',end='')
            elif count >= len(data) * 0.7:
                print('70%...',end='')
            elif count >= len(data) * 0.6:
                print('60%...',end='')
            elif count >= len(data) * 0.5:
                print('50%...',end='')
            elif count >= len(data) * 0.4:
                print('40%...',end='')
            elif count >= len(data) * 0.3:
                print('30%...',end='')
            elif count >= len(data) * 0.2:
                print('20%...',end='')
            elif count >= len(data) * 0.1:
                print('10%...',end='')

    return normalized_data

def signal_cropping(x_data, y_data, content, window_size, offset, num_subject, num_classes, split_ratio=1.0, x_data_2=0, y_data_2=0):
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
        print('ERROR: An offset equals to 0 would result in "infinite" equal windows.')
        return x_data, y_data
    # Checking the split_ratio parameter
    elif split_ratio <= 0 or split_ratio > 1:
        print('ERROR: The split_ratio parameter needs to be in the interval (0,1].')
        return x_data, y_data
    else:
        i = window_size
        while i <= content.shape[1] * split_ratio:
            arr = content[: , (i-window_size):i]
            x_data.append(arr)

            arr2 = np.zeros((1,num_classes))
            arr2[0, num_subject] = 1
            y_data.append(arr2)

            i += offset

        if split_ratio == 1.0:
            return x_data, y_data
        
        while i <= content.shape[1]:
            arr = content[: , (i-window_size):i]
            x_data_2.append(arr)

            arr2 = np.zeros((1,num_classes))
            arr2[0, num_subject] = 1
            y_data_2.append(arr2)

            i += offset

        return x_data, y_data, x_data_2, y_data_2

def crop_data(data, data_tasks, num_classes, window_size, offset, split_ratio=1.0, verbose=0):
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
        data_amount = len(data_tasks) * num_classes
        print('Data is being cropped: 0%...',end='')

    # Checking the split_ratio parameter
    if split_ratio <= 0 or split_ratio > 1:
        print('ERROR: The split_ratio parameter needs to be in the interval (0,1].')
        return None
    elif split_ratio == 1:
        for task in data_tasks:
            for i in range(1, num_classes + 1):
                x_dataL, y_dataL = signal_cropping(x_dataL, y_dataL, data[i-1],
                                                   window_size, offset, i, num_classes)

                if verbose == 1:
                    count += 1
                    if count == data_amount:
                        print('100%')
                    elif count >= data_amount * 0.9:
                        print('90%...',end='')
                    elif count >= data_amount * 0.8:
                        print('80%...',end='')
                    elif count >= data_amount * 0.7:
                        print('70%...',end='')
                    elif count >= data_amount * 0.6:
                        print('60%...',end='')
                    elif count >= data_amount * 0.5:
                        print('50%...',end='')
                    elif count >= data_amount * 0.4:
                        print('40%...',end='')
                    elif count >= data_amount * 0.3:
                        print('30%...',end='')
                    elif count >= data_amount * 0.2:
                        print('20%...',end='')
                    elif count >= data_amount * 0.1:
                        print('10%...',end='')

        if verbose == 1:
            print('Data is being transformed to an numpy array and being reshaped.')

        x_data = np.asarray(x_dataL, dtype = object).astype('float32')
        y_data = np.asarray(y_dataL, dtype = object).astype('float32')

        # The initial format of a "x_data" (EEG signal) is "a x num_channels x window_size", but the 
        # input shape of the CNN is "a x window_size x num_channels".
        x_data = x_data.reshape(x_data.shape[0], x_data.shape[2], x_data.shape[1])

        # The initial format of a "y_data" (label) is "a x 1 x num_classes", but the correct format
        # is "a x num_classes".
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[2])

        return x_data, y_data
    else:
        for task in data_tasks:
            for i in range(1, num_classes + 1):
                x_dataL, y_dataL, x_dataL_2, y_dataL_2 = signal_cropping(x_dataL, y_dataL, data[i-1],
                                                                         window_size, offset, i, num_classes,
                                                                         split_ratio, x_dataL_2, y_dataL_2)
                
                if verbose == 1:
                    count += 1
                    if count == data_amount:
                        print('100%')
                    elif count >= data_amount * 0.9:
                        print('90%...',end='')
                    elif count >= data_amount * 0.8:
                        print('80%...',end='')
                    elif count >= data_amount * 0.7:
                        print('70%...',end='')
                    elif count >= data_amount * 0.6:
                        print('60%...',end='')
                    elif count >= data_amount * 0.5:
                        print('50%...',end='')
                    elif count >= data_amount * 0.4:
                        print('40%...',end='')
                    elif count >= data_amount * 0.3:
                        print('30%...',end='')
                    elif count >= data_amount * 0.2:
                        print('20%...',end='')
                    elif count >= data_amount * 0.1:
                        print('10%...',end='')

        if verbose == 1:
            print('Data is being transformed to an numpy array and being reshaped.')

        x_data = np.asarray(x_dataL, dtype = object).astype('float32')
        x_data_2 = np.asarray(x_dataL_2, dtype = object).astype('float32')
        y_data = np.asarray(y_dataL, dtype = object).astype('float32')
        y_data_2 = np.asarray(y_dataL_2, dtype = object).astype('float32')

        # The initial format of a "x_data" (EEG signal) is "a x num_channels x window_size", but the 
        # input shape of the CNN is "a x window_size x num_channels".
        x_data = x_data.reshape(x_data.shape[0], x_data.shape[2], x_data.shape[1])
        x_data_2 = x_data_2.reshape(x_data_2.shape[0], x_data_2.shape[2], x_data_2.shape[1])

        # The initial format of a "y_data" (label) is "a x 1 x num_classes", but the correct format
        # is "a x num_classes".
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[2])
        y_data_2 = y_data_2.reshape(y_data_2.shape[0], y_data_2.shape[2])

        return x_data, y_data, x_data_2, y_data_2

# def load_data(folder_path, train_tasks, test_tasks, num_classes, filter, sample_frequency, window_size, offset, train_val_ratio, verbose=0):
#     """
#     Returns the processed signals and labels for training (x_train and y_train), validation (x_val and y_val) and
#     testing (x_test and y_test).

#     The return of this function is in the format: x_train, x_val, x_test, y_train, y_val, y_test.

#     Parameters:
#         - folder_path: path of the folder in which the the EDF files are stored.
#         E.g. if this python script is in the same folder as the sub-folder used to store the EDF files, and this
#         sub-folder is called "Dataset", then this parameter should be: './Dataset/';
#         - train_tasks: list that contains the numbers of the experimental runs that will be used to create train
#         and validation data;
#         - test_tasks: list that contains the numbers of the experimental runs that will be used to create testing
#         data.
#         - num_classes: total number of classes (individuals);
#         - filter: a list with size 2, where the first value is the lowcut of the band-pass filter used in
#         pre-processing, and the second value is the highcut;
#         - sample_frequency: frequency of the sampling;
#         - window_size: sliding window size;
#         - offset: sliding window offset (deslocation);
#         - train_val_ratio: ratio for composing training and validation data.
    
#     Optional Parameters:
#         - verbose: if set to 1, prints what type of data (training/validation or testing) is currently being
#         processed. Default value is 0.
#     """

#     # Processing x_train, y_train, x_val and y_val
#     if(verbose):
#         print('Training and Validation data is being processed...')

#     x_trainL = list()
#     x_valL = list()
#     y_trainL = list()
#     y_valL = list()

#     for train_task in train_tasks:
#         if(verbose):
#             print(f'* Using task {train_task}:')

#         for i in range(1, num_classes + 1):
#             if(verbose):
#                 print(f'  > Loading data from subject {i}.')

#             train_content = read_EDF(folder_path+'S{:03d}/S{:03d}R{:02d}.edf'.format(i, i, train_task), ['C1..', 'Cz..', 'C2..', 'Af3.', 'Afz.', 'Af4.', 'O1..', 'Oz..', 'O2..'])
#             train_content = pre_processing(train_content, filter[0], filter[1], sample_frequency, 12, 'sosfilt')
#             train_content = normalize_signal(train_content, 'all_channels')
#             x_trainL, y_trainL, x_valL, y_valL = signal_cropping(x_trainL, y_trainL, train_content, window_size, offset, i, num_classes, train_val_ratio, x_valL, y_valL)
    
#     x_train = np.asarray(x_trainL, dtype = object).astype('float32')
#     x_val = np.asarray(x_valL, dtype = object).astype('float32')
#     y_train = np.asarray(y_trainL, dtype = object).astype('float32')
#     y_val = np.asarray(y_valL, dtype = object).astype('float32')

#     # Processing x_test and y_test
#     if(verbose):
#         print('\nTesting data is being processed...')

#     x_testL = list()
#     y_testL = list()

#     for test_task in test_tasks:
#         if(verbose):
#             print(f'* Using task {test_task}:')

#         for i in range(1, num_classes + 1):
#             if(verbose):
#                 print(f'  > Loading data from subject {i}.')

#             test_content = read_EDF(folder_path+'S{:03d}/S{:03d}R{:02d}.edf'.format(i, i, test_task), ['C1..', 'Cz..', 'C2..', 'Af3.', 'Afz.', 'Af4.', 'O1..', 'Oz..', 'O2..'])
#             test_content = pre_processing(test_content, filter[0], filter[1], sample_frequency, 12, 'sosfilt')
#             test_content = normalize_signal(test_content, 'all_channels')
#             x_testL, y_testL = signal_cropping(x_testL, y_testL, test_content, window_size, window_size, i, num_classes)

#     x_test = np.asarray(x_testL, dtype = object).astype('float32')
#     y_test = np.asarray(y_testL, dtype = object).astype('float32')

#     # The initial format of a "x_data" (EEG signal) is "a x num_channels x window_size", but the 
#     # input shape of the CNN is "a x window_size x num_channels".
#     x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1])
#     x_val = x_val.reshape(x_val.shape[0], x_val.shape[2], x_val.shape[1])
#     x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[1])

#     # The initial format of a "y_data" (label) is "a x 1 x num_classes", but the correct format
#     # is "a x num_classes".
#     y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])
#     y_val = y_val.reshape(y_val.shape[0], y_val.shape[2])
#     y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])

#     return x_train, x_val, x_test, y_train, y_val, y_test

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