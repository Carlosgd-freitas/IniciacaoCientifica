import utils

import math
import random
import numpy as np
import tensorflow.keras as keras

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
            if(mode != 'second_data_only'):
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
            if(mode != 'first_data_only'):
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
                    flag = utils.verbose_each_10_percent(count, data_amount, flag)

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
                    flag = utils.verbose_each_10_percent(count, data_amount, flag)

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

def get_crop_positions(dataset_type, num_signals, signal_size, window_size, offset, split_ratio):
    """
    Stores and returns the information of all cropping that will be done in the EEG signals.

    Parameters:
        - dataset_type: which type of dataset will be created by the data generator. Valid types are 'train',
        'validation' and 'test';
        - num_signals: number of signals being processed;
        - signal_size: full size of the signals being processed;
        - window_size: size of the sliding window;
        - offset: amount of samples the window will slide in each iteration;
        - split_ratio: a number in the interval (0,1]. (split_ratio * 100)% of the processed signals will be
        stored separetly from the other [100 - (split_ratio * 100)]%.
    """
    crop_positions = []

    if(dataset_type == 'train' or dataset_type == 'test'):
        first_i = window_size
        stop = signal_size * split_ratio
    
    elif(dataset_type == 'validation'):
        first_i = math.floor(signal_size * split_ratio) + offset
        stop = signal_size

    signal_index = 0
    while(signal_index < num_signals):
        i = first_i

        while(i <= stop):
            # Each crop position is a tuple: (file in which the crop will take place, end of the cropping)
            one_crop_position = (signal_index, i)
            crop_positions.append(one_crop_position)

            i += offset
        
        signal_index += 1

    return crop_positions

class DataGenerator(keras.utils.Sequence):
    """
    Generates data for the model on the fly, using a sliding window for data augmentation.
    """
    def __init__(self, list_files, batch_size, dim, offset, full_signal_size, n_channels,
                n_classes, tasks, dataset_type, split_ratio, processed_data_path, shuffle=False):
        """
        Initialization function of the class.
        
        Parameters:
            - list_files: a list of csv file names, in which the preprocessed data are stored;
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
            stored separetly from the other [100 - (split_ratio * 100)]%;
            - processed_data_path: .
        
        Optional Parameters:
            - shuffle: if the data being fed into the model will be shuffled or not at each epoch. Default value is
            False.
        """
        # Intializing variables
        self.list_files = list_files
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

        # Calculating the number of samples per file
        if(self.dataset_type == 'train'):
            self.samples_per_file = utils.n_samples_with_sliding_window(0, self.full_signal_size * self.split_ratio, self.dim, self.offset)
        elif(self.dataset_type == 'validation'):
            self.samples_per_file = utils.n_samples_with_sliding_window(self.full_signal_size * self.split_ratio, self.full_signal_size, self.offset, self.offset) - 1
        elif(self.dataset_type == 'test'):
            self.samples_per_file = utils.n_samples_with_sliding_window(0, self.full_signal_size, self.dim, self.offset)

        # Loading all files from list_files
        data = []
        subjects = []

        for task in self.tasks:
            i = 0

            while(i < self.n_classes):
                file_x = np.loadtxt(processed_data_path + 'processed_data/task' + str(task) + '/' + list_files[i], delimiter=';', usecols=range(self.n_channels))
                string = processed_data_path + 'processed_data/task' + str(task) + '/' + list_files[i]

                file_x = np.asarray(file_x, dtype = object).astype('float32')
                data.append(file_x)

                string = string.split("_subject_")[1]      # 'X.csv'
                subject = int(string.split(".csv")[0])     # X
                subjects.append(subject)

                i += 1
                
        data = np.asarray(data, dtype = object).astype('float32')
        
        # Storing the information of all cropping that will be done in the EEG signals
        crop_positions = get_crop_positions(self.dataset_type, data.shape[0], data[0].shape[0], self.dim,
            self.offset, self.split_ratio)

        # if(self.dataset_type == 'train' or self.dataset_type == 'test'):
        #     signal_index = 0
        #     while(signal_index < data.shape[0]):
        #         i = self.dim

        #         while(i <= data[0].shape[0] * self.split_ratio):
        #             # Each crop position is a tuple: (file in which the crop will take place, end of the cropping)
        #             one_crop_position = (signal_index, i)
        #             crop_positions.append(one_crop_position)

        #             i += self.offset
                
        #         signal_index += 1

        # elif(self.dataset_type == 'validation'):
        #     signal_index = 0
        #     while(signal_index < data.shape[0]):
        #         i = math.floor(data[0].shape[0] * self.split_ratio) + self.offset

        #         while(i <= data[0].shape[0]):
        #             one_crop_position = (signal_index, i)
        #             crop_positions.append(one_crop_position)

        #             i += self.offset

        #         signal_index += 1
        
        # print(f'self.dataset_type = {self.dataset_type}, len(crop_positions) = {len(crop_positions)}')
        # print(f'crop_positions = {crop_positions}')

        self.data = data
        self.subjects = subjects
        self.crop_positions = crop_positions

        ###
        if(self.dataset_type == 'validation'):
            print(f'self.crop_positions = {self.crop_positions}')

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """

        return math.ceil(len(self.crop_positions) / self.batch_size)

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        index = self.next_index

        x = None
        y = None

        # If the batch being generated does have batch_size samples
        # if(self.last_sample_used + self.batch_size < len(self.crop_positions)):
        #     x = np.empty((self.batch_size, self.dim, self.n_channels))
        #     y = np.empty((self.batch_size, self.n_classes))

        #     crop_positions = self.crop_positions[index*self.batch_size : (index+1)*self.batch_size]

        #     for i in range(0, 100):
        #         file_index = crop_positions[i][0]
        #         crop_end = crop_positions[i][1]

        #         x[i] = self.data[file_index][(crop_end-self.dim):crop_end]

        #         subject = self.subjects[file_index]

        #         label = np.zeros((1, self.n_classes))
        #         label[0, subject-1] = 1

        #         y[i] = label
            
        #     if(self.last_sample_used != 0):
        #         self.last_sample_used += self.batch_size
        #     else:
        #         self.last_sample_used += self.batch_size - 1
            
        #     # Storing the N first batches for later use, if needed
        #     if(index < self.cache_size):
        #         self.cache_x[index] = x[i]
        #         self.cache_y[index] = label

        # If the batch being generated is using samples from crop_positions
        if(self.last_sample_used < len(self.crop_positions) - 1):
            x = []
            y = []

            crop_positions = self.crop_positions[index*self.batch_size : (index+1)*self.batch_size]

            count = 0
            for crop_position in crop_positions:
                file_index, crop_end = crop_position
                sample = self.data[file_index][(crop_end-self.dim):crop_end]

                x.append(sample)

                subject = self.subjects[file_index]

                label = np.zeros((1, self.n_classes))
                label[0, subject-1] = 1

                y.append(label)

                count += 1

            # count = 0
            # for i in range(0, self.batch_size):

            #     if(self.last_sample_used + count == len(self.crop_positions) - 1):
            #         break

            #     file_index = crop_positions[i][0]
            #     crop_end = crop_positions[i][1]
            #     sample = self.data[file_index][(crop_end-self.dim):crop_end]

            #     x.append(sample)

            #     subject = self.subjects[file_index]

            #     label = np.zeros((1, self.n_classes))
            #     label[0, subject-1] = 1

            #     y.append(label)

            #     count += 1
            
            x = np.asarray(x, dtype = object).astype('float32')
            y = np.asarray(y, dtype = object).astype('float32')

            # The initial format of "y" (label) is "a x 1 x num_classes", but the correct format
            # is "a x num_classes".
            y = y.reshape(y.shape[0], y.shape[2])

            if(self.last_sample_used != 0):
                self.last_sample_used += count
            else:
                self.last_sample_used += count - 1

            # print(f'x.shape = {x.shape}')
            # print(f'y.shape = {y.shape}')

            # Storing the N first batches for later use, if needed
            if(index < self.cache_size):
                self.cache_x[index] = x
                self.cache_y[index] = label
        
        # Taking data from cache
        else:

            if(self.cache_next_index == self.cache_size):
                self.cache_next_index = 0

            x = self.cache_x[self.cache_next_index]
            y = self.cache_y[self.cache_next_index]

            self.cache_next_index += 1

        self.next_index += 1

        return (x, y)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """
        if self.shuffle == True:
            random.shuffle(self.crop_positions)

        self.next_index = 0       # Index of the next batch
        self.last_sample_used = 0 # Index of the last sample used

        # Cache: stores the N first batches, that can be used later if the asynchronous behavior of Data Generators
        # make no use of them at first
        self.cache_size = 10
        self.cache_next_index = 0
        self.cache_x = np.empty((self.cache_size, self.batch_size, self.dim, self.n_channels))
        self.cache_y = np.empty((self.cache_size, self.batch_size, self.n_classes))
    
    def return_all_data(self):
        """
        Returns all data at once.
        """
        x = []
        y = []

        for crop_position in self.crop_positions:
            file_index, crop_end = crop_position
            sample = self.data[file_index][(crop_end-self.dim):crop_end]

            x.append(sample)

            subject = self.subjects[file_index]

            label = np.zeros((1, self.n_classes))
            label[0, subject-1] = 1

            y.append(label)
        
        x = np.asarray(x, dtype = object).astype('float32')
        y = np.asarray(y, dtype = object).astype('float32')

        # The initial format of "y" (label) is "a x 1 x num_classes", but the correct format
        # is "a x num_classes".
        y = y.reshape(y.shape[0], y.shape[2])

        return (x, y)