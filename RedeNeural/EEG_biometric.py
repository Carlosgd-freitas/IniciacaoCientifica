import models
import preprocessing
import utils
import data_manipulation
import loader

import argparse
import sys
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from numpy import savetxt, loadtxt

random.seed(1051)
np.random.seed(1051)
tf.random.set_seed(1051)

# Hyperparameters
batch_size = 100                # Batch Size
training_epochs = 40            # Total number of training epochs
initial_learning_rate = 0.01    # Initial learning rate

# Parameters used in functions.load_data()
# folder_path = './Dataset_CSV/'
folder_path = '/media/work/carlosfreitas/IniciacaoCientifica/RedeNeural/Dataset_CSV/'
processed_data_path = '/media/work/carlosfreitas/IniciacaoCientifica/RedeNeural/'
num_classes = 109               # Total number of classes (individuals)

# Parameters used in functions.filter_data()
band_pass_1 = [1, 50]           # First filter option, 1~50Hz
band_pass_2 = [10, 30]          # Second filter option, 10~30Hz
band_pass_3 = [30, 50]          # Third filter option, 30~50Hz
sample_frequency = 160          # Frequency of the sampling
filter_order = 12               # Order of the filter
filter_type = 'filtfilt'        # Type of the filter used: 'sosfilt' or 'filtfilt'

# Parameters used in functions.normalize_data()
normalize_type = 'each_channel' # Type of the normalization that will be applied: 'each_channel' or 'all_channels'

# Parameters used in functions.crop_data()
window_size = 1920              # Sliding window size, used when composing the dataset
full_signal_size = 9600         # 9600 for Tasks 1-2 , 19200 for Tasks 3-14
offset = 35                     # Sliding window offset (deslocation), used when composing the dataset
split_ratio = 0.9               # 90% for training | 10% for validation

# Other Parameters
num_channels = 64               # Number of channels in an EEG signal

# Channels for some lobes of the brain
frontal_lobe   = ['Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..',
                  'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.']
motor_cortex   = ['C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..']
occipital_lobe = ['Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..']

# 9 channels present in Yang et al. article
frontal_lobe_yang = ['Af3.', 'Afz.', 'Af4.']
motor_cortex_yang = ['C1..', 'Cz..', 'C2..']
occipital_lobe_yang = ['O1..', 'Oz..', 'O2..']
all_channels_yang = ['C1..', 'Cz..', 'C2..', 'Af3.', 'Afz.', 'Af4.', 'O1..', 'Oz..', 'O2..']

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

# sun modelo
# 1000 epochs - without filtering and data augmentation - 46,4952% acurácia e 10,3115% EER
#  500 epochs - without filtering and data augmentation - 45,5024% acurácia e  9,3899% EER
#  200 epochs - without filtering and data augmentation - 31,4380% acurácia e 12,8991% EER
#
# sun modelo, versão com LSTMCell, StackedRNNCells e RNN
#  500 epochs - without filtering and data augmentation - 31,4531% acurácia e 28,7219% EER
#  100 epochs - without filtering and data augmentation - 23,5259% acurácia e 29,9559% EER
# 
# Usando 40 epocas daki pra baixo
# 5 blocos LSTM, 64 units com window_size e offset = 160 - 39,2900% acurácia, 48,4723% EER e 0.0708 Decidibilidade
#
# 5 blocos GRU bidirecional, 64 units com window_size e offset = 160 - 17,6294% acurácia, 48,7600% EER e 0.0601 Decidibilidade
#
# arquitetura ->   tempo para treinar ; tempo para testar
#    lstm 128 ->  45 min for training ; 1.92 seconds for testing
#    lstm 256 ->  64 min for training ; 2.51 seconds for testing
#     gru 128 ->  86 min for training ; 3.68 seconds for testing
#     gru 256 -> 129 min for training ; 4.99 seconds for testing
#
# lstm 128  4 units -> 37 min for training ; 1.56 seconds for testing
# lstm 128  6 units -> 53 min for training ; 2.31 seconds for testing
# lstm 128 10 units -> 84 min for training ; 3.57 seconds for testing
#
# p1 yang 64 canais -> (all killed)
# p1 yang  3 canais -> 198 min for training ; 3.38 seconds for testing
#
# EO / EC:
# lstm 128, 160 windows size,  1 offset -> killed
# lstm 128, 160 windows size,  5 offset -> 86,1612% acurácia ; 48,7972% EER ; 0,0651 Decidibilidade
#                                          48 min for training ; 3,72 seconds for testing
# lstm 128, 160 windows size, 20 offset -> 72,3977% acurácia ; 48,3229% EER ; 0,0865 Decidibilidade
#                                          13 min for training ; 3,54 seconds for testing
# lstm 128, 160 windows size, 40 offset -> 53,5499% acurácia ; 47,7848% EER ; 0,1233 Decidibilidade
#                                          6 min for training ; 3,89 seconds for testing
# lstm 128, 160 windows size, 80 offset -> 46,7058% acurácia ; 48,6823% EER ; 0,0582 Decidibilidade
#                                          3 min for training ; 3,89 seconds for testing
# lstm 128, 160 windows size, 120 offset-> 46,8562% acurácia ; 48,4491% EER ; 0,0741 Decidibilidade
#                                          2 min for training ; 3,53 seconds for testing

# functions.create_csv_database_from_edf('./Dataset/','./All_Channels_Yang/', num_classes, channels = all_channels_yang)

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datagen', action='store_true',
                    help='the model will use Data Generators to crop data on the fly')
parser.add_argument('--noptrain', action='store_true',
                    help='train/validation data will not be preprocessed and stored. This argument can only be'+
                    ' specified if --datagen was also specified')
parser.add_argument('--noptest', action='store_true',
                    help='test data will not be preprocessed and stored. This argument can only be specified if'+
                    ' --datagen was also specified')
parser.add_argument('--nofit', action='store_true',
                    help='model.fit will not be executed. The weights will be gathered from the file'+
                    ' \'model_weights.h5\', that is generated if you have ran the model in Identification mode'
                    ' at least once')
parser.add_argument('--noimode', action='store_true',
                    help='the model won\'t run in Identification Mode')
parser.add_argument('--novmode', action='store_true',
                    help='the model won\'t run in Verification Mode')

parser.add_argument('-train', nargs="+", type=int, required=True, 
                    help='list of tasks used for training and validation. All specified tasks need to be higher than\n'+
                    ' 0 and lower than 15. This is a REQUIRED flag')
parser.add_argument('-test', nargs="+", type=int, required=True, 
                    help='list of tasks used for testing. All specified tasks need to be higher than 0 and lower than\n'+
                    ' 15. This is a REQUIRED flag')
args = parser.parse_args()

train_tasks = args.train
test_tasks = args.test

if(not args.datagen):
    if(args.noptrain or args.noptest):
        print('WARNING: When not using Data Generators, both training/validation and testing data will be ' +
        'processed, so there isn\'t a option to use the --noptrain or --noptest flags.\n')

for task in train_tasks:
    if(task <= 0 or task >= 15):
        print('ERROR: All training/validation and testing tasks need to be higher than 0 and lower than 15.\n')
        sys.exit()

for task in test_tasks:
    if(task <= 0 or task >= 15):
        print('ERROR: All training/validation and testing tasks need to be higher than 0 and lower than 15.\n')
        sys.exit()

# Defining the optimizer and the learning rate scheduler
opt = SGD(learning_rate = initial_learning_rate, momentum = 0.9)
lr_scheduler = LearningRateScheduler(models.scheduler, verbose=0)

# Not using Data Generators
if(not args.datagen):
    # Loading the raw data
    train_content, test_content = loader.load_data(folder_path, train_tasks, test_tasks, 'csv', num_classes, 1)   

    # Filtering the raw data
    train_content = preprocessing.filter_data(train_content, band_pass_3, sample_frequency, filter_order, filter_type, 1)
    test_content = preprocessing.filter_data(test_content, band_pass_3, sample_frequency, filter_order, filter_type, 1)

    # Normalize the filtered data
    train_content = preprocessing.normalize_data(train_content, 'sun', 1)
    test_content = preprocessing.normalize_data(test_content, 'sun', 1)

    # Getting the training, validation and testing data
    x_train, y_train, x_val, y_val = data_manipulation.crop_data(train_content, train_tasks, num_classes,
                                                        window_size, offset, split_ratio)
    x_test, y_test = data_manipulation.crop_data(test_content, test_tasks, num_classes, window_size,
                                        window_size)

    # Training the model
    if(not args.nofit):

        # Creating the model
        model = models.create_model_mixed(window_size, num_channels, num_classes)
        model.summary()

        # Compiling, defining the LearningRateScheduler and training the model
        model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

        fit_begin = time.time()

        results = model.fit(x_train,
                            y_train,
                            batch_size = batch_size,
                            epochs = training_epochs,
                            callbacks = [lr_scheduler],
                            validation_data = (x_val, y_val)
                            )

        fit_end = time.time()
        print(f'Training time in seconds: {fit_end - fit_begin}')
        print(f'Training time in minutes: {(fit_end - fit_begin)/60.0}')
        print(f'Training time in hours: {(fit_end - fit_begin)/3600.0}\n')

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

        # Saving model weights
        model.save('model_weights.h5')

    # Running the model in Identification Mode
    if(not args.noimode):

        # Evaluate the model to see the accuracy
        model = models.create_model_mixed(window_size, num_channels, num_classes)
        model.summary()
        model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights('model_weights.h5', by_name=True)

        print('\nEvaluating on training set...')
        (loss, accuracy) = model.evaluate(x_train, y_train, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        print('Evaluating on validation set...')
        (loss, accuracy) = model.evaluate(x_val, y_val, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        print('Evaluating on testing set...')
        test_begin = time.time()

        (loss, accuracy) = model.evaluate(x_test, y_test, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        test_end = time.time()
        print(f'Evaluating on testing set time in miliseconds: {(test_end - test_begin) * 1000.0}')
        print(f'Evaluating on testing set time in seconds: {test_end - test_begin}')
        print(f'Evaluating on testing set time in minutes: {(test_end - test_begin)/60.0}\n')

    # Running the model in Verification Mode
    if(not args.novmode):

        # Removing the last layers of the model and getting the features array
        model_for_verification = models.create_model_mixed(window_size, num_channels, num_classes, True)

        model_for_verification.summary()
        model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model_for_verification.load_weights('model_weights.h5', by_name=True)
        x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

        # Calculating EER and Decidability
        y_test_classes = utils.one_hot_encoding_to_classes(y_test)
        d, eer, thresholds = utils.calc_metrics(x_pred, y_test_classes, x_pred, y_test_classes)
        print(f'EER: {eer*100.0} %')
        print(f'Decidability: {d}')

# Using Data Generators
else:

    # Processing train/validation data
    if(not args.noptrain):

        for task in train_tasks:

            if(not os.path.exists(processed_data_path + 'processed_data/task'+str(task))):
                folder = Path(processed_data_path + 'processed_data/task'+str(task))
                folder.mkdir(parents=True)

                # Loading the raw data
                train_content, test_content = loader.load_data(folder_path, [task], [], 'csv', num_classes)   

                # Filtering the raw data
                train_content = preprocessing.filter_data(train_content, band_pass_3, sample_frequency, filter_order, filter_type)

                # Normalize the filtered data
                train_content = preprocessing.normalize_data(train_content, 'sun')

                # Getting the training, validation and testing data
                x_train, y_train = data_manipulation.crop_data(train_content, [task], num_classes, full_signal_size,
                                                    full_signal_size, reshape='data_generator')

                list = []
                for index in range(0, x_train.shape[0]):
                    data = x_train[index]
                    string = 'x_subject_' + str(index+1)
                    savetxt(processed_data_path + 'processed_data/task' + str(task) + '/' + string + '.csv', data, fmt='%f', delimiter=';')
                    print(processed_data_path + 'processed_data/task' + str(task) + '/' + string + '.csv was saved.')
                    list.append(string+'.csv')
                    
                savetxt(processed_data_path + 'processed_data/task' + str(task) + '/' + 'x_list.csv', [list], delimiter=',', fmt='%s')
                print(f'file names were saved to processed_data/task{task}/x_list.csv')
    
    # Processing test data
    if(not args.noptest):
        for task in test_tasks:

            if(not os.path.exists(processed_data_path + 'processed_data/task'+str(task))):
                folder = Path(processed_data_path + 'processed_data/task'+str(task))
                folder.mkdir(parents=True)

                # Loading the raw data
                train_content, test_content = loader.load_data(folder_path, [], [task], 'csv', num_classes)   

                # Filtering the raw data
                test_content = preprocessing.filter_data(test_content, band_pass_3, sample_frequency, filter_order, filter_type)

                # Normalize the filtered data
                test_content = preprocessing.normalize_data(test_content, 'sun')

                # Getting the training, validation and testing data
                x_test, y_test = data_manipulation.crop_data(test_content, [task], num_classes, full_signal_size,
                                                    full_signal_size, reshape='data_generator')

                list = []
                for index in range(0, x_test.shape[0]):
                    data = x_test[index]
                    string = 'x_subject_' + str(index+1)
                    savetxt(processed_data_path + 'processed_data/task' + str(task) + '/' + string + '.csv', data, fmt='%f', delimiter=';')
                    print(processed_data_path + 'processed_data/task' + str(task) + '/' + string + '.csv was saved.')
                    list.append(string+'.csv')
                    
                savetxt(processed_data_path + 'processed_data/task' + str(task) + '/' + 'x_list.csv', [list], delimiter=',', fmt='%s')
                print(f'file names were saved to processed_data/task{task}/x_list.csv')

    # Training the model
    if(not args.nofit):

        # Creating the model
        model = models.create_model_mixed(window_size, num_channels, num_classes)
        model.summary()

        # Getting the file names that contains the preprocessed data
        x_train_list = []
        x_test_list = []

        for task in train_tasks:
            x_train_list.append(loadtxt(processed_data_path + 'processed_data/task'+str(task)+'/x_list.csv', delimiter=',', dtype='str'))

        for task in test_tasks:
            x_test_list.append(loadtxt(processed_data_path + 'processed_data/task'+str(task)+'/x_list.csv', delimiter=',', dtype='str'))

        x_train_list = np.asarray(x_train_list).astype('str')
        x_train_list = x_train_list.reshape(-1)
        x_train_list = x_train_list.tolist()

        x_test_list = np.asarray(x_test_list).astype('str')
        x_test_list = x_test_list.reshape(-1)
        x_test_list = x_test_list.tolist()

        # Defining the data generators
        training_generator = data_manipulation.DataGenerator(x_train_list, batch_size, window_size, offset, full_signal_size,
                                                    num_channels, num_classes, train_tasks, 'train', split_ratio)
        validation_generator = data_manipulation.DataGenerator(x_train_list, batch_size, window_size, offset, full_signal_size,
                                                    num_channels, num_classes, train_tasks, 'validation', split_ratio)
        testing_generator = data_manipulation.DataGenerator(x_test_list, batch_size, window_size, window_size, full_signal_size,
                                                    num_channels, num_classes, test_tasks, 'test', 1.0)

        # Compiling, defining the LearningRateScheduler and training the model
        model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

        fit_begin = time.time()

        results = model.fit(training_generator,
                            validation_data = validation_generator,
                            epochs = training_epochs,
                            callbacks = [lr_scheduler],
                            )

        fit_end = time.time()
        print(f'Training time in seconds: {fit_end - fit_begin}')
        print(f'Training time in minutes: {(fit_end - fit_begin)/60.0}')
        print(f'Training time in hours: {(fit_end - fit_begin)/3600.0}\n')

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
        
        # Saving model weights
        model.save('model_weights.h5')

    # Running the model in Identification Mode
    if(not args.noimode):

        # Evaluate the model to see the accuracy
        model = models.create_model_mixed(window_size, num_channels, num_classes)
        model.summary()
        model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights('model_weights.h5', by_name=True)

        print('\nEvaluating on training set...')
        (loss, accuracy) = model.evaluate(training_generator, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        print('Evaluating on validation set...')
        (loss, accuracy) = model.evaluate(validation_generator, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        print('Evaluating on testing set...')
        test_begin = time.time()

        (loss, accuracy) = model.evaluate(testing_generator, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        test_end = time.time()
        print(f'Evaluating on testing set time in miliseconds: {(test_end - test_begin) * 1000.0}')
        print(f'Evaluating on testing set time in seconds: {test_end - test_begin}')
        print(f'Evaluating on testing set time in minutes: {(test_end - test_begin)/60.0}\n')
    
    # Running the model in Verification Mode
    if(not args.novmode):

        # List of files that contains x_test data
        x_test_list = []
        for task in test_tasks:
            x_test_list.append(loadtxt(processed_data_path + 'processed_data/task'+str(task)+'/x_list.csv', delimiter=',', dtype='str'))
        x_test_list = np.asarray(x_test_list).astype('str')
        x_test_list = x_test_list.reshape(-1)
        x_test_list = x_test_list.tolist()

        # Loading x_test data
        temp_x = []
        subjects = []
        for task in test_tasks:
            for i, ID in enumerate(x_test_list):
                file_x = np.loadtxt(processed_data_path + 'processed_data/task' + str(task) + '/' + x_test_list[i], delimiter=';', usecols=range(num_channels))
                string = processed_data_path + 'processed_data/task' + str(task) + '/' + x_test_list[i]

                file_x = np.asarray(file_x, dtype = object).astype('float32')
                file_x = file_x.T
                temp_x.append(file_x)

                string = string.split("_subject_")[1]    # 'X.csv'
                subject = int(string.split(".csv")[0])   # X
                subjects.append(subject)

        # Cropping y_test data
        x_dataL = list()
        y_dataL = list()

        pos = 0
        for data in temp_x:
            x_dataL, y_dataL = data_manipulation.signal_cropping(x_dataL, y_dataL, data, window_size, window_size, subjects[pos],
                                num_classes, mode='labels_only')
            pos += 1
        
        y_test = np.asarray(y_dataL, dtype = object).astype('float32')
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])

        # Defining the generator
        testing_generator = data_manipulation.DataGenerator(x_test_list, batch_size, window_size, window_size, full_signal_size,
                                                    num_channels, num_classes, test_tasks, 'test', 1.0)

        # Removing the last layers of the model and getting the features array
        model_for_verification = models.create_model_mixed(window_size, num_channels, num_classes, True)

        model_for_verification.summary()
        model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model_for_verification.load_weights('model_weights.h5', by_name=True)
        x_pred = model_for_verification.predict(testing_generator, batch_size=batch_size)

        # Calculating EER and Decidability
        y_test_classes = utils.one_hot_encoding_to_classes(y_test)
        d, eer, thresholds = utils.calc_metrics(x_pred, y_test_classes, x_pred, y_test_classes)
        print(f'EER: {eer*100.0} %')
        print(f'Decidability: {d}')
