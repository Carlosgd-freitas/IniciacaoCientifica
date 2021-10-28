import models
import functions

import shutil
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam
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
train_tasks = [1]               # Tasks used for training and validation
test_tasks = [2]                # Tasks used for testing
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

# First process the data, then train the model, then evaluate the model
print('Press [1] and [ENTER] to process the data')
print('Press [2] and [ENTER] to run the model')
option = int(input('Enter option: '))
    
if(option == 1):  
    if(os.path.exists('processed_data')):
        shutil.rmtree('processed_data', ignore_errors=True)
    os.mkdir('processed_data')

    # Loading the raw data
    train_content, test_content = functions.load_data(folder_path, train_tasks, test_tasks, 'csv', num_classes, 1)   

    # Filtering the raw data
    train_content = functions.filter_data(train_content, band_pass_3, sample_frequency, filter_order, filter_type, 1)
    test_content = functions.filter_data(test_content, band_pass_3, sample_frequency, filter_order, filter_type, 1)

    # Normalize the filtered data
    train_content = functions.normalize_data(train_content, 'sun', 1)
    test_content = functions.normalize_data(test_content, 'sun', 1)

    test_content = np.asarray(test_content, dtype='float32')
    print(f'test_content.shape = {test_content.shape}')
    print(f'test_content = {test_content}')
    input('quitaste?')

    # Getting the training, validation and testing data
    # x_train, y_train, x_val, y_val = functions.crop_data(train_content, train_tasks, num_classes,
    #                                                      window_size, offset, split_ratio)
    # x_test, y_test = functions.crop_data(test_content, test_tasks, num_classes, window_size, window_size)

    # print('\nData formats:')
    # print(f'x_train: {x_train.shape}')
    # print(f'x_val: {x_val.shape}')
    # print(f'x_test: {x_test.shape}')
    # print(f'y_train: {y_train.shape}')
    # print(f'y_val: {y_val.shape}')
    # print(f'y_test: {y_test.shape}\n')

    list = []
    list_2 = []
    for index in range(0, x_train.shape[0]):
        data = x_train[index]
        string = 'x_train_' + str(index)
        savetxt('processed_data/'+string+'.csv', data, fmt='%f', delimiter=';')
        list.append(string+'.csv')
        
        data = y_train[index]
        string = 'y_train_' + str(index)
        savetxt('processed_data/'+string+'.csv', data, fmt='%d', delimiter=';')
        list_2.append(string+'.csv')
    savetxt('processed_data/x_train_list.csv', [list], delimiter=',', fmt='%s')
    savetxt('processed_data/y_train_list.csv', [list_2], delimiter=',', fmt='%s')

    list = []
    list_2 = []
    for index in range(0, x_val.shape[0]):
        data = x_val[index]
        string = 'x_val_' + str(index)
        savetxt('processed_data/'+string+'.csv', data, fmt='%f', delimiter=';')
        list.append(string+'.csv')
        
        data = y_val[index]
        string = 'y_val_' + str(index)
        savetxt('processed_data/'+string+'.csv', data, fmt='%d', delimiter=';')
        list_2.append(string+'.csv')
    savetxt('processed_data/x_val_list.csv', [list], delimiter=',', fmt='%s')
    savetxt('processed_data/y_val_list.csv', [list_2], delimiter=',', fmt='%s')
    
    list = []
    list_2 = []
    for index in range(0, x_test.shape[0]):
        data = x_test[index]
        string = 'x_test_' + str(index)
        savetxt('processed_data/'+string+'.csv', data, fmt='%f', delimiter=';')
        list.append(string+'.csv')
        
        data = y_test[index]
        string = 'y_test_' + str(index)
        savetxt('processed_data/'+string+'.csv', data, fmt='%d', delimiter=';')
        list_2.append(string+'.csv')
    savetxt('processed_data/x_test_list.csv', [list], delimiter=',', fmt='%s')
    savetxt('processed_data/y_test_list.csv', [list_2], delimiter=',', fmt='%s')

elif(option == 2):
    # Creating the model
    model = models.create_model(window_size, num_channels, num_classes)
    model.summary()

    # Composing the dictionary
    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    x_train_list.append(loadtxt('processed_data/x_train_list.csv', delimiter=',', dtype='str'))
    y_train_list.append(loadtxt('processed_data/y_train_list.csv', delimiter=',', dtype='str'))
    x_val_list.append(loadtxt('processed_data/x_val_list.csv', delimiter=',', dtype='str'))
    y_val_list.append(loadtxt('processed_data/y_val_list.csv', delimiter=',', dtype='str'))
    x_test_list.append(loadtxt('processed_data/x_test_list.csv', delimiter=',', dtype='str'))
    y_test_list.append(loadtxt('processed_data/y_test_list.csv', delimiter=',', dtype='str'))

    x_train_list = np.asarray(x_train_list).astype('str')
    x_train_list = x_train_list.tolist()
    x_train_list = x_train_list[0]

    # y_train_list = np.asarray(y_train_list).astype('str')
    # y_train_list = y_train_list.tolist()
    # y_train_list = y_train_list[0]

    x_val_list = np.asarray(x_val_list).astype('str')
    x_val_list = x_val_list.tolist()
    x_val_list = x_val_list[0]

    # y_val_list = np.asarray(y_val_list).astype('str')
    # y_val_list = y_val_list.tolist()
    # y_val_list = y_val_list[0]

    x_test_list = np.asarray(x_test_list).astype('str')
    x_test_list = x_test_list.tolist()
    x_test_list = x_test_list[0]

    # y_test_list = np.asarray(y_test_list).astype('str')
    # y_test_list = y_test_list.tolist()
    # y_test_list = y_test_list[0]

    # data = {'train': x_train_list, 'validation': x_val_list, 'test': x_test_list}
    # labels = {'train': y_train_list, 'validation': y_val_list, 'test': y_test_list}

    print('x_train_list:')
    print(x_train_list)
    print('\ny_train_list:')
    print(y_train_list)

    training_generator = functions.DataGenerator(x_train_list, batch_size, window_size, offset,
                                                 num_channels, num_classes, False)
    validation_generator = functions.DataGenerator(x_val_list, batch_size, window_size, offset,
                                                   num_channels, num_classes, False)
    testing_generator = functions.DataGenerator(x_test_list, batch_size, window_size, window_size,
                                                num_channels, num_classes, False)                                             

    # Defining the optimizer, compiling, defining the LearningRateScheduler and training the model
    opt = SGD(learning_rate = initial_learning_rate, momentum = 0.9)
    # opt = Adam(learning_rate = 0.0001)

    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

    fit_begin = time.time()

    callback = LearningRateScheduler(models.scheduler, verbose=0)
    results = model.fit_generator(generator = training_generator,
                        validation_data = validation_generator,
                        epochs = training_epochs,
                        callbacks = [callback],
                        )

    # results = model.fit(x_train,
    #                     y_train,
    #                     batch_size = batch_size,
    #                     epochs = training_epochs,
    #                     callbacks = [callback],
    #                     validation_data = (x_val, y_val)
    #                     )

    fit_end = time.time()
    print(f'Training time in seconds: {fit_end - fit_begin}')
    print(f'Training time in minutes: {(fit_end - fit_begin)/60.0}')
    print(f'Training time in hours: {(fit_end - fit_begin)/3600.0}\n')

    # Saving model weights
    model.save('model_weights.h5')

    # Evaluate the model to see the accuracy
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

    # Removing the last layers of the model and getting the features array
    # model_for_verification = models.create_model(window_size, num_channels, num_classes, True)

    # model_for_verification.summary()
    # model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # model_for_verification.load_weights('model_weights.h5', by_name=True)
    # x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

    # # Calculating EER and Decidability
    # y_test_classes = functions.one_hot_encoding_to_classes(y_test)
    # d, eer, thresholds = functions.calc_metrics(x_pred, y_test_classes, x_pred, y_test_classes)
    # print(f'EER: {eer*100.0} %')
    # print(f'Decidability: {d}')

else:
    print('ERROR: Enter a valid option.')