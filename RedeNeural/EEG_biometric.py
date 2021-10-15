import models
import functions

import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam

np.random.seed()

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
window_size = 1920 # 160              # Sliding window size, used when composing the dataset
offset = 35 # 160                     # Sliding window offset (deslocation), used when composing the dataset
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
# 5 blocos LSTM com 16 units cada - 90,6422% acurácia, 47,1237% EER e 0.1045 Decidibilidade
# 5 blocos LSTM com 32 units cada - 92,2936% acurácia, 47,6884% EER e 0.1032 Decidibilidade
# 5 blocos LSTM com 64 units cada - 94,3119% acurácia, 48,8325% EER e 0.1184 Decidibilidade
# melhor LSTM com window_size e offset = 160 - 39,2900% acurácia, 48,4723% EER e 0.0708 Decidibilidade
#
# 5 blocos GRU bidirecionais com 16 units cada - 74,8624% acurácia, 47,8345% EER e 0.1074 Decidibilidade
# 5 blocos GRU bidirecionais com 32 units cada - 81,1009% acurácia, 47,6330% EER e 0.1225 Decidibilidade
# 5 blocos GRU bidirecionais com 64 units cada - 85,1376% acurácia, 47,2504% EER e 0.1825 Decidibilidade
# melhor GRU bidirecional com window_size e offset = 160 - 17,6294% acurácia, 48,7600% EER e 0.0601 Decidibilidade

# functions.create_csv_database_from_edf('./Dataset/','./Dataset_CSV/', num_classes)

# Creating the model
model = models.create_model_mixed(window_size, num_channels, num_classes)
model.summary()

# Loading the raw data
train_content, test_content = functions.load_data(folder_path, train_tasks, test_tasks, 'csv', num_classes, verbose=1)   

# Filtering the raw data
train_content = functions.filter_data(train_content, band_pass_3, sample_frequency, filter_order, filter_type, verbose=1)
test_content = functions.filter_data(test_content, band_pass_3, sample_frequency, filter_order, filter_type, verbose=1)

# Normalize the filtered data
train_content = functions.normalize_data(train_content, 'sun', verbose=1)
test_content = functions.normalize_data(test_content, 'sun', verbose=1)

# Apply data augmentation (sliding window cropping) on normalized data
x_train, y_train, x_val, y_val = functions.crop_data(train_content, train_tasks, num_classes,
                                                     window_size, offset, split_ratio, verbose=1)
x_test, y_test = functions.crop_data(test_content, test_tasks, num_classes, window_size, window_size, verbose=1)

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
# opt = Adam(learning_rate = 0.0001)

model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

fit_begin = time.time()

callback = LearningRateScheduler(models.scheduler, verbose=0)
results = model.fit(x_train,
                    y_train,
                    batch_size = batch_size,
                    epochs = training_epochs,
                    callbacks = [callback],
                    validation_data = (x_val, y_val)
                    )

fit_end = time.time()
print(f'Training time in seconds: {fit_end - fit_begin}')
print(f'Training time in minutes: {(fit_end - fit_begin)/60.0}')
print(f'Training time in hours: {(fit_end - fit_begin)/3600.0}\n')

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
test_begin = time.time()

(loss, accuracy) = model.evaluate(x_test, y_test, verbose = 0)
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
model_for_verification = models.create_model_mixed(window_size, num_channels, num_classes, True)

model_for_verification.summary()
model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
model_for_verification.load_weights('model_weights.h5', by_name=True)
x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

# Calculating EER and Decidability
y_test_classes = functions.one_hot_encoding_to_classes(y_test)
d, eer, thresholds = functions.calc_metrics(x_pred, y_test_classes, x_pred, y_test_classes)
print(f'EER: {eer*100.0} %')
print(f'Decidability: {d}')
