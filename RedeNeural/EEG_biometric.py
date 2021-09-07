import models
import functions
# import genetic_algorithm as ga

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD

import pickle
from deap import algorithms, base, tools, creator
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.models import Model, Sequential
import array, random

np.random.seed()

# Hyperparameters
batch_size = 100                # Batch Size
training_epochs = 60            # Total number of training epochs
initial_learning_rate = 0.01    # Initial learning rate

# Parameters used in functions.load_data()
# folder_path = './Dataset/'
folder_path = '/media/work/carlosfreitas/IniciacaoCientifica/RedeNeural/Dataset/'
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

# Valores utilizando modo 'filtfilt':
# offset: 20, num_classes: 108
# acurácia / EER / Decidibilidade
# band_pass_1 = 5.1852% / 7.5671% / 3.0792
# band_pass_2 = 4.2593% / 3.6131% / 4.1049
# band_pass_3 = 3.4862% / 0.8253% / 6.5402

# Creating the model
# model = models.create_model(window_size, num_channels, num_classes)
# model = models.create_model_inception(window_size, num_channels, num_classes)
# model = models.create_model_SE(window_size, num_channels, num_classes)
# model = models.create_model_transformers(window_size, num_channels, num_classes)
# model = models.create_model_LSTM(window_size, num_channels, num_classes)
# model = models.create_model_GRU(window_size, num_channels, num_classes)
# model.summary()

# Loading the raw data
train_content, test_content = functions.load_data(folder_path, train_tasks, test_tasks, num_classes)   

# Filtering the raw data
train_content = functions.filter_data(train_content, band_pass_3, sample_frequency, filter_order, filter_type)
test_content = functions.filter_data(test_content, band_pass_3, sample_frequency, filter_order, filter_type)

# Normalize the filtered data
# train_content = functions.normalize_data(train_content, normalize_type)
# test_content = functions.normalize_data(test_content, normalize_type)

# Apply data augmentation (sliding window cropping) on normalized data
x_train, y_train, x_val, y_val = functions.crop_data(train_content, train_tasks, num_classes,
                                                     window_size, offset, split_ratio)
x_test, y_test = functions.crop_data(test_content, test_tasks, num_classes, window_size, window_size)

# Printing data formats
print('\nData formats:')
print(f'x_train: {x_train.shape}')
print(f'x_val: {x_val.shape}')
print(f'x_test: {x_test.shape}')
print(f'y_train: {y_train.shape}')
print(f'y_val: {y_val.shape}')
print(f'y_test: {y_test.shape}\n')

##################################### Genetic algorithm #####################################
class SwishActivation(Activation):
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

max_conv_layers = 0
max_dense_layers = 4 # precisa de pelo menos um para o softmax

filter_range_max = 512
kernel_range_max = 7
max_dense_nodes = 512

# fixado para A AD100
input_shape = (1920, 64) # input_shape = (window_size, num_channels)
n_classes = 109

def decode(genome, verbose=False):
    batch_normalization=True
    dropout=True
    max_pooling=True
    optimizers=None 
    activations=None

    optimizer = [
        'adam',
        'rmsprop',
        'adagrad',
        'adadelta'
    ]

    activation =  [
        'relu',
        'sigmoid',
        swish_act,
    ]

    convolutional_layer_shape = [
        "active",
        "num filters",
        "kernel_size",
        "batch normalization",
        "activation",
        "dropout",
        "max pooling activation",
        "max pooling size",
    ]

    dense_layer_shape = [
        "active",
        "num nodes",
        "batch normalization",
        "activation",
        "dropout",
    ]

    convolution_layers = max_conv_layers
    convolution_layer_size = len(convolutional_layer_shape)
    dense_layers = max_dense_layers # excluindo a softmax layer
    dense_layer_size = len(dense_layer_shape)

    model = models.create_model_mixed(window_size, num_channels, num_classes)
    x = model.output

    offset = 0

    # mapear entre um range para outro (fazer de forma mais eficiente!)
    def map_range(value, leftMin, leftMax, rightMin, rightMax):
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

        # Convert the 0-1 range into a value in the right range.
        return rightMin + (valueScaled * rightSpan)

    for i in range(dense_layers):
        if round(genome[offset])==1:
            dense = None
            dense = Dense(round(map_range(genome[offset + 1],0,1,4,max_dense_nodes)))
            x = Dense(round(map_range(genome[offset + 1],0,1,4,max_dense_nodes))) (x)
            if round(genome[offset + 2]) == 1:
                x = BatchNormalization()(x)
        
            x = Activation(activation[round(map_range(genome[offset + 3],0,1,0,len(activation)-1))])(x)
            x = Dropout(float(map_range(genome[offset + 4], 0, 1, 0, 0.7)))(x)

            if verbose==True:
                print('\n Dense%d' % i)
                print('Max Nodes = %d' % round(map_range(genome[offset + 1],0,1,4,max_dense_nodes)))
                if round(genome[offset + 2]) == 1:
                    print('Batch Norm')
                print('Activation=%s' % activation[int(round(genome[offset + 3]))])
                print('Dropout=%f' % float(map_range(genome[offset + 5], 0, 1, 0, 0.7)))
            
        offset += dense_layer_size

    predictions = Dense(n_classes, activation="softmax")(x)
    model = Model(inputs = model.input, outputs = predictions)

    model.compile(loss='categorical_crossentropy',
        optimizer=optimizer[round(map_range(genome[offset],0,1,0,len(activation)-1))],
        metrics=["accuracy"])

    if verbose==True:
        print('\n Optimizer: %s \n' % optimizer[round(map_range(genome[offset],0,1,0,len(activation)-1))])

    return model

def evaluate_individual(genome): 
    n_epochs = 8
    model = decode(genome, True)
    loss, accuracy, num_parameters = None, None, None

    fit_params = {
        'x': x_train,
        'y': y_train,
        'validation_split': 0.2,
        'epochs': n_epochs,
        'verbose': 1,
        'callbacks': [
            EarlyStopping(monitor='val_loss', patience=2, verbose=1)
        ]
    }

    fit_params['validation_data'] = (x_val, y_val)

    model.fit(**fit_params)
    (loss, accuracy) = model.evaluate(x_val, y_val, verbose=0)
    num_parameters = model.count_params()

    # return loss
    return accuracy

def prepare_toolbox(problem_instance, number_of_variables, bounds_low, bounds_up):
    
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    
    toolbox = base.Toolbox()
    
    toolbox.register('evaluate', problem_instance)
    toolbox.register('select', tools.selRoulette)
    
    toolbox.register("attr_float", uniform, bounds_low, bounds_up, number_of_variables)
    toolbox.register("individual1", tools.initIterate, creator.Individual1, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual1)
    
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                     low=bounds_low, up=bounds_up, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, 
                     low=bounds_low, up=bounds_up, eta=20.0, 
                     indpb=1.0/number_of_variables)

    # default
    toolbox.pop_size = 10   # population size
    toolbox.max_gen = 5     # max number of iteration
    toolbox.mut_prob = 1/number_of_variables
    toolbox.cross_prob = 0.3
    
    return toolbox

def ga(toolbox, tools, pop_size, num_generations, recover_last_run=None, checkpoint=None):
    if recover_last_run and checkpoint:
        print("\nRetomando ultima execucao.. ]")
        # A file name has been given, then load the data from the file
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    else:
        print("\nIniciando nova evolucao ]")
        # Start a new evolution
        population = toolbox.population(n=pop_size)
        start_gen = 0
        halloffame = tools.HallOfFame(maxsize=3)
        logbook = tools.Logbook()

    NGEN = num_generations
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    hof = tools.HallOfFame(3)
    
    for gen in range(start_gen, NGEN):
        print('\n **** Geracao %d  ****' % gen )
        population = algorithms.varAnd(population, toolbox, cxpb=0.4, mutpb=0.1)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        population = toolbox.select(population, len(population))

        #if gen % FREQ == 0: # SALVA TODAS GERAÇÕES
        # Fill the dictionary using the dict(key=value[, ...]) constructor
        cp = dict(population=population, generation=gen, halloffame=halloffame,
                  logbook=logbook, rndstate=random.getstate())

        if checkpoint:
            with open(checkpoint, "wb") as cp_file:
                pickle.dump(cp, cp_file)

    # Print top N solutions 
    # best_individuals = tools.selBest(halloffame, k = 3) # if evaluate_individual returns loss
    best_individuals = tools.selWorst(halloffame, k = 3)  # if evaluate_individual returns accuracy
    
    print("\n\n ******* Best solution is: *******\n")
    for bi in best_individuals:
        decode(bi, True)
      
    print("\n")
    print("\n")
    return best_individuals

def genetic_run():

    population_size = 5     # num of solutions in the population
    num_generations = 8     # num of time we generate new population

    creator.create("FitnessMax1", base.Fitness, weights=(-1.0,) * 1)
    creator.create("Individual1", array.array, typecode='d', fitness=creator.FitnessMax1)

    number_of_variables = max_dense_layers*5 + 1 # convlayers, GAPlayer, denselayers, optimizer

    bounds_low, bounds_up = 0, 1 # valores sao remapeados em decode

    toolbox = prepare_toolbox(evaluate_individual, 
                              number_of_variables,
                              bounds_low, bounds_up)

    # chama o metodo genetico
    best_individuals = ga(toolbox, tools, population_size, num_generations)
    return best_individuals

best_individuals = genetic_run()

i = 1
for ind in best_individuals:
    print(f'accuracy do individuo #{i}: {ind.fitness.values}')
    i += 1

model = decode(best_individuals[0], True)
model.summary()
####################################################################################################

# Defining the optimizer, compiling, defining the LearningRateScheduler and training the model
opt = SGD(learning_rate = initial_learning_rate, momentum = 0.9)
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
callback = LearningRateScheduler(models.scheduler, verbose=0)
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

# Removing the last layers of the model and getting the features array

# Using default model
# model_for_verification = models.create_model(window_size, num_channels, num_classes, True) 

# Model with inception blocks
# model_for_verification = models.create_model_with_inception(window_size, num_channels, num_classes, True)

# Model with squeeze & excitation blocks
# model_for_verification = models.create_model_with_SE(window_size, num_channels, num_classes, True)

# Model with transformers
# model_for_verification = models.create_model_transformers(window_size, num_channels, num_classes, True)

# Model with LSTM
# model_for_verification = models.create_model_LSTM(window_size, num_channels, num_classes, True)

# Model with GRU
# model_for_verification = models.create_model_GRU(window_size, num_channels, num_classes, True)

# model_for_verification.summary()
# model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model_for_verification.load_weights('model_weights.h5', by_name=True)
# x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

# Calculating EER and Decidability
# y_test_classes = functions.one_hot_encoding_to_classes(y_test)
# d, eer, thresholds = functions.calc_metrics(x_pred, y_test_classes, x_pred, y_test_classes)
# print(f'EER: {eer*100.0} %')
# print(f'Decidability: {d}')
