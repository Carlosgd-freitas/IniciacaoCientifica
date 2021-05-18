import numpy as np
import pyedflib
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD

# import pandas as pd
# from sklearn import metrics
# from tensorflow.keras.utils import to_categorical

np.random.seed()

# Hyperparameters
batchSize = 100       # Batch Size
trainingEpochs = 5    # Total number of training epochs - Definitivo: 60
learningRate = 0.01   # Initial learning rate

# Tasks:
# 1: Baseline, eyes open                                -> treino
# 2: Baseline, eyes closed                              -> teste
# 3: Task 1 (open and close left or right fist) - Run 1 -> treino
# 7: Task 1 (open and close left or right fist) - Run 2 -> teste

# Função para ler um arquivo EDF
def lerArquivoEDF(caminho, channels=None):
    """
    Função que lê os dados de um arquivo edf. A entrada é o nome do arquivo com o caminho até ele.
    A saída é uma lista de listas como dados numpy.
    """

    reader = pyedflib.EdfReader(caminho)

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

# Função para compor x_train, x_val e x_test
def recortarSinal(dadosX, dadosY, conteudo, tamanhoJanela, offset, limite, nIndividuo, nClasses, divisao=1.0, dadosX2=0, dadosY2=0):
    """
    Função que recorta um conteúdo (sinal EEG).

    Considerando o formato de um sinal EEG como (tam1,tam2):
        - tam1 é a quantidade de canais do sinal (eletrodos usados)
        - tam2 é o número de amostras
    Logo, tam1 não pode ser alterado, ou haveria perda de informações.
    
    Esta função consiste em uma janela deslizante de tamanho igual à __tamanhoJanela__, que irá percorrer
    por todo o sinal EEG __conteudo__, adicionando àquela janela à uma lista __dadosX__ e deslizando uma
    quantidade igual à __offset__. Os labels correspondentes serão adicionados à uma lista __dadosY__,
    utilizando os parâmetros __nIndividuo__ (Índice do indivíduo, de 1 à 109) e __nClasses__ (número total
    de classes).

    A função será executada enquanto o índice do "final da janela" for menor ou igual ao __limite__.

    Deste modo, um __offset__ menor que __tamanhoJanela__ resultará em um maior número de amostras, 
    correspondendo assim à um data augmentation.

    Caso os parâmetros adicionais __divisao__ (decimal correspondente à uma porcentagem), __dadosX2__ e
    __dadosY2__ (ambos sendo listas) sejam utilizados, Z% do processamento será armazenado em __dadosX__
    e __dadosY__ e (100-Z)% será armazenado em __dadosX2__ e __dadosY2__, sendo Z a porcentagem
    correspondente à __divisao__.
    """

    nIndividuo -= 1 # Indivíduo: 1~109 / Vetor arr: 0~108

    if offset < 0:
        print('ERRO: O offset não pode ser negativo.')
        return dadosX, dadosY
    elif offset == 0:
        print('ERRO: Um offset igual à 0 resultaria em "infinitas" janelas iguais.')
        return dadosX, dadosY
    else:
        if divisao == 1.0:
            i = tamanhoJanela
            while True:
                if i > limite:
                    break
                arr = conteudo[: , (i-tamanhoJanela):i]
                dadosX.append(arr)

                arr2 = np.zeros((1,nClasses))
                arr2[0, nIndividuo] = 1
                dadosY.append(arr2)

                i += offset
            return dadosX, dadosY
        elif divisao < 0:
            print('ERRO: O parâmetro de divisão não pode ser negativo.')
            return dadosX, dadosY
        elif divisao == 0:
            print('ERRO: Uma divisão igual à 0%'+' resultaria em duas listas vazias.')
            return dadosX, dadosY
        else:
            i = tamanhoJanela
            while True:
                if i > (limite * divisao):
                    break
                arr = conteudo[: , (i-tamanhoJanela):i]
                dadosX.append(arr)

                arr2 = np.zeros((1,nClasses))
                arr2[0, nIndividuo] = 1
                dadosY.append(arr2)

                i += offset
            while True:
                if i > limite:
                    break
                arr = conteudo[: , (i-tamanhoJanela):i]
                dadosX2.append(arr)

                arr2 = np.zeros((1,nClasses))
                arr2[0, nIndividuo] = 1
                dadosY2.append(arr2)

                i += offset
            return dadosX, dadosY, dadosX2, dadosY2

# Função que retorna os dados de treinamento, validação e teste para EO/EC
def lerDadosEOEC():
    # O offset definitivo será 20

    # Processamento de x_train, y_train, x_val e y_val
    x_trainL = list()
    x_valL = list()
    y_trainL = list()
    y_valL = list()

    x_train_1 = lerArquivoEDF('./dataset/S001R01.edf')
    x_trainL, y_train, x_valL, y_val = recortarSinal(x_trainL, y_trainL, x_train_1, 1920, 480, 9600, 1, 10, 0.9, x_valL, y_valL)
    x_train_2 = lerArquivoEDF('./dataset/S002R01.edf')
    x_trainL, y_train, x_valL, y_val = recortarSinal(x_trainL, y_trainL, x_train_2, 1920, 480, 9600, 2, 10, 0.9, x_valL, y_valL)
    x_train_3 = lerArquivoEDF('./dataset/S003R01.edf')
    x_trainL, y_train, x_valL, y_val = recortarSinal(x_trainL, y_trainL, x_train_3, 1920, 480, 9600, 3, 10, 0.9, x_valL, y_valL)
    x_train_4 = lerArquivoEDF('./dataset/S004R01.edf')
    x_trainL, y_train, x_valL, y_val = recortarSinal(x_trainL, y_trainL, x_train_4, 1920, 480, 9600, 4, 10, 0.9, x_valL, y_valL)
    x_train_5 = lerArquivoEDF('./dataset/S005R01.edf')
    x_trainL, y_train, x_valL, y_val = recortarSinal(x_trainL, y_trainL, x_train_5, 1920, 480, 9600, 5, 10, 0.9, x_valL, y_valL)
    x_train_6 = lerArquivoEDF('./dataset/S006R01.edf')
    x_trainL, y_train, x_valL, y_val = recortarSinal(x_trainL, y_trainL, x_train_6, 1920, 480, 9600, 6, 10, 0.9, x_valL, y_valL)
    x_train_7 = lerArquivoEDF('./dataset/S007R01.edf')
    x_trainL, y_train, x_valL, y_val = recortarSinal(x_trainL, y_trainL, x_train_7, 1920, 480, 9600, 7, 10, 0.9, x_valL, y_valL)
    x_train_8 = lerArquivoEDF('./dataset/S008R01.edf')
    x_trainL, y_train, x_valL, y_val = recortarSinal(x_trainL, y_trainL, x_train_8, 1920, 480, 9600, 8, 10, 0.9, x_valL, y_valL)
    x_train_9 = lerArquivoEDF('./dataset/S009R01.edf')
    x_trainL, y_train, x_valL, y_val = recortarSinal(x_trainL, y_trainL, x_train_9, 1920, 480, 9600, 9, 10, 0.9, x_valL, y_valL)
    x_train_10 = lerArquivoEDF('./dataset/S010R01.edf')
    x_trainL, y_train, x_valL, y_val = recortarSinal(x_trainL, y_trainL, x_train_10, 1920, 480, 9600, 10, 10, 0.9, x_valL, y_valL)
    x_train = np.asarray(x_trainL, dtype = object).astype('float32')
    x_val = np.asarray(x_valL, dtype = object).astype('float32')
    y_train = np.asarray(y_trainL, dtype = object).astype('float32')
    y_val = np.asarray(y_valL, dtype = object).astype('float32')

    # Processamento de x_test e y_test
    x_testL = list()
    y_testL = list()

    x_test_1 = lerArquivoEDF('./dataset/S001R02.edf')
    x_testL, y_test = recortarSinal(x_testL, y_testL, x_test_1, 1920, 1920, 9600, 1, 10)
    x_test_2 = lerArquivoEDF('./dataset/S002R02.edf')
    x_testL, y_test = recortarSinal(x_testL, y_testL, x_test_2, 1920, 1920, 9600, 2, 10)
    x_test_3 = lerArquivoEDF('./dataset/S003R02.edf')
    x_testL, y_test = recortarSinal(x_testL, y_testL, x_test_3, 1920, 1920, 9600, 3, 10)
    x_test_4 = lerArquivoEDF('./dataset/S004R02.edf')
    x_testL, y_test = recortarSinal(x_testL, y_testL, x_test_4, 1920, 1920, 9600, 4, 10)
    x_test_5 = lerArquivoEDF('./dataset/S005R02.edf')
    x_testL, y_test = recortarSinal(x_testL, y_testL, x_test_5, 1920, 1920, 9600, 5, 10)
    x_test_6 = lerArquivoEDF('./dataset/S006R02.edf')
    x_testL, y_test = recortarSinal(x_testL, y_testL, x_test_6, 1920, 1920, 9600, 6, 10)
    x_test_7 = lerArquivoEDF('./dataset/S007R02.edf')
    x_testL, y_test = recortarSinal(x_testL, y_testL, x_test_7, 1920, 1920, 9600, 7, 10)
    x_test_8 = lerArquivoEDF('./dataset/S008R02.edf')
    x_testL, y_test = recortarSinal(x_testL, y_testL, x_test_8, 1920, 1920, 9600, 8, 10)
    x_test_9 = lerArquivoEDF('./dataset/S009R02.edf')
    x_testL, y_test = recortarSinal(x_testL, y_testL, x_test_9, 1920, 1920, 9600, 9, 10)
    x_test_10 = lerArquivoEDF('./dataset/S010R02.edf')
    x_testL, y_test = recortarSinal(x_testL, y_testL, x_test_10, 1920, 1920, 9600, 10, 10)
    x_test = np.asarray(x_testL, dtype = object).astype('float32')
    y_test = np.asarray(y_testL, dtype = object).astype('float32')

    # O formato inicial de um dadosY será "a x 1 x b". O reshape fará com que um dadosY fique com o
    # formato correto "a x b"
    y_train = y_train.reshape(y_train.shape[0],y_train.shape[2])
    y_val = y_val.reshape(y_val.shape[0],y_val.shape[2])
    y_test = y_test.reshape(y_test.shape[0],y_test.shape[2])

    return x_train, x_val, x_test, y_train, y_val, y_test

# Dados
x_train, x_val, x_test, y_train, y_val, y_test = lerDadosEOEC()

# Imprimindo formatos
print(f'x_train: {x_train.shape}')
print(f'x_val: {x_val.shape}')
print(f'x_test: {x_test.shape}')
print(f'y_train: {y_train.shape}')
print(f'y_val: {y_val.shape}')
print(f'y_test: {y_test.shape}')

# One Hot Enconding
# y_train = to_categorical(y_train, num_classes=10)
# y_val = to_categorical(y_val, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)

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

# Criando o modelo
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
    model.add(Dense(10, activation='softmax')) # A camada final terá 109 nodos

    return model

model = create_model()
model.summary()

## Teste de resgape antigo ##
# train_data = tensorflow.data.Dataset.from_tensor_slices((x_train, y_train))
# valid_data = tensorflow.data.Dataset.from_tensor_slices((x_val, y_val))

# results = model.fit(train_data,
#                     batch_size = batchSize,
#                     epochs = trainingEpochs,
#                     validation_data = valid_data
#                     )
#############################

# Optimizador
opt = SGD(learning_rate=learningRate, momentum=0.9)

# Compilando o modelo
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape dos dados de input -> a x 64 x 1920 -> a x 1920 x 64
a = x_train.shape[0]
b = x_train.shape[1]
c = x_train.shape[2]

x_train = x_train.reshape(a, c, b)

a = x_val.shape[0]
b = x_val.shape[1]
c = x_val.shape[2]

x_val = x_val.reshape(a, c, b)

a = x_test.shape[0]
b = x_test.shape[1]
c = x_test.shape[2]

x_test = x_test.reshape(a, c, b)

# Definindo o LearningRateScheduler
callback = LearningRateScheduler(scheduler, verbose=0)

# Executando o modelo
results = model.fit(x_train,
                    y_train,
                    batch_size = batchSize,
                    epochs = trainingEpochs,
                    callbacks = [callback],
                    validation_data = (x_val, y_val)
                    )

# Test the model
prediction_values = model.predict_classes(x_test)
prediction_values = np.argmax(model.predict(x_test), axis=-1)

# Evaluate the model to see the accuracy
print("Evaluating on training set...")
(loss, accuracy) = model.evaluate(x_train,y_train, verbose=0)
print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

# print("Evaluating on testing set...")
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