# Função que retorna os dados de treinamento, validação e teste para EO/EC
def lerDadosEOEC():
    # O offset definitivo será 20

    # Processamento de x_train
    x_trainL = list()
    x_train_1 = lerArquivoEDF('./dataset/S001R01.edf')
    x_trainL = recortarSinal(x_trainL, x_train_1, 1920, 960, 9600)
    x_train_2 = lerArquivoEDF('./dataset/S002R01.edf')
    x_trainL = recortarSinal(x_trainL, x_train_2, 1920, 960, 9600)
    x_train_3 = lerArquivoEDF('./dataset/S003R01.edf')
    x_trainL = recortarSinal(x_trainL, x_train_3, 1920, 960, 9600)
    x_train_4 = lerArquivoEDF('./dataset/S004R01.edf')
    x_trainL = recortarSinal(x_trainL, x_train_4, 1920, 960, 9600)
    x_train_5 = lerArquivoEDF('./dataset/S005R01.edf')
    x_trainL = recortarSinal(x_trainL, x_train_5, 1920, 960, 9600)
    x_train_6 = lerArquivoEDF('./dataset/S006R01.edf')
    x_trainL = recortarSinal(x_trainL, x_train_6, 1920, 960, 9600)
    x_train_7 = lerArquivoEDF('./dataset/S007R01.edf')
    x_trainL = recortarSinal(x_trainL, x_train_7, 1920, 960, 9600)
    x_train_8 = lerArquivoEDF('./dataset/S008R01.edf')
    x_trainL = recortarSinal(x_trainL, x_train_8, 1920, 960, 9600)
    x_train_9 = lerArquivoEDF('./dataset/S009R01.edf')
    x_trainL = recortarSinal(x_trainL, x_train_9, 1920, 960, 9600)
    x_train = np.asarray(x_trainL, dtype = object)

    # Processamento de y_train
    y_train = montarLabels(81, 10, 9, 1)

    # Processamento de x_val
    x_valL = list()
    x_val_1 = lerArquivoEDF('./dataset/S010R01.edf')
    x_valL = recortarSinal(x_valL, x_val_1, 1920, 960, 9600)
    x_val = np.asarray(x_valL, dtype = object)

    # Processamento de y_val
    y_val = montarLabels(9, 10, 9, 10)

    # Processamento de x_test
    x_testL = list()
    x_test_1 = lerArquivoEDF('./dataset/S001R02.edf')
    x_testL = recortarSinal(x_testL, x_test_1, 1920, 1920, 9600)
    x_test_2 = lerArquivoEDF('./dataset/S002R02.edf')
    x_testL = recortarSinal(x_testL, x_test_2, 1920, 1920, 9600)
    x_test_3 = lerArquivoEDF('./dataset/S003R02.edf')
    x_testL = recortarSinal(x_testL, x_test_3, 1920, 1920, 9600)
    x_test_4 = lerArquivoEDF('./dataset/S004R02.edf')
    x_testL = recortarSinal(x_testL, x_test_4, 1920, 1920, 9600)
    x_test_5 = lerArquivoEDF('./dataset/S005R02.edf')
    x_testL = recortarSinal(x_testL, x_test_5, 1920, 1920, 9600)
    x_test_6 = lerArquivoEDF('./dataset/S006R02.edf')
    x_testL = recortarSinal(x_testL, x_test_6, 1920, 1920, 9600)
    x_test_7 = lerArquivoEDF('./dataset/S007R02.edf')
    x_testL = recortarSinal(x_testL, x_test_7, 1920, 1920, 9600)
    x_test_8 = lerArquivoEDF('./dataset/S008R02.edf')
    x_testL = recortarSinal(x_testL, x_test_8, 1920, 1920, 9600)
    x_test_9 = lerArquivoEDF('./dataset/S009R02.edf')
    x_testL = recortarSinal(x_testL, x_test_9, 1920, 1920, 9600)
    x_test_10 = lerArquivoEDF('./dataset/S010R02.edf')
    x_testL = recortarSinal(x_testL, x_test_10, 1920, 1920, 9600)
    x_test = np.asarray(x_testL, dtype = object)

    # Processamento de y_test
    y_test = montarLabels(50, 10, 5, 1)

    return x_train, x_val, x_test, y_train, y_val, y_test

# Função para compor y_train, y_val e y_test
def montarLabels(nAmostras, nClasses, nAmostrasPorIndividuo, individuoinicial):
    """
    Função que monta as labels referentes à x_train, x_val e x_test.

    Parâmetros:
        - nAmostras: número total de amostras. Este valor é equivalente ao primeiro parâmetro no formato
        de x_train, x_val e x_test.
        - nClasses: número de indivíduos, cujos dados foram lidos pela função lerArquivoEDF().
        - nAmostrasPorIndividuo: número de amostras geradas por indivíduo. Este número dependerá do offset
        utilizado na função recortarSinal().
        - individuoinicial: Índice do primeiro indivíduo lido (1~109). Este parâmetro é particularmente
        útil na definição de y_val.
    
    O numpy array retornado na saída dessa função terá um formato nAmostras x nClasses.
    """
    arr = np.zeros((nAmostras, nClasses))
    
    i = 0                    # linhas = nAmostras
    j = individuoinicial - 1 # colunas = nClasses
    k = 0

    while i < nAmostras:
        k = 0
        while k < nAmostrasPorIndividuo:
            arr[i, j] = 1
            i += 1
            k += 1
        j += 1

    return arr

# Código antigo do distribution do signal_cropping
if distribution == 1.0:
            i = window_size
            while i <= content.shape[1]:
                arr = content[: , (i-window_size):i]
                x_data.append(arr)

                arr2 = np.zeros((1,num_classes))
                arr2[0, num_subject] = 1
                y_data.append(arr2)

                i += offset
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
            while i <= content.shape[1]:
                arr = content[: , (i-window_size):i]
                x_data_2.append(arr)

                arr2 = np.zeros((1,num_classes))
                arr2[0, num_subject] = 1
                y_data_2.append(arr2)

                i += offset
            return x_data, y_data, x_data_2, y_data_2

def same_subject(index_1, index_2, total_samples):
    """
    Takes the index of 2 samples and returns if they correspond to the same subject or not.

    A part of this function consists on calculating aux and aux2:
        - aux: index of the first sample that corresponds to the same subject as the lowest index in the input;
        - aux_2: index of the last sample that corresponds to the same subject as the lowest index in the input;

    If the indexes of the 2 samples are in the interval [aux, aux2], then this function returns 1,
    indicating they correspond to the same subject. Otherwise, this function returns 0, indicating they
    correspond to different subjects.

    Parameters:
        - index_1: index of the first sample;
        - index_2: index of the second sample;
        - total_samples: total number of samples;
    """

    samples_per_subject = total_samples / num_classes
    aux = 0

    if index_1 == index_2: # Same sample
        return 1
    elif index_1 < index_2:
        while aux < index_1:
            if aux + samples_per_subject > index_1:
                break
            else:
                aux += samples_per_subject
        aux_2 = aux + samples_per_subject - 1

        if index_1 >= aux and index_1 <= aux_2 and index_2 >= aux and index_2 <= aux_2:
            return 1
        else:
            return 0
    else:
        while aux < index_2:
            if aux + samples_per_subject > index_2:
                break
            else:
                aux += samples_per_subject
        aux_2 = aux + samples_per_subject - 1

        if index_1 >= aux and index_1 <= aux_2 and index_2 >= aux and index_2 <= aux_2:
            return 1
        else:
            return 0

# All-vs-all comparations
i = 0
j = 0
k = x_pred.shape[0]
t = 200 # Threshold

true_genuine = 0
false_genuine = 0
true_imposter = 0
false_imposter = 0

while i < k:
    j = 0
    while j < k:
        dist = np.linalg.norm(x_pred[i] - x_pred[j])

        if dist <= t: # Genuine
            if same_subject(i, j, k) == 1:
                true_genuine += 1
            else:
                false_genuine += 1
        else: # Imposter
            if same_subject(i, j, k) == 1:
                true_imposter += 1
            else:
                false_imposter += 1
        j += 1
    i += 1

print(f'true_genuine = {true_genuine}')
print(f'false_genuine = {false_genuine}')
print(f'true_imposter = {true_imposter}')
print(f'false_imposter = {false_imposter}')

far = false_genuine/(false_genuine + true_imposter)
frr = false_imposter/(false_imposter + true_genuine)

print(f'far = {far}')
print(f'frr = {frr}')

# Plotando o data augmentation
def load_data(folder_path, train_task, test_task):
    """
    Returns the processed signals and labels for training (x_train and y_train), validation (x_val and y_val) and
    testing (x_test and y_test).

    The return of this function is in the format: x_train, x_val, x_test, y_train, y_val, y_test.

    Parameters:
        - folder_path: path of the folder in which the the EDF files are stored.
        E.g. if this python script is in the same folder as the sub-folder used to store the EDF files, and this
        sub-folder is called "Dataset", then this parameter should be: './Dataset/';
        - train_task: number of the experimental run that will be used to create train and validation data;
        - test_task: number of the experimental run that will be used to create testing data.
    """

    # Processing x_train, y_train, x_val and y_val
    x_trainL = list()
    x_valL = list()
    y_trainL = list()
    y_valL = list()

    fig, axs = plt.subplots(3) #

    for i in range(1, num_classes + 1):
        content_EO = read_EDF(folder_path+'S{:03d}R{:02d}.edf'.format(i,train_task))
        content_EO = pre_processing(content_EO, 4)

        if i == 1:
            print(f'content_EO[0].shape = {content_EO[0].shape}')
            axs[0].axis([0, 10000, 0, 1])
            axs[0].plot(content_EO[0], 'b-') #

        x_trainL, y_trainL, x_valL, y_valL = signal_cropping(x_trainL, y_trainL, content_EO, window_size, offset, i, num_classes, distribution, x_valL, y_valL)
    
    x_train = np.asarray(x_trainL, dtype = object).astype('float32')
    x_val = np.asarray(x_valL, dtype = object).astype('float32')
    y_train = np.asarray(y_trainL, dtype = object).astype('float32')
    y_val = np.asarray(y_valL, dtype = object).astype('float32')

    #
    print(f'x_train.shape = {x_train.shape}')
    print(f'x_train[0].shape = {x_train[0].shape}')
    print(f'x_train[1].shape = {x_train[1].shape}')
    
    axs[1].axis([0, 2000, 0, 1])
    axs[1].plot(x_train[0][0], 'r-')
    axs[2].axis([0, 2000, 0, 1])
    axs[2].plot(x_train[1][0], 'g-')
    plt.show()
    #

    # Processing x_test and y_test
    x_testL = list()
    y_testL = list()

    for i in range(1, num_classes + 1):
        content_EC = read_EDF(folder_path+'S{:03d}R{:02d}.edf'.format(i,test_task))
        content_EC = pre_processing(content_EC, 4)
        x_testL, y_testL = signal_cropping(x_testL, y_testL, content_EC, window_size, window_size, i, num_classes)

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

def InceptionBasicBlock(input_img, block_index, filters_sizes=(64, 96, 128, 16, 32, 128, 32), factor=1):
    """
    Creates and returns an inception block for a CNN.

    Parameters:
        - input_img: input data for the inception block;
        - block_index: index of the inception block;
    
    Optional Parameters:
        - filters_sizes: tuple of filter sizes for each of the 7 convolution layers of this inception block. Default
        tuple is (64, 96, 128, 16, 32, 128, 32);
        - factor: used to multiply the number of filters used in each convolution layer simultaneously. Default
        value is 1:
    """
    conv1_1_1 = Conv1D(int(filters_sizes[0] * factor), 1, padding='same', activation='relu', name=f'conv1_{block_index}_1_f{factor}')(input_img)
    conv2_1_1 = Conv1D(int(filters_sizes[1] * factor), 1, padding='same', activation='relu', name=f'conv2_{block_index}_1_f{factor}')(input_img)
    conv2_1_2 = Conv1D(int(filters_sizes[2] * factor), 5, padding='same', activation='relu', name=f'conv2_{block_index}_2_f{factor}')(conv2_1_1)
    conv3_1_1 = Conv1D(int(filters_sizes[3] * factor), 1, padding='same', activation='relu', name=f'conv3_{block_index}_1_f{factor}')(input_img)
    conv3_1_2 = Conv1D(int(filters_sizes[4] * factor), 3, padding='same', activation='relu', name=f'conv3_{block_index}_2_f{factor}')(conv3_1_1)
    conv4_1_1 = Conv1D(int(filters_sizes[5] * factor), 2, padding='same', activation='relu', name=f'conv4_{block_index}_1_f{factor}')(input_img)
    maxP_3_1 = MaxPooling1D(pool_size=3, strides=1, padding="same", name=f'maxP_3_{block_index}_f{factor}')(conv4_1_1)
    conv4_1_2 = Conv1D(int(filters_sizes[6] * factor), 1, padding='same', activation='relu', name=f'conv4_{block_index}_2_f{factor}')(maxP_3_1)

    result = Concatenate(axis=2)([conv1_1_1, conv2_1_2, conv3_1_2, conv4_1_2])
    return result