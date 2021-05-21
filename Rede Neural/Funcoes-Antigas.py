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