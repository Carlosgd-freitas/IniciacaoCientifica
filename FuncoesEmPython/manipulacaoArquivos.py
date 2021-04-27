# Importando biliotecas
import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import pyedflib
from scipy.signal import firwin, filtfilt
import random

# Funções em Python I
def lerArquivoTexto(caminho):
    """
    Função que lê os dados de um arquivo txt. A entrada é o nome com o caminho até o arquivo txt
    e a saída é uma lista onde cada item é uma linha do arquivo txt.
    """
    conteudo = list()

    # O script pode ser executado em outro diretório
    caminhoAtual = os.path.dirname(__file__)
    novoCaminho = os.path.relpath(caminho, caminhoAtual)

    try:
        f = open(novoCaminho, 'r')
    except:
        print('ERRO: O arquivo a ser lido não existe.')
    else:
        conteudo = f.readlines()
        f.close()
    return conteudo

def escreverArquivoTexto(caminho, dados):
    """
    Função que escreve os dados de uma lista de strings em um arquivo txt. A entrada é o nome com o
    caminho até o arquivo txt além da lista de strings. Não há saída. Cada linha do arquivo é um item
    da lista.
    """
    # O script pode ser executado em outro diretório
    caminhoAtual = os.path.dirname(__file__)
    novoCaminho = os.path.relpath(caminho, caminhoAtual)

    try:
        f = open(novoCaminho, 'w')
    except:
        print('ERRO: O arquivo a ser escrito não existe.')
    else:
        i = 0
        while i < len(dados):
            f.write(dados[i])
            i += 1
        f.close()

def lerArquivoCSV(caminho):
    """
    Função que lê os dados de um arquivo csv. A entrada é o nome com o caminho até o arquivo csv e a
    saída é uma lista de listas onde cada item da lista externa é uma linha do arquivo csv, e cada
    item das listas internas são os dados presentes em uma coluna.
    """
    conteudo = list()
    # O script pode ser executado em outro diretório
    caminhoAtual = os.path.dirname(__file__)
    novoCaminho = os.path.relpath(caminho, caminhoAtual)

    try:
        f = open(novoCaminho, 'r')
    except:
        print('ERRO: O arquivo a ser lido não existe.')
    else:
        leitorCSV = csv.reader(f, delimiter=',')
        for row in leitorCSV:
            conteudo.append(row)

        f.close()
    return conteudo

def escreverArquivoCSV(caminho, dados):
    """
    Função que escreve os dados de uma lista de listas de strings em um arquivo csv. A entrada é
    o nome com o caminho até o arquivo csv e a lista de listas de strings e não há saída. Cada
    linha do arquivo é um item da lista externa, e cada coluna é um item das listas internas.
    """
    # O script pode ser executado em outro diretório
    caminhoAtual = os.path.dirname(__file__)
    novoCaminho = os.path.relpath(caminho, caminhoAtual)

    try:
        f = open(novoCaminho, 'w')
    except:
        print('ERRO: O arquivo a ser escrito não existe.')
    else:
        escritorCSV = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        i = 0
        while i < len(dados):
            escritorCSV.writerow(dados[i])
            i += 1
        f.close()

def listarDiretorio(caminho):
    """
    Função que recebe o caminho até uma pasta e retorna uma lista com todos os arquivos dentro
    dessa pasta e suas subpastas.
    """
    conteudo = list()
    # O script pode ser executado em outro diretório
    caminhoAtual = os.path.dirname(__file__)
    novoCaminho = os.path.relpath(caminho, caminhoAtual)

    try:
        conteudo = os.listdir(novoCaminho)
    except:
        print('ERRO: O diretório não foi encontrado.')
        
    return conteudo

# Funções em Python II
def escreverParesAleatoriosTexto(caminho, n):
    """
    Função para salvar em um arquivo txt pares de números aleatórios. A função deve receber
    o valor n (número de pares). A saída é o arquivo txt com os n pares de números aleatórios de
    zero a um. O padrão de escrita para o par é número1;número2. Cada par deve estar em uma linha.
    """
    # O script pode ser executado em outro diretório
    caminhoAtual = os.path.dirname(__file__)
    novoCaminho = os.path.relpath(caminho, caminhoAtual)

    try:
        f = open(novoCaminho, 'w')
    except:
        print('ERRO: O arquivo a ser escrito não existe.')
    else:
        i = 0
        while i < n:
            x = random.random()
            y = random.random()
            f.write(str(x) + ';' + str(y) + '\n')
            i += 1
        return f

def escreverParesAleatoriosCSV(caminho, n):
    """
    Função para salvar em um arquivo csv pares de números aleatórios. A função deve receber
    o valor n (número de pares). A saída é o arquivo csv com os n pares de números aleatórios de
    zero a um. Cada elemento do par fica em uma coluna. Cada par deve estar em uma linha.
    """
    # O script pode ser executado em outro diretório
    caminhoAtual = os.path.dirname(__file__)
    novoCaminho = os.path.relpath(caminho, caminhoAtual)

    try:
        f = open(novoCaminho, 'w')
    except:
        print('ERRO: O arquivo a ser escrito não existe.')
    else:
        i = 0
        while i < n:
            x = random.random()
            y = random.random()
            f.write(str(x) + ',' + str(y) + '\n')
            i += 1
        return f

def escreverParesSequenciaEAleatorioTexto(caminho, n):
    """
    Função para salvar em um arquivo txt pares com um número não aleatório e um número
    aleatório. A função deve receber o valor n (número de pares). A saída é o arquivo txt com
    os n pares, onde o primeiro é referente ao termo da sequência e o segundo um número
    aleatório de zero a um. O padrão de escrita para o par é número1;número2. Cada par deve
    estar em uma linha. O primeiro termo vai de 1 até n.
    """
    # O script pode ser executado em outro diretório
    caminhoAtual = os.path.dirname(__file__)
    novoCaminho = os.path.relpath(caminho, caminhoAtual)

    try:
        f = open(novoCaminho, 'w')
    except:
        print('ERRO: O arquivo a ser escrito não existe.')
    else:
        i = 0
        while i < n:
            x = random.random()
            f.write(str(i+1) + ';' + str(x) + '\n')
            i += 1
        return f

def escreverParesSequenciaEAleatorioCSV(caminho, n):
    """
    Função para salvar em um arquivo csv pares com um número não aleatório e um número
    aleatório. A função deve receber o valor n (número de pares). A saída é o arquivo csv com
    os n pares, onde o primeiro é referente ao termo da sequência e o segundo um número
    aleatório de zero a um. Cada elemento do par fica em uma coluna. Cada par deve estar em uma
    linha. O primeiro termo vai de 1 até n.
    """
    # O script pode ser executado em outro diretório
    caminhoAtual = os.path.dirname(__file__)
    novoCaminho = os.path.relpath(caminho, caminhoAtual)

    try:
        f = open(novoCaminho, 'w')
    except:
        print('ERRO: O arquivo a ser escrito não existe.')
    else:
        i = 0
        while i < n:
            x = random.random()
            f.write(str(i+1) + ',' + str(x) + '\n')
            i += 1
        return f

def plotarParesTexto(caminho , modo):
    """
    Para o parâmetro modo = 0:
    Função que lê de um txt de pares de números aleatórios e plota esses pontos no espaço.
    Cada par está em uma linha, e cada elemento do par está separado por um ponto e vírgula (;).

    Para o parâmetro modo = 1:
    Criar uma função que lê de um txt de pares onde o primeiro número é o termo da sequência e o
    segundo é um número aleatório e plota esses pontos no espaço em forma de um grafo linha. Cada
    par está em uma linha, e cada elemento do par é separado por ponto e vírgula (;).
    """

    pares = lerArquivoTexto(caminho)
    
    # Adicionando Labels
    plt.xlabel('primeiro número do par')
    plt.ylabel('segundo número do par')

    # Os dois números do par são aleatórios de 0 a 1
    if modo == 0:
        # Removendo os caracteres '\n', separando e convertendo os valores
        for i in range(0,len(pares)):
            pares[i] = pares[i].rstrip('\n')
            pares[i] = pares[i].split(';')
            pares[i][0] = float(pares[i][0])
            pares[i][1] = float(pares[i][1])

        # Definindo os valores mínimos e máximos do eixo x, e os valores mínimos e máximos do eixo y
        plt.axis([0, 1, 0, 1])

        # Plotando os pares em um gráfico
        for i in range(0,len(pares)):
            plt.plot(pares[i][0], pares[i][1], 'ro')
    # O primeiro número do par é um termo da sequência de 1 a n, e o segundo é aleatório
    else:
        # Removendo os caracteres '\n', separando e convertendo os valores
        for i in range(0,len(pares)):
            pares[i] = pares[i].rstrip('\n')
            pares[i] = pares[i].split(';')
            pares[i][0] = float(pares[i][0])
            pares[i][1] = float(pares[i][1])

        # Definindo os valores mínimos e máximos do eixo x, e os valores mínimos e máximos do eixo y
        plt.axis([1, len(pares), 0, 1])

        # Dividindo a lista de pares em duas
        lista1, lista2 = map(list, zip(*pares))

        # Plotando os pares em um gráfico
        plt.plot(lista1, lista2, 'r-')

    plt.show()

def plotarParesCSV(caminho , modo):
    """
    Para o parâmetro modo = 0:
    Função que lê de um csv de pares de números aleatórios e plota esses pontos no espaço. Cada
    par está em uma linha, e cada elemento do par em uma coluna diferente.

    Para o parâmetro modo = 1:
    Criar uma função que lê de um csv de pares onde o primeiro número é o termo da sequência e o
    segundo é um número aleatório e plota esses pontos no espaço em forma de um grafo linha. Cada
    par está em uma linha, e cada elemento do par em uma coluna diferente.
    """

    pares = lerArquivoCSV(caminho)
    
    # Adicionando Labels
    plt.xlabel('primeiro número do par')
    plt.ylabel('segundo número do par')

    # Os dois números do par são aleatórios de 0 a 1
    if modo == 0:
        # Removendo os caracteres '\n', separando e convertendo os valores
        for i in range(0,len(pares)):
            pares[i][0] = float(pares[i][0])
            pares[i][1] = float(pares[i][1])

        # Definindo os valores mínimos e máximos do eixo x, e os valores mínimos e máximos do eixo y
        plt.axis([0, 1, 0, 1])

        # Plotando os pares em um gráfico
        for i in range(0,len(pares)):
            plt.plot(pares[i][0], pares[i][1], 'ro')
    # O primeiro número do par é um termo da sequência de 1 a n, e o segundo é aleatório
    else:
        # Removendo os caracteres '\n', separando e convertendo os valores
        for i in range(0,len(pares)):
            pares[i][0] = float(pares[i][0])
            pares[i][1] = float(pares[i][1])

        # Definindo os valores mínimos e máximos do eixo x, e os valores mínimos e máximos do eixo y
        plt.axis([1, len(pares), 0, 1])

        # Dividindo a lista de pares em duas
        lista1, lista2 = map(list, zip(*pares))

        # Plotando os pares em um gráfico
        plt.plot(lista1, lista2, 'r-')

    plt.show()

# Manipulação de arquivos EDF
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

def plotarEDF(conteudo , modo):
    """
    Para o parâmetro modo = 0:
    Função que recebe uma matriz numpy e plota cada linha da matriz em um subplot diferente, até 5 subplots.

    Para o parâmetro modo = 1:
    Função que recebe uma matriz numpy e plota cada linha da matriz no mesmo subplot, sobrepondo 5 sinais.
    """

    # Plota cada linha da matriz em um subplot diferente, até 5 subplots
    if modo == 0:
        # Plotando o conteúdo em um gráfico
        # Criando 5 subplots verticalmente, com o tamanho total de 8x6 polegadas
        fig, axs = plt.subplots(5 ,figsize=(8, 6))

        axs[0].axis([0, 2000, -50, 50])
        axs[0].plot(conteudo[0][0:1920], 'r-')

        axs[1].axis([0, 2000, -50, 50])
        axs[1].plot(conteudo[1][0:1920], 'g-')

        axs[2].axis([0, 2000, -50, 50])
        axs[2].plot(conteudo[2][0:1920], 'b-')

        axs[3].axis([0, 2000, -50, 50])
        axs[3].plot(conteudo[3][0:1920], 'y-')

        axs[4].axis([0, 2000, -50, 50])
        axs[4].plot(conteudo[4][0:1920], 'm-')
    # Plota cada linha da matriz no mesmo subplot, sobrepondo 5 sinais
    else:
        # Definindo os valores mínimos e máximos do eixo x, e os valores mínimos e máximos do eixo y
        plt.axis([0, 2000, -50, 50])

        # As primeiras 1920 amostras de cada canal serão plotadas para o mesmo gráfico
        plt.plot(conteudo[0][0:1920], 'r-')
        plt.plot(conteudo[1][0:1920], 'g-')
        plt.plot(conteudo[2][0:1920], 'b-')
        plt.plot(conteudo[3][0:1920], 'y-')
        plt.plot(conteudo[4][0:1920], 'm-')

    plt.show()