# Importando biliotecas
import os
import csv
import random
import matplotlib.pyplot as plt
import numpy as np

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