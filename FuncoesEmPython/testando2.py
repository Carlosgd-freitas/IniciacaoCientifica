import manipulacaoArquivos
import random

random.seed()
x = 0

while x != -1:
    x = 0
    print('1  - escreverParesAleatoriosTexto')
    print('2  - escreverParesAleatoriosCSV')
    print('3  - escreverParesSequenciaEAleatorioTexto')
    print('4  - escreverParesSequenciaEAleatorioCSV')
    print('5  - plotarParesTexto')
    print('6  - plotarParesCSV')
    print('-1 - SAIR')
    print()

    try:
        x = int(input('Qual função deseja testar? '))
    except:
        print('ERRO: Um erro ocorreu.')
        break
    else:
        if x == 1: # Testando a primeira função (escreverParesAleatoriosTexto)
            print('=== Testando a primeira função ===')
            arqTxt = input('Digite o caminho até o arquivo txt: ')
            n = int(input('Digite o número de pares n: '))
            arquivo = manipulacaoArquivos.escreverParesAleatoriosTexto(arqTxt,n)
            arquivo.close()

            lista = manipulacaoArquivos.lerArquivoTexto(arqTxt)
            if lista:
                print(lista)
        elif x == 2: # Testando a segunda função (escreverParesAleatoriosCSV)
            print('=== Testando a segunda função ===')
            arqCsv = input('Digite o caminho até o arquivo csv: ')
            n = int(input('Digite o número de pares n: '))
            arquivo = manipulacaoArquivos.escreverParesAleatoriosCSV(arqCsv,n)
            arquivo.close()

            lista = manipulacaoArquivos.lerArquivoCSV(arqCsv)
            if lista:
                print(lista)
        elif x == 3: # Testando a terceira função (escreverParesSequenciaEAleatorioTexto)
            print('=== Testando a terceira função ===')
            arqTxt = input('Digite o caminho até o arquivo txt: ')
            n = int(input('Digite o número de pares n: '))
            arquivo = manipulacaoArquivos.escreverParesSequenciaEAleatorioTexto(arqTxt,n)
            arquivo.close()

            lista = manipulacaoArquivos.lerArquivoTexto(arqTxt)
            if lista:
                print(lista)
        elif x == 4: # Testando a quarta função (escreverParesSequenciaEAleatorioCSV)
            print('=== Testando a quarta função ===')
            arqCsv = input('Digite o caminho até o arquivo csv: ')
            n = int(input('Digite o número de pares n: '))
            arquivo = manipulacaoArquivos.escreverParesSequenciaEAleatorioCSV(arqCsv,n)
            arquivo.close()

            lista = manipulacaoArquivos.lerArquivoCSV(arqCsv)
            if lista:
                print(lista)
        elif x == 5: # Testando a quinta função (plotarParesTexto)
            print('=== Testando a quinta função ===')
            arqTxt = input('Digite o caminho até o arquivo txt: ')

            x2 = 0
            while x2 != -1:
                print()
                print('0  - Pares de números aleatórios')
                print('1  - Pares de números em que o primeiro é um termo da sequência de 1 a n, e o segundo é aleatório')
                print('-1 - SAIR')
                print()

                try:
                    x2 = int(input('Qual modo deseja testar? '))
                except:
                    print('ERRO: Um erro ocorreu.')
                    break
                else:
                    if (x2 == 0) or (x2 == 1):
                        manipulacaoArquivos.plotarParesTexto(arqTxt, x2)
                    elif x != -1:
                        print('ERRO: Digite 0 ou 1 para testar uma modo desta função ou -1 para sair desse menu.')
                    print()
        elif x == 6: # Testando a sexta função (plotarParesCSV)
            print('=== Testando a sexta função ===')
            arqTxt = input('Digite o caminho até o arquivo csv: ')

            x2 = 0
            while x2 != -1:
                print()
                print('0  - Pares de números aleatórios')
                print('1  - Pares de números em que o primeiro é um termo da sequência de 1 a n, e o segundo é aleatório')
                print('-1 - SAIR')
                print()

                try:
                    x2 = int(input('Qual modo deseja testar? '))
                except:
                    print('ERRO: Um erro ocorreu.')
                    break
                else:
                    if (x2 == 0) or (x2 == 1):
                        manipulacaoArquivos.plotarParesCSV(arqTxt, x2)
                    elif x != -1:
                        print('ERRO: Digite 0 ou 1 para testar uma modo desta função ou -1 para sair desse menu.')
                    print()
        elif x != -1:
            print('ERRO: Digite um número de 1 a 6 para testar uma função ou -1 para finalizar o programa.')
        print()