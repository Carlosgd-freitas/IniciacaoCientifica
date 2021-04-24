import manipulacaoArquivos

# Para passar um arquivo:
# teste.txt       -> o arquivo está na mesma pasta que o script
# pasta/teste.txt -> o arquivo está em uma sub-pasta da pasta que contém o script
# ../teste.txt    -> o arquivo está na "pasta pai" da pasta que contém o script
# As funções implementadas não funcionam para arquivos que não estejam em nenhuma das 3 situações acima

x = 0

while x != -1:
    x = 0
    print('1  - lerArquivoTexto')
    print('2  - escreverArquivoTexto')
    print('3  - lerArquivoCSV')
    print('4  - escreverArquivoCSV')
    print('5  - listarDiretorio')
    print('-1 - SAIR')
    print()

    try:
        x = int(input('Qual função deseja testar? '))
    except:
        print('ERRO: Um erro ocorreu.')
        break
    else:
        if x == 1: # Testando a primeira função (lerArquivoTexto)
            print('=== Testando a primeira função ===')
            arqTxt = input('Digite o caminho até o arquivo txt: ')
            lista = manipulacaoArquivos.lerArquivoTexto(arqTxt)
            if lista:
                print(lista)
        elif x == 2: # Testando a segunda função (escreverArquivoTexto)
            print('=== Testando a segunda função ===')
            arqTxt = input('Digite o caminho até o arquivo txt: ')
            manipulacaoArquivos.escreverArquivoTexto(arqTxt, lista)
        elif x == 3: # Testando a terceira função (lerArquivoCSV)
            print('=== Testando a terceira função ===')
            arqCsv = input('Digite o caminho até o arquivo csv: ')
            lista = manipulacaoArquivos.lerArquivoCSV(arqCsv)
            if lista:
                print(lista)
        elif x == 4: # Testando a quarta função (escreverArquivoCSV)
            print('=== Testando a quarta função ===')
            arqCsv = input('Digite o caminho até o arquivo csv: ')
            manipulacaoArquivos.escreverArquivoCSV(arqCsv, lista)
        elif x == 5: # Testando a quinta função (listarDiretorio)
            print('=== Testando a quinta função ===')
            arqTxt = input('Digite o caminho até o diretório: ')
            lista = manipulacaoArquivos.listarDiretorio(arqTxt)
            if lista:
                print(lista)
        elif x != -1:
            print('ERRO: Digite um número de 1 a 5 para testar uma função ou -1 para finalizar o programa.')
        print()