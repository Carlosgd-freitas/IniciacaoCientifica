import manipulacaoArquivos

x = 0

while x != -1:
    x = 0
    print('1  - lerArquivoEDF')
    print('2  - plotarEDF')
    print('-1 - SAIR')
    print()

    try:
        x = int(input('Qual função deseja testar? '))
    except:
        print('ERRO: Um erro ocorreu.')
        break
    else:
        if x == 1: # Testando a primeira função (lerArquivoEDF)
            print('=== Testando a primeira função ===')
            arqEdf = input('Digite o caminho até o arquivo edf: ')
            conteudo = manipulacaoArquivos.lerArquivoEDF(arqEdf)
        elif x == 2: # Testando a segunda função (plotarEDF)
            print('=== Testando a segunda função ===')
            enter = input('Esta função utilizará o conteúdo lido na função lerArquivoEDF. Aperte [ENTER] para continuar.')

            x2 = 0
            while x2 != -1:
                print()
                print('0  - Cada linha em um subplot diferente')
                print('1  - Cada linha no mesmo subplot (sinais sobrepostos)')
                print('-1 - SAIR')
                print()

                try:
                    x2 = int(input('Qual modo deseja testar? '))
                except:
                    print('ERRO: Um erro ocorreu.')
                    break
                else:
                    if (x2 == 0) or (x2 == 1):
                        manipulacaoArquivos.plotarEDF(conteudo, x2)
                    elif x != -1:
                        print('ERRO: Digite 0 ou 1 para testar uma modo desta função ou -1 para sair desse menu.')
                    print()
        elif x != -1:
            print('ERRO: Digite o número 1 ou 2 para testar uma função ou -1 para finalizar o programa.')
        print()