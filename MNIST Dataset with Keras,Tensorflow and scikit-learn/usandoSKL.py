from mnist import MNIST
from sklearn.metrics import accuracy_score

print('1 - K-Nearest Neighbors')
print('2 - Random Forest')
print('3 - Linear Support Vector Classification')
opcao = int(input('Qual algoritmo de classificação deseja usar?'))

if opcao == 1:
    from sklearn.neighbors import KNeighborsClassifier

    print("Loading dataset...")
    mndata = MNIST("./data/")
    images, labels = mndata.load_training()

    clf = KNeighborsClassifier()

    # Train on the first 58000 images:
    train_x = images[:58000]
    train_y = labels[:58000]

    print("Train model")
    clf.fit(train_x, train_y)

    # Test on the next 2000 images:
    test_x = images[58000:60000]
    expected = labels[58000:60000].tolist()

    print("Compute predictions")
    predicted = clf.predict(test_x)

    print("Accuracy: ", accuracy_score(expected, predicted))
elif opcao == 2:
    from sklearn.ensemble import RandomForestClassifier

    print("Loading dataset...")
    mndata = MNIST("./data/")
    images, labels = mndata.load_training()

    clf = RandomForestClassifier(n_estimators=100)

    # Train on the first 58000 images:
    train_x = images[:58000]
    train_y = labels[:58000]

    print("Train model")
    clf.fit(train_x, train_y)

    # Test on the next 2000 images:
    test_x = images[58000:60000]
    expected = labels[58000:60000].tolist()

    print("Compute predictions")
    predicted = clf.predict(test_x)

    print("Accuracy: ", accuracy_score(expected, predicted))
elif opcao == 3:
    from sklearn.svm import LinearSVC

    print("Loading dataset...")
    mndata = MNIST("./data/")
    images, labels = mndata.load_training()

    clf = LinearSVC()

    # Train on the first 58000 images:
    train_x = images[:58000]
    train_y = labels[:58000]

    print("Train model")
    clf.fit(train_x, train_y)

    # Test on the next 2000 images:
    test_x = images[58000:60000]
    expected = labels[58000:60000].tolist()

    print("Compute predictions")
    predicted = clf.predict(test_x)

    print("Accuracy: ", accuracy_score(expected, predicted))
else:
    print('ERRO: Digite um número de 1 a 3.')

