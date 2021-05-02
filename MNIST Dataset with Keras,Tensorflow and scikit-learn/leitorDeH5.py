import h5py

filename = "CNN_tf.keras_mnist.h5"

h5 = h5py.File(filename,'r')
enter = input("Aperte [ENTER] para continuar.")

h5.close()