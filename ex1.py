from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
from keras import losses
from read_data import readmat
from numpy import array,log,exp
import numpy as np
from theano import compile
import matplotlib.pyplot as plt
dslength = 390
import numpy
import os
compile.mode.Mode(linker='py', optimizer=None)
real_wiener_all,wiener_all= array(readmat('network_input.mat'))
#wiener_all_cv, real_wiener_all_cv = array(readmat('network_cv.mat'))

X = np.zeros((dslength,257))
Y = np.zeros((dslength,257))

wiener_all = array(wiener_all)
real_wiener_all = array(real_wiener_all)

for i in range( dslength):
    for j in range(257):
        X1 = wiener_all[i][j]
        #X1 = [X1[k] + 0.01 for k in range(len(X1))]
        X[i][j] = X1

        Y1 = real_wiener_all[i][j]
       # Y1 = [Y1[k] + 0.01 for k in range(len(Y1))]
        Y[i][j] = Y1


O = np.ones((1,1,257))

X = np.expand_dims(X,2)
Y = np.expand_dims(Y,2)
X = X.reshape((len(X),-1,257))
Y = Y.reshape((len(Y), -1,257))
X0 = np.expand_dims(X[0],2)
X0 = X0.reshape((1,1,257))
yaml_file = open('model4.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model4.h5")

print("Loaded model from disk")

loaded_model.compile(loss=losses.mean_squared_error, optimizer='rmsprop', metrics=['accuracy', 'mae'])
score = loaded_model.evaluate(X, Y, verbose=1,batch_size = 5)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


Y1 = loaded_model.predict(X,verbose = 1,batch_size = 5)



for i in range(len(Y)):
    Ys = Y[i][0]
    Ys = np.expand_dims(Ys,2)
    Y1s = Y1[i][0]
    Y1s = np.expand_dims(Y1s,2)
    I = X[i][0]
    out = plt.plot(Ys[0:100],label = 'outputs')
    pre = plt.plot(Y1s[0:100],label = 'predicted')
    inp = plt.plot(I[0:100],label = 'inputs')
    plt.legend()
    plt.show()
