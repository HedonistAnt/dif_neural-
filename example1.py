from read_data import readmat
import numpy as np
from numpy import array, floor,log
from keras import optimizers,losses
from keras.models import Model
from keras.layers import Input, Dense,Activation
from matplotlib.pyplot import plot as plt
from keras.optimizers import SGD, RMSprop,Adagrad,Nadam

if __name__ == "__main__":

    real_wiener_all,wiener_all = readmat('/media/hedonistant/16E47210E471F1FB/CHiME4/network_input.mat')
    #wiener_all_cv,real_wiener_all_cv = array(readmat ('network_cv.mat'))


    from keras.models import Sequential
    from keras.layers import LSTM,Dense
    dslength = int(floor(len(wiener_all)*0.8)) - 2
    val_length = len(wiener_all) - dslength -8
    vector_size = 257
    X = np.zeros((dslength,vector_size))
    Y = np.zeros((dslength,vector_size))

    VX = np.zeros((val_length,vector_size))
    VY = np.zeros((val_length,vector_size))
    wiener_all = array(wiener_all)
    real_wiener_all = array(real_wiener_all)

    for i in range( dslength):

        for j in range(vector_size):

            X1 = wiener_all[i][j]
            X[i][j]=X1
            Y1 = real_wiener_all[i][j]
            Y[i][j]=Y1
    for i in range(dslength,dslength+val_length-1):
        for j in range(vector_size):
            X1 = wiener_all[i][j]
            VX[i-dslength][j] = X1

            Y1 = real_wiener_all[i][j]
            VY[i-dslength][j] = Y1


    X = np.expand_dims(X,2)
    Y = np.expand_dims(Y,2)

    VX = np.expand_dims(VX, 2)
    VY = np.expand_dims(VY, 2)




    X = X.reshape((dslength,-1,vector_size))
    Y = Y.reshape((dslength, -1, vector_size))

    VX = VX.reshape((val_length, -1, vector_size))
    VY = VY.reshape((val_length, -1, vector_size))

    model = Sequential()
    #model.add(Dense(257, activation='linear',batch_input_shape=(5,1,257)))
    model.add(LSTM(vector_size, return_sequences=True,stateful=True,batch_input_shape=(5,1,vector_size)))
    model.add(Dense(257))

    #model.add(Dense(2*vector_size))
    #model.add(Dense(vector_size))
    #model.add(Dense(257))
    #model.add(Dense(257,activation='relu'))
    #model.add(Dense(257))
   # model.add(Dense(2048,activation='relu'))
   # model.add(Dense(257))
    """
    model = Model()
    a = Input(shape=(1,257))
    b = Dense(257)(a)
    model = Model(inputs=a, outputs=b)
    """
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.00)
    model.compile(loss=losses.mean_squared_error, optimizer=rmsprop, metrics=['accuracy', 'mae',])
    history = model.fit(x=X, y=Y, epochs=300, shuffle=False,batch_size=5,validation_data=(VX,VY))

    print(history.history.keys())

    model_yaml = model.to_yaml()
    with open("model4.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("model4.h5")
    print("Saved model to disk")



