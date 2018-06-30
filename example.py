from read_data import readmat
import numpy as np
from numpy import array, floor
from keras import optimizers,losses


if __name__ == "__main__":
    wiener_all,real_wiener_all = array(readmat('network_input.mat'))
    #wiener_all_cv,real_wiener_all_cv = array(readmat ('network_cv.mat'))

    print(wiener_all)
    from keras.models import Sequential
    from keras.layers import LSTM,Dense,TimeDistributed
    dslength = 8000
    print(len(wiener_all))
    X = np.zeros((dslength,257))
    Y = np.zeros((dslength,257))
    print(X.shape)
    wiener_all = array(wiener_all)
    real_wiener_all = array(real_wiener_all)

    for i in range( dslength):
        for j in range(257):
            X[i][j] = wiener_all[i][j]
            Y[i][j] = real_wiener_all[i][j]
    X = np.expand_dims(X,2)
    Y = np.expand_dims(Y,2)

    print(X.shape, Y.shape)

    X = X.reshape((dslength,-1,257))
    Y = Y.reshape((dslength, -1, 257))

    model = Sequential()
    model.add(LSTM(257,stateful=True,return_sequences=True,batch_input_shape=(10,1,257)))
    #model.add(Dense(257))
    model.compile(loss=losses.mean_squared_error, optimizer='rmsprop', metrics=['accuracy', 'mae'])
    model.fit(x=X, y=Y, epochs=150, shuffle=False,batch_size=10)

    model_yaml = model.to_yaml()
    with open("model2.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("model2.h5")
    print("Saved model to disk")