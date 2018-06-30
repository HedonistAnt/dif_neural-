from read_data import readmat
import numpy as np
from numpy import array
from keras import optimizers,losses


if __name__ == "__main__":
    wiener_all,real_wiener_all = array(readmat('network_input.mat'))


    from keras.models import Sequential
    from keras.layers import LSTM,Dense,TimeDistributed

    X = np.zeros((len(wiener_all),257))
    Y = np.zeros((len(wiener_all),257))

    wiener_all = array(wiener_all)
    real_wiener_all = array(real_wiener_all)
    print(len(wiener_all))
    for i in range( len(wiener_all)):
        for j in range(257):
            X[i][j] = wiener_all[i][j]
            Y[i][j] = real_wiener_all[i][j]
    X = np.expand_dims(X,2)
    Y = np.expand_dims(Y,2)



    print(X.shape,Y.shape)
    model = Sequential()
    model.add(LSTM(1,stateful=False,return_sequences=True,input_shape=(257,1)))
    #model.add(Dense(257))
    model.compile(loss=losses.mean_squared_error, optimizer='adam', metrics=['accuracy'])



    model.fit(x=X,y=Y,epochs=100,shuffle=False)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")