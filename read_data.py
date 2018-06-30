from scipy.io import loadmat
import numpy as np
def readmat(file_name):
    data = loadmat(file_name)
    wiener_all = data['Input']
    real_wiener_all = data ['Output']

    return  (*real_wiener_all),(*wiener_all)
if __name__ == "__main__":
    wiener_all,real_wiener_all = readmat('network_input.mat')
    print(len(wiener_all))
    for i in range(len(wiener_all)):
        print(wiener_all[i].shape,real_wiener_all[i].shape)
