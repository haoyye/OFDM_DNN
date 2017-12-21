import os
from Train import train
from Test import test
import numpy as np
class sysconfig(object):
    Pilots = 8        # number of pilots
    with_CP_flag = True 
    SNR = 20
    Clipping = False
    Train_set_path = '../H_dataset/'
    Test_set_path = '../H_dataset/'
    Model_path = '../Models/'
    pred_range = np.arange(16,32)
    learning_rate = 0.001
    learning_rate_decrease_step = 2000     
    

def main():
    config = sysconfig()
    print(config.Train_set_path)
    IS_Training = True
    if IS_Training:
        train(config)
    else:
        test(config)
main()

