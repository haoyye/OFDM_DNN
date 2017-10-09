from __future__ import division
import numpy as np
import scipy.interpolate 
import tensorflow as tf
import math
import os
from utils import *

### =================== Deep Learning Training ================




 


def train(config):    
    K = 64
    CP = K//4
    P = config.Pilots # number of pilot carriers per OFDM block
    allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
    mu = 2
    CP_flag = config.with_CP_flag
    if P<K:
        pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.
        dataCarriers = np.delete(allCarriers, pilotCarriers)
        
    else:   # K = P
        pilotCarriers = allCarriers
        dataCarriers = []


    payloadBits_per_OFDM = K*mu  
    
    SNRdb = config.SNR  # signal to noise-ratio in dB at the receiver 
    Clipping_Flag = config.Clipping 
    
    Pilot_file_name = 'Pilot_'+str(P)
    if os.path.isfile(Pilot_file_name):
        print ('Load Training Pilots txt')
        bits = np.loadtxt(Pilot_file_name, delimiter=',')
    else:
        bits = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
        np.savetxt(Pilot_file_name, bits, delimiter=',')


    pilotValue = Modulation(bits,mu)
    
    CP_flag = config.with_CP_flag
    training_epochs = 20
    display_step = 5
    model_saving_step = 5
    test_step = 1000
    examples_to_show = 10

    # Training parameters
    training_epochs = 20
    batch_size = 256
    display_step = 5
    test_step = 1000
    examples_to_show = 10

    # Network Parameters
    n_hidden_1 = 500
    n_hidden_2 = 250 # 1st layer num features
    n_hidden_3 = 120 # 2nd layer num features
    n_input = 256 # MNIST data input (img shape: 28*28)
    n_output = 16 #4
    # tf Graph input (only pictures)
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])
    
    def encoder(x):
        weights = {                    
            'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1)),
            'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=0.1)),
            'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],stddev=0.1)),
            'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output],stddev=0.1)),            
        }
        biases = {            
            'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1],stddev=0.1)),
            'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2],stddev=0.1)),
            'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3],stddev=0.1)),
	    'encoder_b4': tf.Variable(tf.truncated_normal([n_output],stddev=0.1)),	  
            
        }
        
        # Encoder Hidden layer with sigmoid activation #1
        #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
	layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
	layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4']))
        return layer_4


        
    y_pred = encoder(X)
    # Targets (Labels) are the input data.
    y_true = Y

    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Start Training
    config_GPU = tf.ConfigProto()
    config_GPU.gpu_options.allow_growth = True
    # The H information set
    H_folder_train = config.Train_set_path
    H_folder_test = config.Test_set_path
    train_idx_low = 1
    train_idx_high = 301
    test_idx_low = 301
    test_idx_high = 401
    # Saving Channel conditions to a large matrix
    channel_response_set_train = []
    for train_idx in range(train_idx_low,train_idx_high):
        H_file = H_folder_train + str(train_idx) + '.txt'
	with open(H_file) as f:
            for line in f:
      	        numbers_str = line.split()
                numbers_float = [float(x) for x in numbers_str]
                h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)])+1j*np.asarray(numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                channel_response_set_train.append(h_response)
    channel_response_set_test = []
    for test_idx in range(test_idx_low,test_idx_high):
	H_file = H_folder_test + str(test_idx) + '.txt'
	with open(H_file) as f:
            for line in f:
      	        numbers_str = line.split()
                numbers_float = [float(x) for x in numbers_str]
                h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)])+1j*np.asarray(numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                channel_response_set_test.append(h_response)

    print ('length of training channel response', len(channel_response_set_train), 'length of testing channel response', len(channel_response_set_test))
    saver = tf.train.Saver()
    with tf.Session(config=config_GPU) as sess:
        sess.run(init)
        traing_epochs = 20000
        learning_rate_current = config.learning_rate   
        for epoch in range(traing_epochs):
            print(epoch)
            if epoch > 0 and epoch % config.learning_rate_decrease_step == 0:
                learning_rate_current = learning_rate_current/5                    
            avg_cost = 0.
            total_batch = 50	
            #print (K, P,pilotValue, learning_rate_current)
            for index_m in range(total_batch):
                input_samples = []
                input_labels = []
                for index_k in range(0, 1000):
                    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
		    channel_response = channel_response_set_train[np.random.randint(0,len(channel_response_set_train))]
		    signal_output, para = ofdm_simulate(bits,channel_response,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag)
                    #signal_output, para = ofdm_simulate(bits,channel_response,SNRdb,pilotValue)    
		    input_labels.append(bits[config.pred_range])
		    input_samples.append(signal_output)  
		batch_x = np.asarray(input_samples)
		batch_y = np.asarray(input_labels)
		_,c = sess.run([optimizer,cost], feed_dict={X:batch_x,
                                                                Y:batch_y,
                                                                learning_rate:learning_rate_current})
		avg_cost += c / total_batch
            if epoch % display_step == 0:
                print("Epoch:",'%04d' % (epoch+1), "cost=", \
                       "{:.9f}".format(avg_cost))
                input_samples_test = []
                input_labels_test = []
		test_number = 1000
		# set test channel response for this epoch		    
		if epoch % test_step == 0:
		    print ("Big Test Set ")
		    test_number = 10000
                for i in range(0, test_number):
                    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))			
                    channel_response= channel_response_set_test[np.random.randint(0,len(channel_response_set_test))]
                    signal_output, para = ofdm_simulate(bits,channel_response,SNRdb,mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag)
                    #signal_output, para = ofdm_simulate(bits,channel_response,SNRdb,pilotValue)
                    input_labels_test.append(bits[config.pred_range])
                    input_samples_test.append(signal_output)
                batch_x = np.asarray(input_samples_test)
                batch_y = np.asarray(input_labels_test)
                encode_decode = sess.run(y_pred, feed_dict = {X:batch_x})
                mean_error = tf.reduce_mean(abs(y_pred - batch_y))
                BER = 1-tf.reduce_mean(tf.reduce_mean(tf.to_float(tf.equal(tf.sign(y_pred-0.5), tf.cast(tf.sign(batch_y-0.5),tf.float32))),1))               
                # print("OFDM Detection QAM output number is", n_output, "SNR = ", SNRdb, "Num Pilot", P,"prediction and the mean error on test set are:", mean_error.eval({X:batch_x}), mean_error_rate.eval({X:batch_x}))
                print("BER on test set ", BER.eval({X:batch_x}))                     
                batch_x = np.asarray(input_samples)
                batch_y = np.asarray(input_labels)
                encode_decode = sess.run(y_pred, feed_dict = {X:batch_x})
                mean_error = tf.reduce_mean(abs(y_pred - batch_y))                    
                BER = 1 - tf.reduce_mean(tf.reduce_mean(tf.to_float(tf.equal(tf.sign(y_pred-0.5), tf.cast(tf.sign(batch_y-0.5),tf.float32))),1))
                #print("prediction and the mean error on train set are:", mean_error.eval({X:batch_x}), mean_error_rate.eval({X:batch_x}))
                print("BER on train set", BER.eval({X:batch_x}))
            if epoch % model_saving_step == 0:
                saving_name = config.Model_path + 'SNR_' + str(SNRdb) + '/DetectionModel_SNR_' + str(SNRdb) + '_Pilot_' + str(P) + '_epoch_' + str(epoch)
		saver.save(sess, saving_name)         
        print("optimization finished")


