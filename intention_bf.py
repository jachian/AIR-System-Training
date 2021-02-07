# -*- coding: utf-8 -*-
"""
Created on Mon May 11 07:21:00 2020

@author: HUB HUB
"""

import pandas as pd
import numpy as np
import re

#imports for keras models
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from keras.layers import LeakyReLU

from keras.models import load_model

# global variables
max_flow_bf = 757
max_flow_dos = 258399
max_flow_ping = 328
max_flow_port = 18291


#####################################################################################################################
#
# This function is used for loading csv documents into Pandas dataframes
#
# Arguments - file_name - The name of the file to load
#           - delimiter - the character delimiter in quites that we want to divide file by.
#
# Returns - data - The pandas dataframe with the content loaded
#
##################################################################################################################
def load_data(file_name, delimit):
    
    #read into a data-frame
    data = pd.read_csv(file_name, delimiter = delimit, low_memory=False)
    
    #convert the datframe to anumpy array and return
    #return data.values
    return data

# required shape [30, 10000, 14]
def classification_rnn(train_x, train_y, validation_x, validation_y, neurons, num_epochs, bch_size, drop, plot_name):
    
    from matplotlib import pyplot
    
    # design network
    model = Sequential()
    model.add(LSTM(units = neurons, input_shape=(train_x.shape[1], train_x.shape[2])))
    # model.add(Dropout(drop))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # fit network
    history = model.fit(train_x, train_y, epochs = num_epochs, batch_size = bch_size, 
                        validation_data=(validation_x, validation_y), verbose=2, shuffle=False)
    
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    
    pyplot.title(plot_name)
    pyplot.ylabel('Loss')
    pyplot.xlabel('epochs')    
    
    pyplot.legend()
    pyplot.savefig(plot_name)
    pyplot.show()
    pyplot.clf()
    
    # model.save(plot_name[:6] + ".h5")
    return history



# required shape [30, 10000, 14]
def prediction_rnn(train_x, train_y, validation_x, validation_y, neurons, num_epochs, bch_size, plot_name):
    
    from matplotlib import pyplot
    
    output_len = train_x.shape[1]
    
    # design network
    model = Sequential()
    model.add(LSTM(units = neurons, activation = 'relu', input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(RepeatVector(output_len))
    model.add(LSTM(units = neurons, return_sequences=True))
    model.add(LeakyReLU(alpha=0.1))    
    model.add(TimeDistributed(Dense(100)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(13))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    
    # fit network
    history = model.fit(train_x, train_y, epochs = num_epochs, batch_size = bch_size, 
                        validation_data=(validation_x, validation_y), verbose=2, shuffle=False)
    

    
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    
    pyplot.title(plot_name)
    pyplot.ylabel('Loss')
    pyplot.xlabel('epochs')    
    
    pyplot.legend()
    pyplot.savefig(plot_name)
    pyplot.show()
    pyplot.clf()

    model.save(plot_name[:6] + ".h5")
    return history


def plot_batch_sizes():
    
    data = more_training_data(brute_force_data())
    
    batch_size = 2
    
    while batch_size < 40:
        model = classification_rnn(np.array(data[0][0]), np.array(data[0][1]), 
                                   np.array(data[1][0]), np.array(data[1][1]), 
                                   40, 200, batch_size, 0, "BrutForce_batch_size_" +str(batch_size)+ ".png")
        
        batch_size = batch_size + 2
    

    

# required shape [30, 10000, 14]
def classification_rnn1(train_x, train_y, neurons, num_epochs, bch_size, drop, plot_name):
    
    from matplotlib import pyplot
    
    # design network
    model = Sequential()
    model.add(LSTM(units = neurons, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dropout(drop))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # fit network
    history = model.fit(train_x, train_y, epochs = num_epochs, batch_size = bch_size, 
                        verbose=2, shuffle=False)
    
    # plot history
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    
    # pyplot.title(plot_name)
    # pyplot.ylabel('Loss')
    # pyplot.xlabel('epochs')    
    
    # pyplot.legend()
    # pyplot.savefig(plot_name)
    # pyplot.show()
    # pyplot.clf()
    
    model.save(plot_name + ".h5")
    return history

def train_classification():
    data = more_training_data(brute_force_data())
    neurons = 10
    
    while neurons <= 150:
        model = classification_rnn1(np.array(data[0][0]), np.array(data[0][1]),
                                    neurons, 200, 10, 0, "BrutForce_B-10_N-" +str(neurons))
        
        neurons = neurons + 10
    




def train_prediction():
    data = get_prediction_data(more_training_data(brute_force_data()))
    
    model = prediction_rnn(np.array(data[0][0]), np.array(data[0][1]), 
                                   np.array(data[1][0]), np.array(data[1][1]), 
                                   40, 50, 10, "P_Brut.png")


def generate_learning_curves():
        data = more_training_data(brute_force_data())
        
        for i in range(0, 5):
            model = classification_rnn(np.array(data[0][0]), np.array(data[0][1]), 
                                       np.array(data[1][0]), np.array(data[1][1]), 
                                       40, 400, 10, 0.2, "BrutForceCurves" +str(i)+ ".png")

    


def confusion_matrix():
    
    data = brute_force_data()
    
    model = load_model('Brut-0.h5')
    
    predicted = []
    test_data = data[2][0]
    print("length of test_data == " +str(test_data))
    
    for i in range(0, len(test_data)):
        prediction = model.predict(np.array([test_data[i]]), verbose = 2)[0][0]
        predicted.append(prediction)
    
    print("predeicted list == ")
    print(predicted)
    print("length of predicted == " +str(predicted))
    
    expected = data[2][1]
    print("Expected list == ")
    print(expected)
    print("length of Expected == " +str(len(expected)))
    
    predicted = to_binary(predicted)

    print("predeicted list == ")
    print(predicted)
    print("length of predicted == " +str(predicted))
    
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(expected,predicted)
    print(matrix)
    
    return matrix


def to_binary(float_list):
    
    binary_list = []
    
    for i in range(0, len(float_list)):
        if(float_list[i] >= 0.5):
            binary_list.append(1)
        else:
            binary_list.append(0)
    
    return binary_list

        

def classification_testing():
    
    #get briteforce data
    # data = more_training_data(brute_force_data()) 
    data = more_training_data(normal_data())

    model = classification_rnn(np.array(data[0][0]), np.array(data[0][1]), 
                               np.array(data[1][0]), np.array(data[1][1]), 
                               35, 450, 10, 
                               "plot_Normal_n_35_b_10.png")
    
    model = classification_rnn(np.array(data[0][0]), np.array(data[0][1]), 
                               np.array(data[1][0]), np.array(data[1][1]), 
                               40, 450, 10, 
                               "plot_Normal_n_40_b_10.png")
    
    model = classification_rnn(np.array(data[0][0]), np.array(data[0][1]), 
                               np.array(data[1][0]), np.array(data[1][1]), 
                               45, 450, 10, 
                               "plot_Normal_n_45_b_10.png")
    
    """
    #run through training sequences
    neurons = 30
    epochs = 450
    batch_size = 5
    
    while neurons <= 110:
        while batch_size <= 50:
            
            model = classification_rnn(np.array(data[0][0]), np.array(data[0][1]), 
                                       np.array(data[1][0]), np.array(data[1][1]), 
                                       neurons, epochs, batch_size, 
                                       "plot_BF_n" +str(neurons)+ "ep_" +str(neurons)+ "bs_" +str(batch_size)+ ".png")
            
            batch_size = batch_size + 5
        # end while
        neurons = neurons + 5
    # end while
    print("Done brute force>>>>>>")
    
    #get briteforce data
    data = more_training_data(dos_data())
    
    #run through training sequences
    neurons = 30
    epochs = 450
    batch_size = 5
    
    while neurons <= 110:
        while batch_size <= 50:
            
            model = classification_rnn(np.array(data[0][0]), np.array(data[0][1]), 
                                       np.array(data[1][0]), np.array(data[1][1]), 
                                       neurons, epochs, batch_size,
                                       "plot_DOS_n" +str(neurons)+ "ep_" +str(neurons)+ "bs_" +str(batch_size)+ ".png")
            
            batch_size = batch_size + 5
        # end while
        neurons = neurons + 5
    # end while 
    print("Done DOS>>>>>>")           
            
    #get briteforce data
    data = more_training_data(port_scan_data())
    
    #run through training sequences
    neurons = 30
    epochs = 450
    batch_size = 5
    
    while neurons <= 110:
        while batch_size <= 50:
            
            model = classification_rnn(np.array(data[0][0]), np.array(data[0][1]), 
                                       np.array(data[1][0]), np.array(data[1][1]), 
                                       neurons, epochs, batch_size,
                                       "plot_PorSc_n" +str(neurons)+ "ep_" +str(neurons)+ "bs_" +str(batch_size)+ ".png")
            
            batch_size = batch_size + 5
        # end while
        neurons = neurons + 5
    # end while
    print("Done Port Scan>>>>>>")        
    
    #get briteforce data
    data = more_training_data(ping_scan_data())
    
    #run through training sequences
    neurons = 30
    epochs = 450
    batch_size = 5
    
    while neurons <= 110:
        while batch_size <= 50:
            
            model = classification_rnn(np.array(data[0][0]), np.array(data[0][1]), 
                                       np.array(data[1][0]), np.array(data[1][1]), 
                                       neurons, epochs, batch_size,
                                       "plot_PingSc_n" +str(neurons)+ "ep_" +str(neurons)+ "bs_" +str(batch_size)+ ".png")
            
            batch_size = batch_size + 5
        # end while
        neurons = neurons + 5
    # end while
    print("Done Ping Scan>>>>>>")

    #get briteforce data
    data = more_training_data(normal_data())
    
    #run through training sequences
    neurons = 30
    epochs = 450
    batch_size = 5
    
    while neurons <= 110:
        while batch_size <= 50:
            
            model = classification_rnn(np.array(data[0][0]), np.array(data[0][1]), 
                                       np.array(data[1][0]), np.array(data[1][1]), 
                                       neurons, epochs, batch_size,
                                       "plot_Nor_n" +str(neurons)+ "ep_" +str(neurons)+ "bs_" +str(batch_size)+ ".png")
            
            batch_size = batch_size + 5
        # end while
        neurons = neurons + 5
    # end while 

    #get briteforce data
    data = more_training_data(suspicious_data())
    
    #run through training sequences
    neurons = 30
    epochs = 450
    batch_size = 5
    
    while neurons <= 110:
        while batch_size <= 50:
            
            model = classification_rnn(np.array(data[0][0]), np.array(data[0][1]), 
                                       np.array(data[1][0]), np.array(data[1][1]), 
                                       neurons, epochs, batch_size,
                                       "plot_Sus_n" +str(neurons)+ "ep_" +str(neurons)+ "bs_" +str(batch_size)+ ".png")
            
            batch_size = batch_size + 5
        # end while
        neurons = neurons + 5
    # end while
    print("Done brute force>>>>>>")
    """
    #end block commenting
    

    


def train_classification_rnns():
    
    #get bruteforce data
    bf_data = brute_force_data()
    
    # now train calssification rnn
    bf_rnn = classification_rnn(np.array(bf_data[0][0]), np.array(bf_data[0][1]),
                                np.array(bf_data[1][0]), np.array(bf_data[1][1]))
    
    return bf_rnn

    


#####################################################################################################################
#
# This function is used to determine length of the longest flow sequence in a file
#
# Arguments - file_name - The name of the file to load ( BruteForceAll, etc.)
#
# Returns - tuple (longest length, name of file with longest sequence)
#
###############################################################################################################
def longest_sequence(file_name):
    
    files_list = load_data(file_name, ",")
    
    files = files_list["Files"]
    
    longest = 0
    longest_file = " "
    
    for i in range(0, len(files)):
        flow = load_data(files[i], ",")
        
        if(len(flow) > longest):
            longest = len(flow)
            longest_file = files[i]
        
    # end for
    
    return (longest, longest_file)

def longest_flow():
    
    # find the longest sequence from each flow type
    l1 = longest_sequence("BruteForceAll.csv")
    l2 = longest_sequence("DOSAll.csv")
    l3 = longest_sequence("PingScansAll.csv")
    l4 = longest_sequence("PortScansAll.csv")
    
    lyst = [l1[0], l2[0], l3[0], l4[0]]
    print(str(lyst))
    
    return max(lyst)
    

#####################################################################################################################
#
# This function is used to determine length of the longest flow sequence in a file
#
# Arguments - file_name - The name of the file to load ( BruteForceAll, etc.)
#
# Returns - tuple (longest length, name of file with longest sequence)
#
##############################################################################################################
def brute_force_data():
    
    # load the train validation and test lists
    train = load_data("BruteForce_train.csv", ",")["Files"]
    validation = load_data("BruteForce_val.csv", ",")["Files"]
    test = load_data("BruteForce_test.csv", ",")["Files"]
    
    #set up lists for the x data
    train_x = []
    validation_x = []
    test_x = []
    
    # set up the y data for classification
    train_y = []
    validation_y = []
    test_y = []
    
    # the vector pad
    vector_pad = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    #Fill in the y values  #########################################
    for i in range(0, len(train)):
        file_name = train[i]
        
        if(file_name[0 : 10] == 'bruteForce'):
            train_y.append(1)
        else:
            train_y.append(0)
    
    #now we pre-pad
    # train_y = pre_pad_list(train_y, 0)
    

    for i in range(0, len(validation)):
        file_name = validation[i]
        
        if(file_name[0 : 10] == 'bruteForce'):
            validation_y.append(1)
        else:
            validation_y.append(0)

    #now we pre-pad
    # validation_y = pre_pad_list(validation_y, 0)

    for i in range(0, len(test)):
        file_name = test[i]
        
        if(file_name[0 : 10] == 'bruteForce'):
            test_y.append(1)
        else:
            test_y.append(0)
        
    #now we pre-pad
    # test_y = pre_pad_list(test_y, 0)
    
    
    # now we get the flows for x values #################################
    for i in range(0, len(train)):
        #load the flow
        flow = load_data(train[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        train_x.append(pre_pad_list(flow.values, vector_pad))

    # now we get the flows for x values #################################
    for i in range(0, len(validation)):
        #load the flow
        flow = load_data(validation[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        validation_x.append(pre_pad_list(flow.values, vector_pad))

    # now we get the flows for x values #################################
    for i in range(0, len(test)):
        #load the flow
        flow = load_data(test[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        test_x.append(pre_pad_list(flow.values, vector_pad))
    
    #check to see if the number of entries are correct
    print("train == " +str(len(train))+ " train_x == " +str(len(train_x)))
    print("val == " +str(len(validation))+ " validation_x == " +str(len(validation_x)))
    print("test == " +str(len(test))+ " test_x == " +str(len(test_x)))

    #check to see if the x and y length match
    print(" train_x == " +str(len(train_x))+ " train_y == " +str(len(train_y)))
    print("validation_x == " +str(len(validation_x))+ " val_y == " +str(len(validation_y)))
    print(" test_x == " +str(len(test_x))+ " test_y == " +str(len(test_y)))
    
    # we want to return the list
    return [(train_x, train_y), (validation_x, validation_y), (test_x, test_y)]



    

#####################################################################################################################
#
# This function is used to determine length of the longest flow sequence in a file
#
# Arguments - file_name - The name of the file to load ( BruteForceAll, etc.)
#
# Returns - tuple (longest length, name of file with longest sequence)
#
##############################################################################################################
def dos_data():
    
    # load the train validation and test lists
    train = load_data("DOS_train.csv", ",")["Files"]
    validation = load_data("DOS_val.csv", ",")["Files"]
    test = load_data("DOS_test.csv", ",")["Files"]
    
    #set up lists for the x data
    train_x = []
    validation_x = []
    test_x = []
    
    # set up the y data for classification
    train_y = []
    validation_y = []
    test_y = []
    
    # the vector pad
    vector_pad = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    #Fill in the y values  #########################################
    for i in range(0, len(train)):
        file_name = train[i]
        
        if(file_name[0 : 3] == 'dos'):
            train_y.append(1)
        else:
            train_y.append(0)
    
    #now we pre-pad
    # train_y = pre_pad_list(train_y, 0)
    

    for i in range(0, len(validation)):
        file_name = validation[i]
        
        if(file_name[0 : 3] == 'dos'):
            validation_y.append(1)
        else:
            validation_y.append(0)

    #now we pre-pad
    # validation_y = pre_pad_list(validation_y, 0)

    for i in range(0, len(test)):
        file_name = test[i]
        
        if(file_name[0 : 3] == 'dos'):
            test_y.append(1)
        else:
            test_y.append(0)
        
    #now we pre-pad
    # test_y = pre_pad_list(test_y, 0)
    
    
    # now we get the flows for x values #################################
    for i in range(0, len(train)):
        #load the flow
        flow = load_data(train[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        train_x.append(pre_pad_list(flow.values, vector_pad))

    # now we get the flows for x values #################################
    for i in range(0, len(validation)):
        #load the flow
        flow = load_data(validation[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        validation_x.append(pre_pad_list(flow.values, vector_pad))

    # now we get the flows for x values #################################
    for i in range(0, len(test)):
        #load the flow
        flow = load_data(test[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        test_x.append(pre_pad_list(flow.values, vector_pad))
    
    #check to see if the number of entries are correct
    print("train == " +str(len(train))+ " train_x == " +str(len(train_x)))
    print("val == " +str(len(validation))+ " validation_x == " +str(len(validation_x)))
    print("test == " +str(len(test))+ " test_x == " +str(len(test_x)))

    #check to see if the x and y length match
    print(" train_x == " +str(len(train_x))+ " train_y == " +str(len(train_y)))
    print("validation_x == " +str(len(validation_x))+ " val_y == " +str(len(validation_y)))
    print(" test_x == " +str(len(test_x))+ " test_y == " +str(len(test_y)))
    
    # we want to return the list
    return [(train_x, train_y), (validation_x, validation_y), (test_x, test_y)]


#####################################################################################################################
#
# This function is used to determine length of the longest flow sequence in a file
#
# Arguments - file_name - The name of the file to load ( BruteForceAll, etc.)
#
# Returns - tuple (longest length, name of file with longest sequence)
#
##############################################################################################################
def port_scan_data():
    
    # load the train validation and test lists
    train = load_data("Port_train.csv", ",")["Files"]
    validation = load_data("Port_val.csv", ",")["Files"]
    test = load_data("Port_test.csv", ",")["Files"]
    
    #set up lists for the x data
    train_x = []
    validation_x = []
    test_x = []
    
    # set up the y data for classification
    train_y = []
    validation_y = []
    test_y = []
    
    # the vector pad
    vector_pad = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    #Fill in the y values  #########################################
    for i in range(0, len(train)):
        file_name = train[i]
        
        if(file_name[0 : 8] == 'portScan'):
            train_y.append(1)
        else:
            train_y.append(0)
    
    #now we pre-pad
    # train_y = pre_pad_list(train_y, 0)
    

    for i in range(0, len(validation)):
        file_name = validation[i]
        
        if(file_name[0 : 8] == 'portScan'):
            validation_y.append(1)
        else:
            validation_y.append(0)

    #now we pre-pad
    # validation_y = pre_pad_list(validation_y, 0)

    for i in range(0, len(test)):
        file_name = test[i]
        
        if(file_name[0 : 8] == 'portScan'):
            test_y.append(1)
        else:
            test_y.append(0)
        
    #now we pre-pad
    # test_y = pre_pad_list(test_y, 0)
    
    
    # now we get the flows for x values #################################
    for i in range(0, len(train)):
        #load the flow
        flow = load_data(train[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        train_x.append(pre_pad_list(flow.values, vector_pad))

    # now we get the flows for x values #################################
    for i in range(0, len(validation)):
        #load the flow
        flow = load_data(validation[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        validation_x.append(pre_pad_list(flow.values, vector_pad))

    # now we get the flows for x values #################################
    for i in range(0, len(test)):
        #load the flow
        flow = load_data(test[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        test_x.append(pre_pad_list(flow.values, vector_pad))
    
    #check to see if the number of entries are correct
    print("train == " +str(len(train))+ " train_x == " +str(len(train_x)))
    print("val == " +str(len(validation))+ " validation_x == " +str(len(validation_x)))
    print("test == " +str(len(test))+ " test_x == " +str(len(test_x)))

    #check to see if the x and y length match
    print(" train_x == " +str(len(train_x))+ " train_y == " +str(len(train_y)))
    print("validation_x == " +str(len(validation_x))+ " val_y == " +str(len(validation_y)))
    print(" test_x == " +str(len(test_x))+ " test_y == " +str(len(test_y)))
    
    # we want to return the list
    return [(train_x, train_y), (validation_x, validation_y), (test_x, test_y)]


#####################################################################################################################
#
# This function is used to determine length of the longest flow sequence in a file
#
# Arguments - file_name - The name of the file to load ( BruteForceAll, etc.)
#
# Returns - tuple (longest length, name of file with longest sequence)
#
##############################################################################################################
def ping_scan_data():
    
    # load the train validation and test lists
    train = load_data("Ping_train.csv", ",")["Files"]
    validation = load_data("Ping_val.csv", ",")["Files"]
    test = load_data("Ping_test.csv", ",")["Files"]
    
    #set up lists for the x data
    train_x = []
    validation_x = []
    test_x = []
    
    # set up the y data for classification
    train_y = []
    validation_y = []
    test_y = []
    
    # the vector pad
    vector_pad = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    #Fill in the y values  #########################################
    for i in range(0, len(train)):
        file_name = train[i]
        
        if(file_name[0 : 8] == 'pingScan'):
            train_y.append(1)
        else:
            train_y.append(0)
    
    #now we pre-pad
    # train_y = pre_pad_list(train_y, 0)
    

    for i in range(0, len(validation)):
        file_name = validation[i]
        
        if(file_name[0 : 8] == 'pingScan'):
            validation_y.append(1)
        else:
            validation_y.append(0)

    #now we pre-pad
    # validation_y = pre_pad_list(validation_y, 0)

    for i in range(0, len(test)):
        file_name = test[i]
        
        if(file_name[0 : 8] == 'pingScan'):
            test_y.append(1)
        else:
            test_y.append(0)
        
    #now we pre-pad
    # test_y = pre_pad_list(test_y, 0)
    
    
    # now we get the flows for x values #################################
    for i in range(0, len(train)):
        #load the flow
        flow = load_data(train[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        train_x.append(pre_pad_list(flow.values, vector_pad))

    # now we get the flows for x values #################################
    for i in range(0, len(validation)):
        #load the flow
        flow = load_data(validation[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        validation_x.append(pre_pad_list(flow.values, vector_pad))

    # now we get the flows for x values #################################
    for i in range(0, len(test)):
        #load the flow
        flow = load_data(test[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        test_x.append(pre_pad_list(flow.values, vector_pad))
    
    #check to see if the number of entries are correct
    print("train == " +str(len(train))+ " train_x == " +str(len(train_x)))
    print("val == " +str(len(validation))+ " validation_x == " +str(len(validation_x)))
    print("test == " +str(len(test))+ " test_x == " +str(len(test_x)))

    #check to see if the x and y length match
    print(" train_x == " +str(len(train_x))+ " train_y == " +str(len(train_y)))
    print("validation_x == " +str(len(validation_x))+ " val_y == " +str(len(validation_y)))
    print(" test_x == " +str(len(test_x))+ " test_y == " +str(len(test_y)))
    
    # we want to return the list
    return [(train_x, train_y), (validation_x, validation_y), (test_x, test_y)]

#####################################################################################################################
#
# This function is used to determine length of the longest flow sequence in a file
#
# Arguments - file_name - The name of the file to load ( BruteForceAll, etc.)
#
# Returns - tuple (longest length, name of file with longest sequence)
#
##############################################################################################################
def normal_data():
    
    # load the train validation and test lists
    train = load_data("Normal_train.csv", ",")["Files"]
    validation = load_data("Normal_val.csv", ",")["Files"]
    test = load_data("Normal_test.csv", ",")["Files"]
    
    #set up lists for the x data
    train_x = []
    validation_x = []
    test_x = []
    
    # set up the y data for classification
    train_y = []
    validation_y = []
    test_y = []
    
    # the vector pad
    vector_pad = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    #Fill in the y values  #########################################
    for i in range(0, len(train)):
        file_name = train[i]
        
        if(file_name[0 : 6] == 'normal'):
            train_y.append(1)
        else:
            train_y.append(0)
    
    #now we pre-pad
    # train_y = pre_pad_list(train_y, 0)
    

    for i in range(0, len(validation)):
        file_name = validation[i]
        
        if(file_name[0 : 6] == 'normal'):
            validation_y.append(1)
        else:
            validation_y.append(0)

    #now we pre-pad
    # validation_y = pre_pad_list(validation_y, 0)

    for i in range(0, len(test)):
        file_name = test[i]
        
        if(file_name[0 : 6] == 'normal'):
            test_y.append(1)
        else:
            test_y.append(0)
        
    #now we pre-pad
    # test_y = pre_pad_list(test_y, 0)
    
    
    # now we get the flows for x values #################################
    for i in range(0, len(train)):
        #load the flow
        flow = load_data(train[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        train_x.append(pre_pad_list(flow.values, vector_pad))

    # now we get the flows for x values #################################
    for i in range(0, len(validation)):
        #load the flow
        flow = load_data(validation[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        validation_x.append(pre_pad_list(flow.values, vector_pad))

    # now we get the flows for x values #################################
    for i in range(0, len(test)):
        #load the flow
        flow = load_data(test[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        test_x.append(pre_pad_list(flow.values, vector_pad))
    
    #check to see if the number of entries are correct
    print("train == " +str(len(train))+ " train_x == " +str(len(train_x)))
    print("val == " +str(len(validation))+ " validation_x == " +str(len(validation_x)))
    print("test == " +str(len(test))+ " test_x == " +str(len(test_x)))

    #check to see if the x and y length match
    print(" train_x == " +str(len(train_x))+ " train_y == " +str(len(train_y)))
    print("validation_x == " +str(len(validation_x))+ " val_y == " +str(len(validation_y)))
    print(" test_x == " +str(len(test_x))+ " test_y == " +str(len(test_y)))
    
    # we want to return the list
    return [(train_x, train_y), (validation_x, validation_y), (test_x, test_y)]


#####################################################################################################################
#
# This function is used to determine length of the longest flow sequence in a file
#
# Arguments - file_name - The name of the file to load ( BruteForceAll, etc.)
#
# Returns - tuple (longest length, name of file with longest sequence)
#
##############################################################################################################
def suspicious_data():
    
    # load the train validation and test lists
    train = load_data("Susp_train.csv", ",")["Files"]
    validation = load_data("Susp_val.csv", ",")["Files"]
    test = load_data("Susp_test.csv", ",")["Files"]
    
    #set up lists for the x data
    train_x = []
    validation_x = []
    test_x = []
    
    # set up the y data for classification
    train_y = []
    validation_y = []
    test_y = []
    
    # the vector pad
    vector_pad = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    #Fill in the y values  #########################################
    for i in range(0, len(train)):
        file_name = train[i]
        
        if(file_name[0 : 10] == 'suspicious'):
            train_y.append(1)
        else:
            train_y.append(0)
    
    #now we pre-pad
    # train_y = pre_pad_list(train_y, 0)
    

    for i in range(0, len(validation)):
        file_name = validation[i]
        
        if(file_name[0 : 10] == 'suspicious'):
            validation_y.append(1)
        else:
            validation_y.append(0)

    #now we pre-pad
    # validation_y = pre_pad_list(validation_y, 0)

    for i in range(0, len(test)):
        file_name = test[i]
        
        if(file_name[0 : 10] == 'suspicious'):
            test_y.append(1)
        else:
            test_y.append(0)
        
    #now we pre-pad
    # test_y = pre_pad_list(test_y, 0)
    
    
    # now we get the flows for x values #################################
    for i in range(0, len(train)):
        #load the flow
        flow = load_data(train[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        train_x.append(pre_pad_list(flow.values, vector_pad))

    # now we get the flows for x values #################################
    for i in range(0, len(validation)):
        #load the flow
        flow = load_data(validation[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        validation_x.append(pre_pad_list(flow.values, vector_pad))

    # now we get the flows for x values #################################
    for i in range(0, len(test)):
        #load the flow
        flow = load_data(test[i], ",")
        
        #delete the first column of data
        del flow[flow.columns[0]]
        
        #enter the flow into the train_x list
        test_x.append(pre_pad_list(flow.values, vector_pad))
    
    #check to see if the number of entries are correct
    print("train == " +str(len(train))+ " train_x == " +str(len(train_x)))
    print("val == " +str(len(validation))+ " validation_x == " +str(len(validation_x)))
    print("test == " +str(len(test))+ " test_x == " +str(len(test_x)))

    #check to see if the x and y length match
    print(" train_x == " +str(len(train_x))+ " train_y == " +str(len(train_y)))
    print("validation_x == " +str(len(validation_x))+ " val_y == " +str(len(validation_y)))
    print(" test_x == " +str(len(test_x))+ " test_y == " +str(len(test_y)))
    
    # we want to return the list
    return [(train_x, train_y), (validation_x, validation_y), (test_x, test_y)]



def pre_pad_list(array, pad):
    
    if len(array) > 10000:
        array = array[0 : 10000]
        
    pad_length = 10000 - len(array)
    
    for i in range(0, pad_length):
        array = np.insert(array, 0, pad, axis = 0)
    
    return array



def more_training_data(data_list):
    
    train_x = data_list[0][0]
    train_y = data_list[0][1]
    
    val_x = data_list[1][0]
    val_y = data_list[1][1]
    
    for i in range(0, len(data_list[1][0])):
        train_x.append(val_x[i])
        train_y.append(val_y[i])
    
    
    #check to see if the x and y length match
    print(" train_x == " +str(len(train_x))+ " train_y == " +str(len(train_y)))
    print("validation_x == " +str(len(data_list[2][0]))+ " val_y == " +str(len(data_list[2][1])))
    
    return [(train_x, train_y), data_list[2]]

# brute_force_data()


def get_prediction_data(data):
    
    #assign the x data
    train_x = data[0][0] # this is a list
    
    #create the y taining data
    train_y = []
    
    for i in range(0, len(train_x)):
        
        item = data_shift(train_x[i], 10) #shift the data item by 10 time units
        
        #append item to new training set
        train_y.append(item)
    
    # deal with the test data as well
    test_x = data[1][0]
    
    test_y = []
    
    for i in range(0, len(test_x)):
        
        item = data_shift(test_x[i], 10) #shift the data item by 10 time units
        
        #append item to new training set
        test_y.append(item)
    
    #check to see if the x and y length match
    print(" train_x == " +str(len(train_x))+ " train_y == " +str(len(train_y)))
    print("test_x == " +str(len(test_x))+ " test_y == " +str(len(test_y)))
    
    # we return the new training and test sets
    return [(train_x, train_y), (test_x, test_y)]


def data_shift(data_list, time_steps):
    
    new_list = data_list # this is anumpy array
    last_entry = data_list[len(data_list) - 1]
    
    for i in range(0, time_steps):
        new_list = np.append(new_list, [last_entry], axis = 0)
    
    #now we return our shifted list
    
    print(" shifted list length == " +str(len(new_list[time_steps : ])))
    
    return new_list[time_steps : ]
    
    

# generate_learning_curves()

train_classification()
# train_prediction()
    
        
# plot_batch_sizes()        
    
    



