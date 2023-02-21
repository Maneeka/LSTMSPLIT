# lstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.layers import Input
from keras.layers import InputLayer
from keras.models import Model
import os
from tensorflow import keras
import time
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy



# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
    # load data
    trainX, trainy, testX, testy = load_dataset()   # trainX : samples, trainy = labels

    # fit, train and save model
    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network = train the model
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)    # trainX : samples, trainy = labels

    # save model
    model.save('trained_model')

    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    print(accuracy * 100.0)


def split_keras_model(model, layer):
    model_f, model_h = Sequential(), Sequential()

    for current_layer in range(0, layer):   # first model has : 0 to layer - 1
        model_f.add(model.layers[current_layer])

    # add input layer
    model_h.add(InputLayer(input_shape=model.layers[layer].input_shape[1:]))

    for current_layer in range(layer, len(model.layers)):   # second model has: layer to end
        model_h.add(model.layers[current_layer])

    return model_f, model_h


def lstmsplit_client(client_model, test_samples, test_lables):
    return client_model.predict(test_samples)


def lstmsplit_edge(edge_model, cut_input):
    return edge_model.predict(cut_input)


def start():
    # load data
    trainX, train_labels, testX, test_labels = load_dataset()

    # load trained model
    reconstructed_model = keras.models.load_model("trained_model")  # accuracy =  91.245 %

    # SPLIT
    client_model, edge_model = split_keras_model(reconstructed_model, 3)

    # predictions and output
    client_output = lstmsplit_client(client_model, testX, test_labels)  # returns predictions array. len = 2947

    # noise addition to client output (cut layer)
    noise = np.random.laplace(0, 0.1, client_output.shape)      # noise = random.normal(0, 0.1, data.shape)     # noise = random.exponential(0.1, data.shape)
    transformed_output = client_output + noise

    final_output = lstmsplit_edge(edge_model, transformed_output)
    print(final_output)

# start()

def latency_plot():
    x = ['1', '2', '3']   # split layers
    y_client = [2.1885, 2.0625, 2.0526]
    z_edge = [0.3652, 0.3514, 0.3389]

    X_axis = np.arange(len(x))

    plt.bar(X_axis - 0.2, y_client, 0.4, label='Client Latency')
    plt.bar(X_axis + 0.2, z_edge, 0.4, label='Edge Latency')

    # Shrink current axis by 20%
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xticks(X_axis, x)
    plt.xlabel("Split Layer")
    plt.ylabel("Time")
    plt.title("SPLIT Model with noise Latency ")
    # plt.legend()
    plt.show()


# latency_plot()
