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
from keras.models import Model
from matplotlib import pyplot
import os
from tensorflow import keras

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

    # split was here before i moved it


    # old code: summarize results
    # summarize_results(scores)

# run the experiment
# run_experiment()

def split_keras_model(model, index):
    # Input:
    # model: A pre-trained Keras Sequential model
    # index: The index of the layer where we want to split the model

    # Output:
    # model1: From layer 0 to index
    # model2: From index+1 layer to the output of the original model
    # The index layer will be the last layer of the model_1 and the same shape of that layer will be the input layer of the model_2

    # Creating the first part...
    # Get the input layer shape
    layer_input_1 = Input(model.layers[0].input_shape[1:])
    # Initialize the model with the input layer
    x = layer_input_1
    # Foreach layer: connect it to the new model
    for layer in model.layers[1:index]:
        x = layer(x)
    # Create the model instance
    model1 = Model(inputs=layer_input_1, outputs=x)


    # Creating the second part...
    # Get the input shape of desired layer
    input_shape_2 = model.layers[index].get_input_shape_at(0)[1:]
    print("Input shape of model 2: " + str(input_shape_2))
    # A new input tensor to be able to feed the desired layer
    layer_input_2 = Input(shape=input_shape_2)

    # Create the new nodes for each layer in the path
    x = layer_input_2
    # Foreach layer connect it to the new model
    for layer in model.layers[index:]:
        x = layer(x)

    # create the model
    model2 = Model(inputs=layer_input_2, outputs=x)

    return (model1, model2)


def lstmsplit_client(client_model, test_samples, test_lables):
    predictions = client_model.predict(test_samples)
    print("predictions")
    print(len(predictions))

def lstmsplit_edge(edge_model, cut_input):
    pass


def newStart():
    # load data
    trainX, train_labels, testX, test_labels = load_dataset()

    # load trained model
    reconstructed_model = keras.models.load_model("trained_model")  # accuracy =  91.245 %

    # SPLIT
    client_model, edge_model = split_keras_model(reconstructed_model, 2)

    # predictions and output
    # TODO: exact input and output od client and edge functions
    client_output = lstmsplit_client(client_model, testX, test_labels)  # returns predictions array. len = 2947
    final_output = lstmsplit_edge(edge_model, client_output)


newStart()
