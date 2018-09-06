import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from sklearn.metrics import f1_score
from data.data_utils import get_data

def keras_model(layer_sizes, dropout, learning_rate):
    '''
    Multilayer perceptron for binary classification.
    :param layer_sizes: list; size for each hidden layer
    :param dropout: float; dropout for hidden layers
    :param learning_rate: float; learning rate for Adam optmizer
    :return: keras model; compiled model for multilayer perceptron
    '''
    model = Sequential()
    for size in layer_sizes:
        model.add(Dense(size))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = Adam(lr = learning_rate)
    model.compile(optimizer = adam, loss = 'binary_crossentropy')
    return model

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    layer_sizes = [100, 150, 50]
    dropout = 0.5
    learning_rate = 0.001
    model = keras_model(layer_sizes, dropout, learning_rate)
    model.fit(X_train, y_train, batch_size = 32, epochs = 200)
    print('f1 score: {}'.format(f1_score(y_test, np.where(model.predict(X_test) > 0.5, 1, 0))))











