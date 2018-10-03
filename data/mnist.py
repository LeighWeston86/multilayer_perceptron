from tensorflow.examples.tutorials.mnist import input_data

def get_mnist():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    X_train = mnist.train.images
    X_test  = mnist.test.images
    y_train = mnist.train.labels
    y_test  = mnist.test.labels
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_mnist()
    print(X_train.shape, y_train.shape)