import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt

from sympy import primefactors

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """

    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0]= 1.0 #add bias terms


    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)



def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

def random_init(shape):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    np.random.seed(np.prod(shape))

    # Implement random uniform initialization beteen -0.1 and 0.1
    x = np.random.uniform(-0.1, 0.1, size=shape)
    # x[:, 0] = 0.0 # bias terms should be initialized to 0
    return x
    # raise NotImplementedError


def zero_init(shape):
    """
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros(shape)
    # raise NotImplementedError


class NN(object):
    def __init__(self, lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size):
        """
        Initialization
        :param lr: learning rate
        :param n_epoch: number of training epochs
        :param weight_init_fn: weight initialization function
        :param input_size: number of units in the input layer
        :param hidden_size: number of units in the hidden layer
        :param output_size: number of units in the output layer
        """
        self.lr = lr
        self.n_epoch = n_epoch
        self.weight_init_fn = weight_init_fn
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size

        # initialize weights and biases for the models
        # HINT: pay attention to bias here
        self.w1 = self.weight_init_fn((self.n_hidden, self.n_input-1))
        #append bias terms

        self.w1 = np.insert(self.w1, 0, 0, axis=1)

        self.w2 = self.weight_init_fn((self.n_output, self.n_hidden))
        self.w2 = np.insert(self.w2, 0, 0, axis=1)


        # initialize parameters for adagrad
        self.epsilon = 1e-5
        self.grad_sum_w1 = np.zeros_like(self.w1)
        self.grad_sum_w2 = np.zeros_like(self.w2)

        # feel free to add additional attributes
        self.Z1 = None
        self.train_ce = 0
        self.test_ce = 0
        self.train_ce_list = []
        self.test_ce_list = []



def print_weights(nn):
    """
    An example of how to use logging to print out debugging infos.

    Note that we use the debug logging level -- if we use a higher logging
    level, we will log things with the default logging configuration,
    causing potential slowdowns.

    Note that we log NumPy matrices on separate lines -- if we do not do this,
    the arrays will be turned into strings even when our logging is set to
    ignore debug, causing potential massive slowdowns.
    :param nn: your model
    :return:
    """
    logging.debug(f"shape of w1: {nn.w1.shape}")
    logging.debug(nn.w1)
    logging.debug(f"shape of w2: {nn.w2.shape}")
    logging.debug(nn.w2)

#sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#sigmoid derivative
def sigmoid_grad(z ,grad_output):
    z = z[1:]
    return grad_output * (1 - z) * z #there might be a mistake here

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

#gradient of cross entropy loss
def cross_entropy_grad(y, y_hat):
    return y_hat - y

#linear forward 
def linear(x, w):
    return np.dot(w, x)

#gradient of linear layer
def linear_grad(x, w, g_a):
    g_w = np.dot(g_a.reshape(g_a.shape[0], 1), x.reshape(1, x.shape[0]))
    g_x = np.dot(w[:, 1:].T, g_a)
    return g_w, g_x

def cross_entropy(y, y_hat):
    return np.dot(y, np.log(y_hat))

# one-hot encoding
def one_hot(y):
    return np.eye(10)[y]


def forward(X, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data
    :param nn: neural network class
    :return: output probability
    """
    A1 = linear(X, nn.w1)
    Z1 = sigmoid(A1)
    nn.Z1 = np.insert(Z1, 0, 1) #add bias terms
    A2 = linear(nn.Z1, nn.w2)
    y_hat = softmax(A2)
    return y_hat


def backward(X, y, y_hat, nn):
    """
    Neural network backward computation.
    Follow the pseudocode!
    :param X: input data
    :param y: label
    :param y_hat: prediction
    :param nn: neural network class
    :return:
    d_w1: gradients for w1
    d_w2: gradients for w2
    """
    g_A2 = cross_entropy_grad(y, y_hat)
    g_w2, g_Z1 = linear_grad(nn.Z1, nn.w2, g_A2)
    g_A1 = sigmoid_grad(nn.Z1, g_Z1)
    g_w1, g_X = linear_grad(X, nn.w1, g_A1)
    return g_w1, g_w2


def test(X, y, nn):
    """
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    """
    predictions = np.zeros(y.shape)
    for i , (example, label) in enumerate(zip(X, y)):
        y_hat = forward(example, nn)
        predictions[i] = np.argmax(y_hat)
    error_rate = np.sum(predictions != y) / len(y)
    return predictions, error_rate
    
def train(X_tr, y_tr, X_te, y_te, metrics_file, nn):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param nn: neural network class
    """
    for epoch in range(nn.n_epoch):
        nn.train_ce = 0
        nn.test_ce = 0
        # shuffle the data
        X_tr_shuffle, y_tr_shuffle = shuffle(X_tr, y_tr, epoch)
        # train the network
        for example , label in zip(X_tr_shuffle, y_tr_shuffle):
            y_hat = forward(example, nn)
            one_hot_label = one_hot(label)
            g_w1, g_w2 = backward(example, one_hot_label, y_hat, nn)
            # update the weights using adagrad
            nn.grad_sum_w1 += g_w1 ** 2
            nn.grad_sum_w2 += g_w2 ** 2
            nn.w1 -= (nn.lr / np.sqrt(nn.grad_sum_w1 + nn.epsilon)) * g_w1
            nn.w2 -= (nn.lr / np.sqrt(nn.grad_sum_w2 + nn.epsilon)) * g_w2

        # compute cross entropy for training 
        for example, label in zip(X_tr_shuffle, y_tr_shuffle):
            y_hat = forward(example, nn)
            one_hot_label = one_hot(label)
            nn.train_ce += cross_entropy(one_hot_label, y_hat)
        # compute cross entropy for testing
        for example, label in zip(X_te, y_te):
            y_hat = forward(example, nn)
            one_hot_label = one_hot(label)
            nn.test_ce += cross_entropy(one_hot_label, y_hat)
        # compute mean negative cross entropy for training and testing
        nn.train_ce /= -len(X_tr)
        nn.test_ce /= -len(X_te)
        # write the metrics to file
        nn.train_ce_list.append(nn.train_ce)
        nn.test_ce_list.append(nn.test_ce)
        with open(metrics_file, 'a') as f:
            f.write('epoch=' + str(epoch+1) + ' crossentropy(train): '+ str(nn.train_ce) + '\n')
            f.write('epoch=' + str(epoch+1) + ' crossentropy(validation): '+ str(nn.test_ce) + '\n')
            f.close()
    return nn.w1, nn.w2


if __name__ == "__main__":

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    # Note: You can access arguments like learning rate with args.learning_rate

    # initialize training / test data and labels
    (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics, n_epochs, n_hid, init_flag, lr) = args2data(args)
    # print(f"X_tr: {X_tr.shape}")
    # print(f"y_tr: {y_tr.shape}")

    #weight initialization function
    if init_flag == 1:
        weight_init_fn = random_init
    elif init_flag == 2:
        weight_init_fn = zero_init


    # Build model
    # my_nn = NN(lr, n_epochs, weight_init_fn, X_tr.shape[1], n_hid, 10)
    # print()
    # my_nn = NN(lr, n_epochs, zero_init, X_tr.shape[1], n_hid, y_tr.shape[1])
    # nn_hid5 = NN(lr, n_epochs, weight_init_fn, X_tr.shape[1], 5, 10)
    # nn_hid20 = NN(lr, n_epochs, weight_init_fn, X_tr.shape[1], 20, 10)
    # nn_hid50 = NN(lr, n_epochs, weight_init_fn, X_tr.shape[1], 50, 10)
    # nn_hid100 = NN(lr, n_epochs, weight_init_fn, X_tr.shape[1], 100, 10)
    # nn_hid200 = NN(lr, n_epochs, weight_init_fn, X_tr.shape[1], 200, 10)
    nn_lr0point1 = NN(0.1, n_epochs, weight_init_fn, X_tr.shape[1], 50, 10)
    nn_lr0point01 = NN(0.01, n_epochs, weight_init_fn, X_tr.shape[1], 50, 10)
    nn_lr0point001 = NN(0.001, n_epochs, weight_init_fn, X_tr.shape[1], 50, 10)

    # train model
    # w1, w2 = train(X_tr, y_tr, X_te, y_te, out_metrics, my_nn)
    # w1_hid5, w2_hid5 = train(X_tr, y_tr, X_te, y_te, out_metrics, nn_hid5)
    # w1_hid20, w2_hid20 = train(X_tr, y_tr, X_te, y_te, out_metrics, nn_hid20)
    # w1_hid50, w2_hid50 = train(X_tr, y_tr, X_te, y_te, out_metrics, nn_hid50)
    # w1_hid100, w2_hid100 = train(X_tr, y_tr, X_te, y_te, out_metrics, nn_hid100)
    # w1_hid200, w2_hid200 = train(X_tr, y_tr, X_te, y_te, out_metrics, nn_hid200)
    w1_lr0point1, w2_lr0point1 = train(X_tr, y_tr, X_te, y_te, out_metrics, nn_lr0point1)
    w1_lr0point01, w2_lr0point01 = train(X_tr, y_tr, X_te, y_te, out_metrics, nn_lr0point01)
    w1_lr0point001, w2_lr0point001 = train(X_tr, y_tr, X_te, y_te, out_metrics, nn_lr0point001)

    # test model and get predicted labels and errors
    # y_pred_tr, tr_error = test(X_tr, y_tr, my_nn)
    # y_pred_te, te_error = test(X_te, y_te, my_nn)

    # write predicted label and error into file
    # with open(out_tr, 'a') as f:
    #     for y in y_pred_tr:
    #         f.write(str(y) + '\n')
    # with open(out_te, 'a') as f:
    #     for y in y_pred_te:
    #         f.write(str(y) + '\n')
    # with open(out_metrics, 'a') as f:
    #     f.write('error(train): ' + str(tr_error) + '\n')
    #     f.write('error(validation): ' + str(te_error) + '\n')
    #     f.close()

#    # plot the loss function for different hidden layer sizes
# plt.plot(nn_hid5.train_ce_list, label='5 hidden units')
# plt.plot(nn_hid20.train_ce_list, label='20 hidden units')
# plt.plot(nn_hid50.train_ce_list, label='50 hidden units')
# plt.plot(nn_hid100.train_ce_list, label='100 hidden units')
# plt.xlabel('epoch')
# plt.ylabel('cross entropy')
# plt.legend()
# plt.title('cross entropy vs epoch')
# plt.show()

#read sgd.txt and get the data using np.loadtxt
# sgt_data = np.loadtxt('sgd.txt', delimiter=': ', dtype=str)
# x = np.float32(sgt_data[:,1][1::2])
# plt.plot(x, label='sgd')
# plt.plot(nn_hid50.test_ce_list, label='Adagrad')
# plt.xlabel('epoch')
# plt.ylabel('cross entropy')
# plt.legend()
# plt.title('cross entropy vs epoch')
# plt.show()

#plot training cross entropy vs epoch
plt.plot(nn_lr0point1.train_ce_list, label='lr=0.1 train cross entropy')
plt.plot(nn_lr0point1.test_ce_list, label='lr=0.1 validation cross entropy')
plt.xlabel('epoch')
plt.ylabel('cross entropy')
plt.legend()
plt.title('cross entropy vs epoch')
plt.show()

plt.plot(nn_lr0point01.train_ce_list, label='0.01 train cross entropy')
plt.plot(nn_lr0point01.test_ce_list, label='0.01 validation cross entropy')
plt.xlabel('epoch')
plt.ylabel('cross entropy')
plt.legend()
plt.title('cross entropy vs epoch')
plt.show()

plt.plot(nn_lr0point001.train_ce_list, label='0.001 train cross entropy')
plt.plot(nn_lr0point001.test_ce_list, label='0.001 validation cross entropy')
plt.xlabel('epoch')
plt.ylabel('cross entropy')
plt.legend()
plt.title('cross entropy vs epoch')
plt.show()

# plot training cross entropy vs number hidden units
# plt.plot([nn_hid5.train_ce, nn_hid20.train_ce, nn_hid50.train_ce, nn_hid100.train_ce, nn_hid200.train_ce], label='train cross entropy')
# plt.plot([nn_hid5.test_ce, nn_hid20.test_ce, nn_hid50.test_ce, nn_hid100.test_ce, nn_hid200.test_ce], label='validation cross entropy')
# # define x axis
# x = np.arange(5)
# # set x axis label
# plt.xticks(x, ('5', '20', '50', '100', '200'))
# plt.xlabel('number of hidden units')
# plt.ylabel('average cross entropy')
# plt.legend()
# plt.title('cross entropy vs number of hidden units')
# plt.show()

