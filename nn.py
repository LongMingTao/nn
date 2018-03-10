from collections import Iterable
import math
import numpy as np


def sigmoid(x, is_logical=True, bound=0.51):
    if isinstance(x, Iterable):
        for i in range(len(x)):
            x[i] = sigmoid(x[i], is_logical)
    else:
        x = 1.0 / (1.0 + math.exp(-x))
        if is_logical:
            x = 1 if x >= bound else 0
    return x


class NN:
    def __init__(self, ns, weight=None):
        self.feature_amount = ns[0]
        self.neural_shape = ns
        self.class_amount = ns[-1]

        if not weight:
            self.__weight = []
            for i in range(0, self.neural_shape.shape[0] - 1):
                self.__weight.append(np.empty((self.neural_shape[i + 1], self.neural_shape[i])))
        else:
            self.__weight = weight

        self.train_x = None

    def set_weight(self, weight):
        self.__weight = weight

    def get_weight(self):
        return self.__weight

    def get(self, x):
        if x.shape[0] != self.feature_amount:
            raise Exception('bad input, feature amount should be {0}, input is {1}'.format(str(self.feature_amount),
                                                                                           str(x.shape[1])))
        return self.__FF(x)

    def train(self, x):
        if x.shape[1] != self.feature_amount:
            raise Exception('bad input, feature amount should be {0}, input is {1}'.format(str(self.feature_amount),
                                                                                           str(x.shape[1])))
        self.train_x = x
        self.__weight = self.__BP(self.train_x)
        return self.__weight

    def __BP(self, t_x):
        return self.__weight

    def __FF(self, x):
        y = x
        for n in self.__weight:
            y = sigmoid(n.dot(y))
            # print y
        y = np.array(y)
        return sigmoid(y)


def test_xnor(data):
    nn = NN(np.array([3, 3, 1]))
    tmp = [np.array(([1, 0, 0], [-30, 20, 20], [10, -20, -10])), np.array([-10, 20, 20])]
    nn.set_weight(tmp)
    print nn.get(np.transpose(data))


def test_and(data):
    nn = NN(np.array([3, 1]))
    tmp = [np.array([-30, 20, 20])]
    nn.set_weight(tmp)
    print nn.get(np.transpose(data))


def test_or(data):
    nn = NN(np.array([3, 1]))
    tmp = [np.array([-10, 20, 20])]
    nn.set_weight(tmp)
    print nn.get(np.transpose(data))


if __name__ == '__main__':
    data = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
    print '----- data -----'
    print data
    print '----- and -----'
    test_and(data)
    print '----- or -----'
    test_or(data)
    print '----- xnor -----'
    test_xnor(data)
