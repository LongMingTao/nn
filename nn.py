import numpy as np


class NN:
    def __init__(self, ns, weight=None):
        self.feature_amount = ns[0]
        self.neural_shape = ns
        self.class_amount = ns[-1]

        if not weight:
            tmp = []
            for i in range(0, self.neural_shape.shape[0] - 1):
                tmp.append(np.empty((self.neural_shape[i + 1], self.neural_shape[i])))
            self.weight = np.array(tmp)
        else:
            self.weight = weight

        self.train_x = None

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
        self.weight = self.__BP(self.train_x)
        return self.weight

    def __BP(self, t_x):

        return self.weight

    def __FF(self, x):
        y = x.copy()
        for n in self.weight:
            y = n.dot(y)
        return y


if __name__ == '__main__':
    nn = NN(np.array([3, 4, 4, 2]))
    print nn.get(np.array([[1, 2, 3], [4, 5, 6]]).reshape(3, -1))
