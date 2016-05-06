import chainer
import chainer.functions as F
import math


class Net(chainer.FunctionSet):

    def __init__(self, filter_size):
        super(Net, self).__init__(
            conv1 = F.Convolution2D(3, 256, filter_size, pad = filter_size // 2),
            conv2 = F.Convolution2D(256, 256, filter_size, pad = filter_size // 2),
            conv3 = F.Convolution2D(256, 1, filter_size, pad = filter_size // 2),
        )


    def forward_super(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        y = self.conv3(h)


        return y

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        y = self.forward_super(x, train=train)

        return F.mean_squared_error(y, t)
