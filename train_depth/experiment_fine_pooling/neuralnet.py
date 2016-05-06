import chainer
import chainer.functions as F
import math


class Net(chainer.FunctionSet):

    def __init__(self):
        super(Net, self).__init__(
            conv1 = F.Convolution2D(3, 16, 5, pad = 2),
            conv2 = F.Convolution2D(16, 16, 5, pad = 2),
            conv3 = F.Convolution2D(16, 16, 3, pad = 1),
            conv4 = F.Convolution2D(16, 16, 3, pad = 1),
            conv5 = F.Convolution2D(16, 16, 3, pad = 1),
            conv6 = F.Convolution2D(16, 1, 3, pad = 1)
        )


    def forward_super(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=1, pad = 1)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 3, stride=1, pad = 1)
        h = F.relu(self.conv5(h))
        y = self.conv6(h)


        return y

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        y = self.forward_super(x, train=train)

        return F.mean_squared_error(y, t)
