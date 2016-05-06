import chainer
import chainer.functions as F
import math


class Conv14Net(chainer.FunctionSet):

    w = 0.05

    def __init__(self):
        super(Conv14Net, self).__init__(
            conv1 = F.Convolution2D(3, 32, 3, pad = 1),
            conv2 = F.Convolution2D(32, 64, 3, pad = 1),
            conv3 = F.Convolution2D(64, 128, 3, pad = 1),
            conv4 = F.Convolution2D(128, 64, 3, pad = 1),
            conv5 = F.Convolution2D(64, 1, 3, pad = 1),

            conv3b = F.Convolution2D(64, 1, 1),
        )


    def forward_super(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h1 = F.relu(self.conv3b(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))

        return h, h1


    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        y, y1 = self.forward_super(x, train=train)

        L = F.mean_squared_error(y, t)
        L1 = F.mean_squared_error(y1, t)

        if train:
            Loss = L + (L1 * 0.5)
        else:
            Loss = L

        return Loss

    def forward_result(self, x_data):
        x = chainer.Variable(x_data, volatile=True)
        y, y1 = self.forward_super(x, train=False)
        return y
