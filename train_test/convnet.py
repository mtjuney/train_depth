import chainer
import chainer.functions as F


class ConvNet(chainer.FunctionSet):

    def __init__(self):
        super(ConvNet, self).__init__(
            conv1 = F.Convolution2D(3, 32, 3, pad = 1),
            conv2 = F.Convolution2D(32, 32, 3, pad = 1),
            conv3 = F.Convolution2D(32, 32, 3, pad = 1),
            conv4 = F.Convolution2D(32, 64, 3, pad = 1),
            conv5 = F.Convolution2D(64, 1, 3, pad = 1)
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        y = F.relu(self.conv5(h))

        # return F.mean_squared_error(y, t), F.accuracy(y, t)
        return F.mean_squared_error(y, t)
