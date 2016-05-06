import chainer
import chainer.functions as F


class Conv2Net(chainer.FunctionSet):

    def __init__(self):
        super(Conv2Net, self).__init__(
            conv1 = F.Convolution2D(3, 32, 5, pad = 2),
            conv2 = F.Convolution2D(32, 32, 5, pad = 2),
            conv3 = F.Convolution2D(32, 32, 5, pad = 2),
            conv4 = F.Convolution2D(32, 64, 5, pad = 2),
            conv5 = F.Convolution2D(64, 1, 5, pad = 2)
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
