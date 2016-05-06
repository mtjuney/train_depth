import chainer
import chainer.functions as F


class DefaultNet(chainer.FunctionSet):

    def __init__(self, NUM_PIXEL_I):
        super(DefaultNet, self).__init__(
            conv1 = F.Convolution2D(3, 32, 3, pad = 1),
            conv2 = F.Convolution2D(32, 32, 5, pad = 2),
            conv3 = F.Convolution2D(32, 8, 3, pad = 1),
            fl1 = F.Linear(NUM_PIXEL_I * 8, NUM_PIXEL_I)
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        y = F.relu(self.fl1(h))

        # return F.mean_squared_error(y, t), F.accuracy(y, t)
        return F.mean_squared_error(y, t)
