import chainer
import chainer.functions as F


class MiniNet(chainer.FunctionSet):

    def __init__(self, NUM_PIXEL_I):
        super(MiniNet, self).__init__(
            conv = F.Convolution2D(3, 2, 3, pad = 1),
            fl = F.Linear(NUM_PIXEL_I * 2, NUM_PIXEL_I)
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        h = F.relu(self.conv(x))
        y = F.relu(self.fl(h))

        # return F.mean_squared_error(y, t), F.accuracy(y, t)
        return F.mean_squared_error(y, t)
