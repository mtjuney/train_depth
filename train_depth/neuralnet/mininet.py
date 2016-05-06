import chainer
import chainer.functions as F


class MiniNet(chainer.FunctionSet):

    def __init__(self):
        super(MiniNet, self).__init__(
            conv1 = F.Convolution2D(3, 2, 3, pad = 1),
            conv2 = F.Convolution2D(2, 1, 3, pad = 1)
        )

    def forward_super(self, x, train=True):
        h = F.relu(self.conv1(x))
        y = F.relu(self.conv2(h))
        return y



    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        y = self.forward_super(x, train=train)

        # return F.mean_squared_error(y, t), F.accuracy(y, t)
        return F.mean_squared_error(y, t)

    def forward_result(self, x_data):
        x = chainer.Variable(x_data, volatile=True)
        y = self.forward_super(x, train=False)
        return y
