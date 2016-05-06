import chainer
import chainer.functions as F


class Conv6Net(chainer.FunctionSet):

    def __init__(self):
        super(Conv6Net, self).__init__(
            conv1 = F.Convolution2D(3, 32, 5, pad = 2),
            conv2 = F.Convolution2D(32, 64, 5, pad = 2),
            conv3 = F.Convolution2D(64, 128, 5, pad = 2),
            conv4 = F.Convolution2D(128, 256, 5, pad = 2),
            conv5 = F.Convolution2D(256, 256, 5, pad = 2),
            conv6 = F.Convolution2D(256, 256, 5, pad = 2),
            conv7 = F.Convolution2D(256, 32, 5, pad = 2),
            conv8 = F.Convolution2D(32, 1, 5, pad = 2)
        )


    def forward_super(self, x):
        h = F.sigmoid(self.conv1(x))
        h = F.sigmoid(self.conv2(h))
        h = F.sigmoid(self.conv3(h))
        h = F.sigmoid(self.conv4(h))
        h = F.sigmoid(self.conv5(h))
        h = F.sigmoid(self.conv6(h))
        h = F.sigmoid(self.conv7(h))
        y = F.sigmoid(self.conv8(h))

        return y

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        y = self.forward_super(x)

        # return F.mean_squared_error(y, t), F.accuracy(y, t)
        return F.mean_squared_error(y, t)

    def forward_result(self, x_data):
        x = chainer.Variable(x_data, volatile=True)
        y = self.forward_super(x)
        return y
