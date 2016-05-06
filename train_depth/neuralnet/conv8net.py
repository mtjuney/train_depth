import chainer
import chainer.functions as F


class Conv8Net(chainer.FunctionSet):

    def __init__(self):
        super(Conv8Net, self).__init__(
            conv1 = F.Convolution2D(3, 32, 5, pad = 2),
            bn1 = F.BatchNormalization(32),
            conv2 = F.Convolution2D(32, 64, 5, pad = 2),
            bn2 = F.BatchNormalization(64),
            conv3 = F.Convolution2D(64, 128, 3, pad = 1),
            bn3 = F.BatchNormalization(128),
            conv4 = F.Convolution2D(128, 32, 3, pad = 1),
            conv5 = F.Convolution2D(32, 1, 3, pad = 1)
        )


    def forward_super(self, x, train=True):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.conv4(h))
        y = F.relu(self.conv5(h))

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
