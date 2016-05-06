import chainer
import chainer.functions as F


class Conv4Net(chainer.FunctionSet):

    def __init__(self):
        super(Conv4Net, self).__init__(
            conv1 = F.Convolution2D(3, 32, 3, pad = 1),
            conv2 = F.Convolution2D(32, 64, 3, pad = 1),
            conv3 = F.Convolution2D(64, 128, 3, pad = 1),
            conv4 = F.Convolution2D(128, 256, 3, pad = 1),
            conv5 = F.Convolution2D(256, 256, 3, pad = 1),
            conv6 = F.Convolution2D(256, 512, 3, pad = 1),
            conv7 = F.Convolution2D(512, 512, 3, pad = 1),
            conv8 = F.Convolution2D(512, 512, 3, pad = 1),
            conv9 = F.Convolution2D(512, 256, 3, pad = 1),
            conv10 = F.Convolution2D(256, 1, 3, pad = 1)
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        y = F.relu(self.conv10(h))

        # return F.mean_squared_error(y, t), F.accuracy(y, t)
        return F.mean_squared_error(y, t)
