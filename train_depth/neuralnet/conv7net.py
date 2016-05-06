import chainer
import chainer.functions as F


class Conv7Net(chainer.FunctionSet):

    def __init__(self):
        super(Conv7Net, self).__init__(
            conv1 = F.Convolution2D(3, 32, 5, pad = 2),
            bn1 = F.BatchNormalization(32),
            conv2 = F.Convolution2D(32, 64, 5, pad = 2),
            bn2 = F.BatchNormalization(64),
            conv3 = F.Convolution2D(64, 128, 5, pad = 2),
            bn3 = F.BatchNormalization(128),
            conv4 = F.Convolution2D(128, 256, 5, pad = 2),
            bn4 = F.BatchNormalization(256),
            conv5 = F.Convolution2D(256, 256, 5, pad = 2),
            bn5 = F.BatchNormalization(256),
            conv6 = F.Convolution2D(256, 256, 5, pad = 2),
            bn6 = F.BatchNormalization(256),
            conv7 = F.Convolution2D(256, 32, 5, pad = 2),
            conv8 = F.Convolution2D(32, 1, 5, pad = 2)
        )


    def forward_super(self, x, train=True):
        h = F.sigmoid(self.bn1(self.conv1(x)))
        h = F.sigmoid(self.bn2(self.conv2(h)))
        h = F.sigmoid(self.bn3(self.conv3(h)))
        h = F.sigmoid(self.bn4(self.conv4(h)))
        h = F.sigmoid(self.bn5(self.conv5(h)))
        h = F.sigmoid(self.bn6(self.conv6(h)))
        h = F.sigmoid(self.conv7(h))
        y = F.sigmoid(self.conv8(h))

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
