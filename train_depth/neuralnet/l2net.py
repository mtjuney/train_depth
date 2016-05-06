import chainer
import chainer.functions as F


class L2Net(chainer.FunctionSet):

    def __init__(self, IN_PIXEL):
        super(L2Net, self).__init__(
            conv1 = F.Convolution2D(3, 64, 5, pad = 2),
            bn1 = F.BatchNormalization(64),
            conv2 = F.Convolution2D(64, 256, 5, pad = 2),
            bn2 = F.BatchNormalization(256),
            conv3 = F.Convolution2D(256, 256, 3, pad = 1),
            conv4 = F.Convolution2D(256, 64, 3, pad = 1),
            fc1 = F.Linear(IN_PIXEL * 64, IN_PIXEL * 8),
            fc2 = F.Linear(IN_PIXEL * 8, IN_PIXEL)
        )


    def forward_super(self, x, train=True):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.dropout(F.relu(self.fc1(h)), train=train)
        y = F.dropout(F.relu(self.fc2(h)), train=train)

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
