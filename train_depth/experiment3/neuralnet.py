import chainer
import chainer.functions as F
import math


class Net1(chainer.FunctionSet):

    def __init__(self):
        super(Net1, self).__init__(
            conv1 = F.Convolution2D(3, 256, 3, pad = 1),
            conv2 = F.Convolution2D(256, 1, 3, pad = 1),
        )


    def forward_super(self, x, train=True):
        h = F.relu(self.conv1(x))
        y = self.conv2(h)


        return y

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        y = self.forward_super(x, train=train)

        return F.mean_squared_error(y, t)

class Net2(chainer.FunctionSet):

    def __init__(self):
        super(Net2, self).__init__(
            conv1 = F.Convolution2D(3, 256, 3, pad = 1),
            conv2 = F.Convolution2D(256, 256, 3, pad = 1),
            conv3 = F.Convolution2D(256, 1, 3, pad = 1),
        )


    def forward_super(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        y = self.conv3(h)

        return y

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        y = self.forward_super(x, train=train)

        return F.mean_squared_error(y, t)

class Net3(chainer.FunctionSet):

    def __init__(self):
        super(Net3, self).__init__(
            conv1 = F.Convolution2D(3, 256, 3, pad = 1),
            conv2 = F.Convolution2D(256, 256, 3, pad = 1),
            conv3 = F.Convolution2D(256, 256, 3, pad = 1),
            conv4 = F.Convolution2D(256, 1, 3, pad = 1),
        )


    def forward_super(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        y = self.conv4(h)


        return y

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        y = self.forward_super(x, train=train)

        return F.mean_squared_error(y, t)

class Net4(chainer.FunctionSet):

    def __init__(self):
        super(Net4, self).__init__(
            conv1 = F.Convolution2D(3, 256, 3, pad = 1),
            conv2 = F.Convolution2D(256, 256, 3, pad = 1),
            conv3 = F.Convolution2D(256, 256, 3, pad = 1),
            conv4 = F.Convolution2D(256, 256, 3, pad = 1),
            conv5 = F.Convolution2D(256, 1, 3, pad = 1),
        )


    def forward_super(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        y = self.conv5(h)

        return y

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        y = self.forward_super(x, train=train)

        return F.mean_squared_error(y, t)

class Net5(chainer.FunctionSet):

    def __init__(self):
        super(Net5, self).__init__(
            conv1 = F.Convolution2D(3, 256, 3, pad = 1),
            conv2 = F.Convolution2D(256, 256, 3, pad = 1),
            conv3 = F.Convolution2D(256, 256, 3, pad = 1),
            conv4 = F.Convolution2D(256, 256, 3, pad = 1),
            conv5 = F.Convolution2D(256, 256, 3, pad = 1),
            conv6 = F.Convolution2D(256, 1, 3, pad = 1),
        )


    def forward_super(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        y = self.conv6(h)

        return y

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        y = self.forward_super(x, train=train)

        return F.mean_squared_error(y, t)
