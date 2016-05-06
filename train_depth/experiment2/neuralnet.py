import chainer
import chainer.functions as F
import math


class Net2(chainer.FunctionSet):

    def __init__(self):
        super(Net2, self).__init__(
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

class Net3(chainer.FunctionSet):

    def __init__(self):
        super(Net3, self).__init__(
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

class Net4(chainer.FunctionSet):

    def __init__(self):
        super(Net4, self).__init__(
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

class Net6(chainer.FunctionSet):

    def __init__(self):
        super(Net6, self).__init__(
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

class Net8(chainer.FunctionSet):

    def __init__(self):
        super(Net8, self).__init__(
            conv1 = F.Convolution2D(3, 256, 3, pad = 1),
            conv2 = F.Convolution2D(256, 256, 3, pad = 1),
            conv3 = F.Convolution2D(256, 256, 3, pad = 1),
            conv4 = F.Convolution2D(256, 256, 3, pad = 1),
            conv5 = F.Convolution2D(256, 256, 3, pad = 1),
            conv6 = F.Convolution2D(256, 256, 3, pad = 1),
            conv7 = F.Convolution2D(256, 256, 3, pad = 1),
            conv8 = F.Convolution2D(256, 1, 3, pad = 1),
        )


    def forward_super(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        y = self.conv8(h)

        return y

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        y = self.forward_super(x, train=train)

        return F.mean_squared_error(y, t)

class Net10(chainer.FunctionSet):

    def __init__(self):
        super(Net10, self).__init__(
            conv1 = F.Convolution2D(3, 256, 3, pad = 1),
            conv2 = F.Convolution2D(256, 256, 3, pad = 1),
            conv3 = F.Convolution2D(256, 256, 3, pad = 1),
            conv4 = F.Convolution2D(256, 256, 3, pad = 1),
            conv5 = F.Convolution2D(256, 256, 3, pad = 1),
            conv6 = F.Convolution2D(256, 256, 3, pad = 1),
            conv7 = F.Convolution2D(256, 256, 3, pad = 1),
            conv8 = F.Convolution2D(256, 256, 3, pad = 1),
            conv9 = F.Convolution2D(256, 256, 3, pad = 1),
            conv10 = F.Convolution2D(256, 1, 3, pad = 1),
        )


    def forward_super(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        y = self.conv10(h)

        return y

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        y = self.forward_super(x, train=train)

        return F.mean_squared_error(y, t)

class Net12(chainer.FunctionSet):

    def __init__(self):
        super(Net12, self).__init__(
            conv1 = F.Convolution2D(3, 256, 3, pad = 1),
            conv2 = F.Convolution2D(256, 256, 3, pad = 1),
            conv3 = F.Convolution2D(256, 256, 3, pad = 1),
            conv4 = F.Convolution2D(256, 256, 3, pad = 1),
            conv5 = F.Convolution2D(256, 256, 3, pad = 1),
            conv6 = F.Convolution2D(256, 256, 3, pad = 1),
            conv7 = F.Convolution2D(256, 256, 3, pad = 1),
            conv8 = F.Convolution2D(256, 256, 3, pad = 1),
            conv9 = F.Convolution2D(256, 256, 3, pad = 1),
            conv10 = F.Convolution2D(256, 256, 3, pad = 1),
            conv11 = F.Convolution2D(256, 256, 3, pad = 1),
            conv12 = F.Convolution2D(256, 1, 3, pad = 1),
        )


    def forward_super(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.relu(self.conv11(h))
        y = self.conv12(h)

        return y

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        y = self.forward_super(x, train=train)

        return F.mean_squared_error(y, t)
