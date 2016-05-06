import chainer
import chainer.functions as F
import math


class Conv11Net(chainer.FunctionSet):

    def __init__(self):
        w = 0.05
        super(Conv11Net, self).__init__(
            conv1 = F.Convolution2D(3, 32, 3, pad = 1, wscale=w),
            conv1a = F.Convolution2D(32, 32, 1, wscale=w),
            conv2 = F.Convolution2D(32, 64, 3, pad = 1, wscale=w),
            conv2a = F.Convolution2D(64, 64, 1, wscale=w),
            conv3 = F.Convolution2D(64, 128, 3, pad = 1, wscale=w),
            conv3a = F.Convolution2D(128, 128, 1, wscale=w),
            conv4 = F.Convolution2D(128, 256, 3, pad = 1, wscale=w),
            conv4a = F.Convolution2D(256, 256, 1, wscale=w),
            conv5 = F.Convolution2D(256, 512, 3, pad = 1, wscale=w),
            conv5a = F.Convolution2D(512, 512, 1, wscale=w),
            conv6 = F.Convolution2D(512, 128, 3, pad = 1, wscale=w),
            conv6a = F.Convolution2D(128, 128, 1, wscale=w),
            conv7 = F.Convolution2D(128, 1, 3, pad = 1, wscale=w),
            conv7a = F.Convolution2D(1, 1, 1, wscale=w),
        )


    def forward_super(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv1a(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv2a(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv3a(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv4a(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv5a(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv6a(h))
        h = F.relu(self.conv7(h))
        y = F.relu(self.conv7a(h))

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
