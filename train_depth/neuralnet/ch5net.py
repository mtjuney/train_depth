import chainer
import chainer.functions as F
import six.moves.cPickle as pickle


class ChNet:

    w = 0.05
    p_w = 7
    p_l = 1
    fs = None

    def __init__(self):
        fs = chainer.FunctionSet(
            conv1 = F.Convolution2D(3, 32, 3, pad = 1),
            conv2 = F.Convolution2D(32, 64, 3, pad = 1),
            conv3 = F.Convolution2D(64, 128, 3, pad = 1),
            conv4 = F.Convolution2D(128, 256, 3, pad = 1),
            conv5 = F.Convolution2D(256, 1, 1),
            conv5b = F.Convolution2D(256, 1, 1),
        )


    def forward_super(self, x, train=True):
        h = F.relu(fs.conv1(x))
        h = F.relu(fs.conv2(h))
        h = F.relu(fs.conv3(h))
        h = F.relu(fs.conv4(h))
        h1 = F.relu(fs.conv5(h))
        h = F.relu(fs.conv5(h))

        h1 = F.average_pooling_2d(h, self.p_w, stride=self.p_w)

        return h, h1


    def forward(self, x_data, y_data, y1_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)
        t1 = chainer.Variable(y1_data, volatile=not train)

        y, y1 = self.forward_super(x, train=train)

        L = F.mean_squared_error(y, t)
        L1 = F.mean_squared_error(y1, t1)

        if train:
            Loss = L + (0.5 * L1)
        else:
            Loss = L

        return Loss

    def forward_result(self, x_data):
        x = chainer.Variable(x_data, volatile=True)
        y, y1 = self.forward_super(x, train=False)
        return y, y1


    def save(self, output='model'):
        pickle.dump(fs, open(output, 'wb'), -1)
