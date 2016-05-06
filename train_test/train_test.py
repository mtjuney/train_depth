import argparse
import os
import sys
import csv
import time
import datetime
import yaml


from PIL import Image
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F

parser = argparse.ArgumentParser(description='Research Program')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()


# config file

if os.path.exists('config.yml'):
    CONFIG_FILE = 'config.yml'
else:
    CONFIG_FILE = 'config_default.yml'


f = open(CONFIG_FILE, 'r')
config = yaml.load(f)
f.close

# static const

n_epoch = config['n_epoch']
batchsize = config['batchsize']

WIDTH_IMG = config['WIDTH_IMG']
HEIGHT_IMG = config['HEIGHT_IMG']

print "n_epoch", n_epoch
print "batchsize", batchsize
print "WIDTH_IMG", WIDTH_IMG
print "HEIGHT_IMG", HEIGHT_IMG



def add_record(filename, row_array):
    with open(filename, 'a') as f:
        record = csv.writer(f, lineterminator='\n')
        record.writerow(row_array)

datetime_string = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
record_loss_filename = 'record_loss/' + datetime_string + '.csv'
record_time_filename = 'record_time/' + datetime_string + '.csv'

with open(record_loss_filename, 'a') as f:
    record = csv.writer(f, lineterminator='\n')
    record.writerows(config.items())

with open(record_time_filename, 'a') as f:
    record = csv.writer(f, lineterminator='\n')
    record.writerows(config.items())


path_train_x = 'data_train/train/x/'
path_train_y = 'data_train/train/y/'

path_test_x = 'data_train/test/x/'
path_test_y = 'data_train/test/y/'




t_start = time.time()

NUM_PIXEL_I = WIDTH_IMG * HEIGHT_IMG



def read_image(path, x_data=False):
    if x_data:
        image = np.asarray(Image.open(path).resize((WIDTH_IMG, HEIGHT_IMG))).transpose(2, 0, 1)
    else:
        image = np.asarray(Image.open(path).resize((WIDTH_IMG, HEIGHT_IMG)))

    # image /= 255
    return image


names_file_train = os.listdir(path_train_x)
train_x_data = np.empty((len(names_file_train), 3, HEIGHT_IMG, WIDTH_IMG), dtype=np.float32)
train_y_data = np.empty((len(names_file_train), HEIGHT_IMG, WIDTH_IMG), dtype=np.float32)

for i in range(len(names_file_train)):
    image_x = read_image(path_train_x + names_file_train[i], x_data=True)
    image_y = read_image(path_train_y + names_file_train[i])

    train_x_data[i] = image_x
    train_y_data[i] = image_y

train_y_data = train_y_data.reshape((len(names_file_train), HEIGHT_IMG * WIDTH_IMG))


names_file_test = os.listdir(path_test_x)
test_x_data = np.empty((len(names_file_test), 3, HEIGHT_IMG, WIDTH_IMG), dtype=np.float32)
test_y_data = np.empty((len(names_file_test), HEIGHT_IMG, WIDTH_IMG), dtype=np.float32)

for i in range(len(names_file_test)):
    image_x = read_image(path_test_x + names_file_test[i], x_data=True)
    image_y = read_image(path_test_y + names_file_test[i])

    test_x_data[i] = image_x
    test_y_data[i] = image_y

test_y_data = test_y_data.reshape((len(names_file_test), HEIGHT_IMG * WIDTH_IMG))


# Read Network

import defaultnet
model = defaultnet.DefaultNet(NUM_PIXEL_I)

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

N_train = len(train_x_data)
N_test = len(test_x_data)
print 'Train Data :', N_train
print 'Test Data :', N_test

optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

add_record(record_time_filename, ['pre', time.time() - t_start])


for epoch in xrange(1, n_epoch + 1):
    print 'epoch', epoch

    t_epoch_start = time.time()

    # training
    perm = np.random.permutation(N_train)
    sum_loss = 0

    for i in xrange(0, N_train, batchsize):
        x_batch = train_x_data[perm[i:i+batchsize]]
        y_batch = train_y_data[perm[i:i+batchsize]]

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        optimizer.zero_grads()

        loss = model.forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize

        sharp_num = (i + 1) * 20 / N_train
        progress_show = '[' + ('#' * sharp_num) + (' ' * (20 - sharp_num)) + ']' + '({}/{})'.format(i, N_train)

        if not (i == 0):
            progress_show = '\r' + progress_show

        sys.stdout.write(progress_show)
        sys.stdout.flush()

    sys.stdout.write('\n')

    # print 'train mean loss = {}, accuracy = {}'.format(sum_loss / N_train, sum_accuracy / N_train)

    loss_show = sum_loss / N_train
    print 'epoch {} : train mean loss = {}'.format(epoch, loss_show)
    # add_record(record_loss_filename, ['epoch{}'.format(epoch), loss_show])
    t_epoch_break = time.time()
    add_record(record_time_filename, ['epoch{}-train'.format(epoch), t_epoch_break - t_epoch_start])



    # evaluation
    perm = np.random.permutation(N_test)
    sum_loss = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = test_x_data[perm[i:i+batchsize]]
        y_batch = test_y_data[perm[i:i+batchsize]]

        loss = model.forward(x_batch, y_batch, train = False)

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize


    loss_show = sum_loss / N_test
    print 'epoch {} : test mean loss = {}'.format(epoch, loss_show)
    add_record(record_loss_filename, ['epoch{}'.format(epoch), loss_show])
    add_record(record_time_filename, ['epoch{}-test'.format(epoch), time.time() - t_epoch_break])


add_record(record_time_filename, ['all', time.time() - t_start])
