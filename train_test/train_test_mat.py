#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images and scale them to 256x256, and make two lists of space-
separated CSV whose first column is full path to image and second column is
zero-origin label (this format is same as that used by Caffe's ImageDataLayer).

"""
from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing
import random
import sys
import threading
import time
import os
import yaml
import csv

import numpy as np
from PIL import Image
import six
import six.moves.cPickle as pickle
from six.moves import queue

# from chainer import computational_graph as c
from chainer import cuda
from chainer import optimizers

from readmat import readDepthMap

parser = argparse.ArgumentParser(description='Learning convnet from ILSVRC2012 dataset')
# parser.add_argument('train', help='Path to training image-label list file')
# parser.add_argument('val', help='Path to validation image-label list file')
# parser.add_argument('--mean', '-m', default='mean.npy',
#                     help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--arch', '-a', default='default',
                    help='Convnet architecture \
                    (mini)')
parser.add_argument('--BATCHSIZE', '-B', type=int, default=25,
                    help='Learning minibatch size')
# parser.add_argument('--val_BATCHSIZE', '-b', type=int, default=250,
#                     help='Validation minibatch size')
# parser.add_argument('--epoch', '-E', default=10, type=int,
#                     help='Number of epochs to learn')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
# parser.add_argument('--loaderjob', '-j', default=20, type=int,
#                     help='Number of parallel data loading processes')
# parser.add_argument('--out', '-o', default='model',
#                     help='Path to save model on each validation')
parser.add_argument('--optimizer', '-o', default='adam', help='optimizer select')
args = parser.parse_args()


# Prepare dataset


if os.path.isfile('config.yml'):
    CONFIG_FILE = 'config.yml'
else:
    CONFIG_FILE = 'config_default.yml'

config = None
with open(CONFIG_FILE, 'r') as f:
    config = yaml.load(f)

N_EPOCH = config['N_EPOCH']
BATCHSIZE = VAL_BATCHSIZE = config['BATCHSIZE']
WIDTH_IMG = config['WIDTH_IMG']
HEIGHT_IMG = config['HEIGHT_IMG']

print("N_EPOCH", N_EPOCH)
print("BATCHSIZE", BATCHSIZE)
print("WIDTH_IMG", WIDTH_IMG)
print("HEIGHT_IMG", HEIGHT_IMG)

path_train_x = 'data_train/train/x/'
path_train_y = 'data_train/train/y/'

path_test_x = 'data_train/test/x/'
path_test_y = 'data_train/test/y/'

# LOGPATH = 'log/' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
# os.mkdir(LOGPATH)
# LOGPATH += '/'

i = 0
while(os.path.isdir('log/{:0>6}'.format(i))):
    i += 1
os.mkdir('log/{:0>6}'.format(i))
LOGPATH = 'log/{:0>6}/'.format(i)


config_write = {'N_EPOCH':N_EPOCH, 'BATCHSIZE':BATCHSIZE, 'WIDTH_IMG':WIDTH_IMG, 'HEIGHT_IMG':HEIGHT_IMG, 'GPU':args.gpu, 'ARCH':args.arch}
f = open(LOGPATH + 'config.yml', 'w')
f.write(yaml.dump(config_write))
f.close()

NUM_PIXEL_I = WIDTH_IMG * HEIGHT_IMG


# Prepare model
model = None
if args.arch == 'mini':
    import mininet
    model = mininet.MiniNet(NUM_PIXEL_I)
elif args.arch == 'conv':
    import convnet
    model = convnet.ConvNet()
elif args.arch == 'conv2':
    import conv2net
    model = conv2net.Conv2Net()
elif args.arch == 'conv3':
    import conv3net
    model = conv3net.Conv3Net()
elif args.arch == 'conv4':
    import conv4net
    model = conv4net.Conv4Net()
else:
    import defaultnet
    model = defaultnet.DefaultNet(NUM_PIXEL_I)

if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()

xp = cuda.cupy if args.gpu >= 0 else np

# Setup optimizer
if args.optimizer == 'adam':
    optimizer = optimizers.Adam()
elif args.optimizer == 'adadelta':
    optimizer = optimizers.AdaDelta()
elif args.optimizer == 'adagrad':
    optimizer = optimizers.AdaGrad()
elif args.optimizer == 'rmsprop':
    optimizer = optimizers.RMSprop()
else:
    optimizer = optimizers.Adam()

# optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)

optimizer.setup(model)


train_list = os.listdir(path_train_x)
val_list = os.listdir(path_test_x)

train_num = len(train_list) - (len(train_list) % BATCHSIZE)
val_num = len(val_list) - (len(val_list) % BATCHSIZE)


def add_record(row_array, record_type):
    if record_type == 'time':
        filename = LOGPATH + 'record_time.csv'
    else:
        filename = LOGPATH + 'record_loss.csv'

    with open(filename, 'a') as f:
        record = csv.writer(f, lineterminator='\n')
        record.writerow(row_array)


# ------------------------------------------------------------------------------
# This example consists of three threads: data feeder, logger and trainer.
# These communicate with each other via Queue.
data_q = queue.Queue(maxsize=1)
res_q = queue.Queue()

def read_image(filename, train_data=True):

    if train_data:
        x_path = path_train_x
        y_path = path_train_y
    else:
        x_path = path_test_x
        y_path = path_test_y

    image_x = np.asarray(Image.open(x_path + filename).resize((WIDTH_IMG, HEIGHT_IMG))).transpose(2, 1, 0)
    image_y = np.asarray(readDepthMap(y_path + filename).resize((WIDTH_IMG, HEIGHT_IMG))).reshape((WIDTH_IMG, HEIGHT_IMG, 1)).transpose(2, 1, 0)

    # image /= 255
    return (image_x, image_y)

# Data feeder


def feed_data():
    i = 0
    count = 0

    x_batch = np.ndarray((BATCHSIZE, 3, HEIGHT_IMG, WIDTH_IMG), dtype=np.float32)
    y_batch = np.ndarray((BATCHSIZE, 1, HEIGHT_IMG, WIDTH_IMG), dtype=np.float32)
    val_x_batch = np.ndarray((VAL_BATCHSIZE, 3, HEIGHT_IMG, WIDTH_IMG), dtype=np.float32)
    val_y_batch = np.ndarray((VAL_BATCHSIZE, 1, HEIGHT_IMG, WIDTH_IMG), dtype=np.float32)

    batch_pool = [None] * BATCHSIZE
    val_batch_pool = [None] * VAL_BATCHSIZE
    pool = multiprocessing.Pool()
    # data_q.put('train')
    for epoch in six.moves.range(1, 1 + N_EPOCH):
        # print('epoch', epoch, file=sys.stderr)
        # print('learning rate', optimizer.lr, file=sys.stderr)
        data_q.put('train')
        perm = np.random.permutation(len(train_list))
        for idx in perm:
            filename = train_list[idx]
            batch_pool[i] = pool.apply_async(read_image, (filename, True))
            i += 1

            if i == BATCHSIZE:
                for j, x in enumerate(batch_pool):
                    (x_batch[j], y_batch[j]) = x.get()
                data_q.put((x_batch.copy(), y_batch.copy()))
                i = 0

            count += 1

        data_q.put('val')
        j = 0
        for filename in val_list:
            val_batch_pool[j] = pool.apply_async(read_image, (filename, False))
            j += 1

            if j == VAL_BATCHSIZE:
                for k, x in enumerate(val_batch_pool):
                    (val_x_batch[k], val_y_batch[k]) = x.get()
                data_q.put((val_x_batch.copy(), val_y_batch.copy()))
                j = 0
        # data_q.put('train')

        # optimizer.lr *= 0.97

    pool.close()
    pool.join()
    data_q.put('end')

# Logger

def show_progress(value, max_value):
    MAX = 20
    sharp_num = value * MAX / max_value
    progress = '[' + ('#' * sharp_num) + ('-' * (20 - sharp_num)) + ']'
    progress += '({}/{})'.format(value, max_value)
    return progress

def log_result():
    train_count = 0
    train_cur_loss = 0
    epoch = 0


    t_train_start = t_val_start = None
    t_train = []
    t_val = []
    while True:
        result = res_q.get()
        if result == 'end':
            t_val.append(time.time() - t_val_start)
            add_record([epoch, t_train[epoch - 1], t_val[epoch - 1]], 'time')
            mean_loss = val_loss * BATCHSIZE / val_count
            add_record([epoch, mean_loss], 'loss')
            print()
            print('val mean loss :{}'.format(mean_loss))
            print()
            break
        elif result == 'train':
            if epoch > 0:
                t_val.append(time.time() - t_val_start)
                add_record([epoch, t_train[epoch - 1], t_val[epoch - 1]], 'time')
                mean_loss = val_loss * BATCHSIZE / val_count
                add_record([epoch, mean_loss], 'loss')
                print()
                print('val mean loss :{}'.format(mean_loss))
                print()

            epoch += 1
            train = True
            print('epoch:' + str(epoch))
            train_count = 0
            t_train_start = time.time()
            # if val_begin_at is not None:
            #     begin_at += time.time() - val_begin_at
            #     val_begin_at = None
            continue
        elif result == 'val':
            t_train.append(time.time() - t_train_start)

            train = False
            print()
            val_count = val_loss = 0
            t_val_start = time.time()
            continue

        loss = result
        if train:
            train_count += BATCHSIZE

            progress = '\rtrain\t' + show_progress(train_count, train_num)
            sys.stdout.write(progress)
            sys.stdout.flush
            # duration = time.time() - begin_at
            # throughput = train_count * BATCHSIZE / duration
            # sys.stderr.write(
            #     '\rtrain {} updates ({} samples) time: {} ({} images/sec)'
            #     .format(train_count, train_count * BATCHSIZE,
            #             datetime.timedelta(seconds=duration), throughput))

            train_cur_loss += loss
            # if train_count % 1000 == 0:
            #     mean_loss = train_cur_loss / 1000
            #     print(file=sys.stderr)
            #     print(json.dumps({'type': 'train', 'iteration': train_count, 'loss': mean_loss}))
            #     sys.stdout.flush()
            #     train_cur_loss = 0
        else:
            val_count += VAL_BATCHSIZE

            progress = '\rval\t' + show_progress(val_count, val_num)
            sys.stdout.write(progress)
            sys.stdout.flush
            # duration = time.time() - val_begin_at
            # throughput = val_count / duration
            # sys.stderr.write(
            #     '\rval   {} batches ({} samples) time: {} ({} images/sec)'
            #     .format(val_count / VAL_BATCHSIZE, val_count,
            #             datetime.timedelta(seconds=duration), throughput))

            val_loss += loss
            # if val_count == 50000:
            #     mean_loss = val_loss * VAL_BATCHSIZE / 50000
            #     print(file=sys.stderr)
            #     print(json.dumps({'type': 'val', 'iteration': train_count, 'loss': mean_loss}))
            #     sys.stdout.flush()

# Trainer


def train_loop():
    graph_generated = False
    while True:
        while data_q.empty():
            time.sleep(0.1)
        inp = data_q.get()
        if inp == 'end':  # quit
            res_q.put('end')
            break
        elif inp == 'train':  # restart training
            res_q.put('train')
            train = True
            continue
        elif inp == 'val':  # start validation
            res_q.put('val')
            train = False
            continue

        # x, y = inp
        # if args.gpu >= 0:
        #     x = cuda.to_gpu(x)
        #     y = cuda.to_gpu(y)

        x = xp.asarray(inp[0])
        y = xp.asarray(inp[1])

        if train:
            optimizer.zero_grads()
            loss = model.forward(x, y)
            loss.backward()
            optimizer.update()

            # if not graph_generated:
            #     with open('graph.dot', 'w') as o:
            #         o.write(c.build_computational_graph((loss,), False).dump())
            #     with open('graph.wo_split.dot', 'w') as o:
            #         o.write(c.build_computational_graph((loss,), True).dump())
            #     print('generated graph')
            #     graph_generated = True

        else:
            loss = model.forward(x, y, train=False)

        res_q.put(float(cuda.to_cpu(loss.data)))
        del loss, x, y

# Invoke threads
feeder = threading.Thread(target=feed_data)
feeder.daemon = True
feeder.start()
logger = threading.Thread(target=log_result)
logger.daemon = True
logger.start()

train_loop()
feeder.join()
logger.join()

# Save final model
pickle.dump(model, open(LOGPATH + 'model', 'wb'), -1)
