import os
import shutil
import random



path_x_train = 'data_train/train/x/'
path_y_train = 'data_train/train/y/'

path_x_test = 'data_train/test/x/'
path_y_test = 'data_train/test/y/'


file_list = os.listdir(path_x_train)

test_sample_num = len(file_list) / 4

test_file_list = random.sample(file_list, test_sample_num)

for test_file in test_file_list:
    shutil.move(path_x_train + test_file, path_x_test + test_file)
    shutil.move(path_y_train + test_file, path_y_test + test_file)

print 'move {} files'.format(test_sample_num)
