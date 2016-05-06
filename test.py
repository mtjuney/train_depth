from os.path import join, relpath
from glob import glob
path = 'data_train/train/x'
files = [relpath(x, path) for x in glob(join(path, '*'))]

print files
