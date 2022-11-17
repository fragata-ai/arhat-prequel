#
# Copyright 2019-2020 FRAGATA COMPUTER SYSTEMS AG
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# Based on neon, Intel(R) Nervana(tm) reference deep learning framework.
# Extracted from the original code and partly modified
# by FRAGATA COMPUTER SYSTEMS AG.
#

import os
import sys
import gzip
import tarfile

import numpy as np

from future.moves.urllib.request import Request, urlopen

PY3 = (sys.version_info[0] >= 3)

if not PY3:
    import cPickle as the_pickle
else:
    import pickle as the_pickle
pickle = the_pickle

def pickle_load(filepath):
    if PY3:
        return pickle.load(filepath, encoding='latin1')
    else:
        return pickle.load(filepath)

def fetch_dataset(url, sourcefile, destfile, totalsz):
    full_url = url + '/' + sourcefile
    print(full_url)
    req = Request(full_url, headers={'User-Agent': 'Arhat'})
    cloudfile = urlopen(req)
    with open(destfile, 'wb') as f:
        data_read = 0
        chunksz = 1024 ** 2
        while 1:
            data = cloudfile.read(chunksz)
            if not data:
                break
            data_read = min(totalsz, data_read + chunksz)
            f.write(data)

def load_data(url, filename, path, size):
     filepath = os.path.join(path, filename)
     fetch_dataset(url, filename, filepath, size)

def unpack_data(filename, path, normalize, contrast_normalize, whiten, output_fn):
    workdir = path
    filepath = os.path.join(path, filename)
    batchdir = os.path.join(workdir, 'cifar-10-batches-py')
    with tarfile.open(filepath, 'r:gz') as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, workdir)

    train_batches = [os.path.join(batchdir, 'data_batch_' + str(i)) for i in range(1, 6)]
    Xlist, ylist = [], []
    for batch in train_batches:
        with open(batch, 'rb') as f:
            d = pickle_load(f)
            Xlist.append(d['data'])
            ylist.append(d['labels'])

    X_train = np.vstack(Xlist)
    y_train = np.vstack(ylist)

    with open(os.path.join(batchdir, 'test_batch'), 'rb') as f:
        d = pickle_load(f)
        X_test, y_test = d['data'], d['labels']

    y_train = y_train.reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    if contrast_normalize:
        norm_scale = 55.0  # Goodfellow
        X_train = global_contrast_normalize(X_train, scale=norm_scale)
        X_test = global_contrast_normalize(X_test, scale=norm_scale)

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    if whiten:
        zca_cache = os.path.join(workdir, 'cifar-10-zca-cache.pkl')
        X_train, X_test = zca_whiten(X_train, X_test, cache=zca_cache)

    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    with open(os.path.join(workdir, output_fn), 'wb') as f:
        f.write(X_train)
        f.write(y_train)
        f.write(X_test)
        f.write(y_test)


def global_contrast_normalize(X, scale=1.0, min_divisor=1e-8):
    X = X - X.mean(axis=1)[:, np.newaxis]

    normalizers = np.sqrt((X ** 2).sum(axis=1)) / scale
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, np.newaxis]

    return X

def zca_whiten(train, test, cache=None):
    if cache and os.path.isfile(cache):
        with open(cache, 'rb') as f:
            (meanX, W) = pickle_load(f)
    else:
        meanX, W = compute_zca_transform(train)
        if cache:
            print("Caching ZCA transform matrix")
            with open(cache, 'wb') as f:
                pickle.dump((meanX, W), f, 2)

    print("Applying ZCA whitening transform")
    train_w = np.dot(train - meanX, W)
    test_w = np.dot(test - meanX, W)

    return train_w, test_w

def compute_zca_transform(imgs, filter_bias=0.1):
    print("Computing ZCA transform matrix")
    meanX = np.mean(imgs, 0)

    covX = np.cov(imgs.T)
    D, E = np.linalg.eigh(covX + filter_bias * np.eye(covX.shape[0], covX.shape[1]))

    assert not np.isnan(D).any()
    assert not np.isnan(E).any()
    assert D.min() > 0

    D = D ** -0.5

    W = np.dot(E, np.dot(np.diag(D), E.T))
    return meanX, W


url = 'http://www.cs.toronto.edu/~kriz'
filename = 'cifar-10-python.tar.gz'
path = '.'
size = 170498071

load_data(url, filename, path, size)
unpack_data(filename, path, True, False, False, 'cifar10_n.dat')
unpack_data(filename, path, False, True, True, 'cifar10_cw.dat')

