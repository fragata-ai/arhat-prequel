#
# Copyright 2019 FRAGATA COMPUTER SYSTEMS AG
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

from future.moves.urllib.request import Request, urlopen

PY3 = (sys.version_info[0] >= 3)

if not PY3:
    import cPickle as the_pickle
else:
    import pickle as the_pickle
pickle = the_pickle

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

def unpack_data(filename, path, output_fn):
    filepath = os.path.join(path, filename)

    with gzip.open(filepath, 'rb') as mnist:
        (X_train, y_train), (X_test, y_test) = pickle.load(mnist)

    with open(os.path.join(path, output_fn), 'wb') as f:
        f.write(X_train)
        f.write(y_train)
        f.write(X_test)
        f.write(y_test)

url = 'https://s3.amazonaws.com/img-datasets'
filename = 'mnist.pkl.gz'
path = '.'
size = 15296311

load_data(url, filename, path, size)
unpack_data(filename, path, 'mnist.dat')

