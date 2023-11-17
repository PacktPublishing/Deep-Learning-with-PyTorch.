#!/usr/bin/env python
import os
import struct
import sys
import numpy
import subprocess
from array import array
from os import path
from PIL import Image #imported from pillow


# funtion to read the MNIST dataset 
def read(dataset):
    if dataset is "training":

        fname_img = "train-images-idx3-ubyte"
        fname_lbl = "train-labels-idx1-ubyte"

    elif dataset is "testing":

        fname_img = "t10k-images-idx3-ubyte"
        fname_lbl = "t10k-labels-idx1-ubyte"

    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    return lbl, img, size, rows, cols

# funtion to extract and  the MNIST dataset 
def write_dataset(labels, data, size, rows, cols, output_dir):

    output_dirs = [
        path.join(output_dir, str(i))
        for i in range(10)
    ]
    for dir in output_dirs:
        if not path.exists(dir):
            os.makedirs(dir)

    # write data
    for (i, label) in enumerate(labels):
        output_filename = path.join(output_dirs[label], str(i) + ".jpg")
        print("writing " + output_filename)

        with open(output_filename, "wb") as h:
            data_i = [
                data[ (i*rows*cols + j*cols) : (i*rows*cols + (j+1)*cols) ]
                for j in range(rows)
            ]
            data_array = numpy.asarray(data_i)


            im = Image.fromarray(data_array)
            im.save(output_filename)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: {0} <output_path>".format(sys.argv[0]))
        print("Please create a folder named Output in the same location as the script")

        sys.exit()

    output_path = sys.argv[1]
    # Download the Dataset from the source
    p = subprocess.Popen('wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', shell=True)
    p.wait()
    p = subprocess.Popen('wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', shell=True)
    p.wait()
    p = subprocess.Popen('wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', shell=True)
    p.wait()
    p = subprocess.Popen('wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', shell=True)
    p.wait()

    # Unzip Downloaded Dataset
    p = subprocess.Popen('gunzip train-images-idx3-ubyte.gz', shell=True)
    p.wait()
    p = subprocess.Popen('gunzip train-labels-idx1-ubyte.gz', shell=True)
    p.wait()
    p = subprocess.Popen('gunzip t10k-images-idx3-ubyte.gz', shell=True)
    p.wait()
    p = subprocess.Popen('gunzip t10k-labels-idx1-ubyte.gz', shell=True)
    p.wait()

    for dataset in ["training", "testing"]:
        labels, data, size, rows, cols = read(dataset)
        write_dataset(labels, data, size, rows, cols,
                      path.join(output_path, dataset))
