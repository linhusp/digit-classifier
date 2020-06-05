import numpy as np
from os import (
    path,
    listdir,
)
from PIL import Image
import csv

BASE = path.join(path.dirname(path.abspath(__file__)), 'database')


def main():
    img_dirs = [path.join(BASE, dir_) for dir_ in listdir(BASE)]
    print(img_dirs)

    train_data = 'train_data.csv'
    train_label = 'train_label.csv'
    test_data = 'test_data.csv'
    test_label = 'test_label.csv'

    for dir_ in img_dirs:
        imgs = [path.join(dir_, img_name) for img_name in listdir(dir_)]

        output_file = dir_ + '_data.csv'
        label_file = dir_ + '_label.csv'
        
        for img in imgs:
            np_img = load_img(img)
            save_data(np_img, output_file)
            print('saved img {} to {}'.format(img, dir_))

        labels = np.array([path.basename(img)[0] for img in imgs])
        save_data(labels, label_file)
        print('saved label to {}'.format(label_file))


def load_img(file_name):
    img = Image.open(file_name)
    img.load()
    return np.asarray(img, dtype='int32')


def save_data(np_data, output_name):
    with open(output_name, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(ndto1d(np_data))


def ndto1d(a):
    return np.reshape(a, (-1))


if __name__ == "__main__":
    main()
