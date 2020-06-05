import numpy as np
np.random.seed(1)
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    LeakyReLU,
    Dropout
)
from os import (
    path,
    listdir
)

# print(tf.config.experimental.list_physical_devices('GPU'))
# BASE = path.join(path.dirname(path.abspath(__file__)))


class KerasImgClassifier:
    def __init__(self, model_path, input_dim, epochs=15, nlabels=2):
        self.model_path = model_path
        self.epochs = epochs
        self.nlabels = nlabels
        self.model = None
        self.input_dim = input_dim

    def train(self, X, y, val_data=None, batch_size=None):
        self.model = self.build_model(input_dim=self.input_dim)
        self.model.fit(X, y,
                       epochs=self.epochs,
                       verbose=1,
                       batch_size=batch_size,
                       validation_data=val_data,
                       validation_steps=None,
                       steps_per_epoch=None)
        self.model.save_weights(self.model_path)

    def build_model(self, input_dim):
        model = Sequential()
        model.add(Conv2D(filters=32,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         input_shape=input_dim,
                         activation='linear'))
        model.add(LeakyReLU(alpha=1e-3))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=64,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         activation='linear'))
        model.add(LeakyReLU(alpha=1e-3))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        # model.add(Dense(units=128, activation='relu'))
        # model.add(Dense(units=128, activation='relu'))
        # model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=128, activation='linear'))
        model.add(LeakyReLU(alpha=1e-3))
        model.add(Dropout(0.5))
        model.add(Dense(units=self.nlabels, activation='softmax'))
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
        return model

    def classify(self, X, label_dict=None):
        y = self.predict(X)
        y = np.argmax(y, axis=1)
        labels = [
            label if label_dict is None else label_dict[label] for label in y
        ]
        return np.array(labels)

    def predict(self, X):
        if self.model is None:
            self.load_model(self.input_dim)
        return self.model.predict(X)

    def load_model(self, input_dim):
        self.model = self.build_model(input_dim)
        self.model.load_weights(self.model_path)


def main():
    model_path = './model/model.h5'
    model = KerasImgClassifier(model_path=model_path,
                               epochs=100,
                               input_dim=None,
                               nlabels=None)
    train(model_path, model)
    test(model_path, model)


def train(model_path, model):
    X = loadX('./database/train_data.csv')
    y = loady('./database/train_label.csv')

    X_val = loadX('./database/valid_data.csv')
    y_val = loady('./database/valid_label.csv')

    X = normalize_img_data(X)
    X_val = normalize_img_data(X_val)

    model.input_dim = X[0, :].shape
    model.nlabels = y.shape[1]
    model.train(X, y, val_data=(X_val, y_val), batch_size=None)


def test(model_path, model):
    print('load pretrained model: {}'.format(path.basename(model_path)))
    X_test = loadX('./database/test_data.csv')
    y_test = loady('./database/test_label.csv')

    X_test = normalize_img_data(X_test)

    model.input_dim = X_test[0, :].shape
    model.nlabels = y_test.shape[1]
    y_pred = model.classify(X_test)

    calculate_accuracy(np.argmax(y_test, axis=1), y_pred)
    y_test = np.argmax(y_test, axis=1)
    true = [y_pred[i] for i in range(y_pred.shape[0]) if y_pred[i] == y_test[i]]
    false = [y_test[i] for i in range(y_test.shape[0]) if y_pred[i] != y_test[i]]
    print('true:', len(true))
    print('false:', len(false))

    for i in range(10):
        print('true {}: {}'.format(i, len([j for j in true if j == i])))
    for i in range(10):
        print('false {}: {}'.format(i, len([j for j in false if j == i])))


def loadX(file_name):
    np_imgs = np.loadtxt(file_name, dtype=float, delimiter=',')
    np_imgs = np.asarray([onedto2d(row, 28) for row in np_imgs])
    X = np_imgs.reshape(np_imgs.shape + (1,))
    print('{} shape = {}'.format(path.basename(file_name), X.shape))
    return X


def loady(file_name):
    label_data = np.loadtxt(file_name, dtype='uint8', delimiter=',')
    labels = np.unique(label_data)
    y = []
    for label in label_data:
        yi = np.zeros((labels.shape[0]))
        yi[label] = 1
        y.append(yi)
    y = np.array(y)
    print('{} shape = {}'.format(path.basename(file_name), y.shape))
    return y


def normalize_img_data(np_img):
    return np_img / 255


def calculate_accuracy(y, pred):
    true = np.array([i for i in range(y.shape[0]) if (y[i] == pred[i]).all()])
    print('accuracy: {0:.4f}'.format(true.shape[0] / y.shape[0]))


def onedto2d(a, shape1):
    return np.reshape(a, (-1, shape1))


if __name__ == "__main__":
    main()
