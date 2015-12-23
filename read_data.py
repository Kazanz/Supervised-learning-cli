import csv
import sys
from collections import OrderedDict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.svm import SVR

import settings


class SVRWorker(object):
    def __init__(self, csv_file=None, plot_file=None, dependent=0, ignore=[], kernels=[]):
        """Does SVR regression and plotting on csv data.

        :param csv_file: The name of the csv file to read.
        :param plot_file: The name of the file to save plots to.
        :param dependent: The index of the column of the dependent variable.
        :param ignore: Columns to ignore.
        :param kernels: List of dicts to pass as kwargs to `sklearn.svm.SVR`.
        """
        self.dependent = dependent
        self.csv_file = csv_file or settings.TRAINING_CSV
        self.plot_file = plot_file or settings.PLOT_FILE
        self.ignore = ignore
        self.train = self.target = None
        self.kernels = kernels
        self.prep_data()

    def prep_data(self):
        with open(self.csv_file, "r") as f:
            reader = csv.reader(f, delimiter=settings.DELIMITER)
            self.train_target_set(reader)

    def train_target_set(self, reader):
        self.train, self.target, self.target_label = self.format_csv_data(reader)

    def format_csv_data(self, reader):
        labels = next(reader)
        data_set = zip(*reader)
        target = np.array(data_set.pop(self.dependent))
        target_label = labels.pop(self.dependent)
        for i in sorted(self.ignore, reverse=True):
            if i > self.dependent:
                i -= 1
            del data_set[i]
            del labels[i]
        train = OrderedDict()
        for label, data in zip(labels, data_set):
            data = np.array(data)
            data.shape += (1,)
            train[label] = data
        return train, target, target_label

    def predictions(self, x):
        return ((kernel['kernel'], self.predict(x, kernel=kernel))
                for kernel in self.kernels)

    def set_kernel(self, **kwargs):
        self.kernel = SVR(**kwargs)

    def predict(self, val, kernel=None, **kwargs):
        print("Predicting: {}".format(kernel['kernel']))
        if kernel:
            self.set_kernel(**kernel)
        return self.kernel.fit(val, self.target).predict(val)

    def create_plots(self, predict=False):
        pdf = PdfPages(self.plot_file)
        for label, data in self.train.iteritems():
            figure = plt.figure(self.train.keys().index(label))
            self.plot(data, self.target, label, self.target_label, predict)
            pdf.savefig(figure.number)
        pdf.close()

    def plot(self, x, y, x_label=None, y_label=None, predict=False):
        print("Plotting: {}".format(x_label))
        plt.scatter(x, y, c='k', label='data')
        plt.hold('on')
        plt.xlabel(x_label or 'data')
        plt.ylabel(y_label or 'target')
        plt.title('Support Vector Regression')
        plt.legend()
        if predict:
            for kernel, prediction in self.predictions(x):
                plt.plot(x, prediction, c='g', label=kernel)


if __name__ == "__main__":
    start = datetime.now()
    dependent = int(sys.argv[1])
    ignore = map(lambda x: int(x), sys.argv[2:])
    kernels = [
        {'kernel': 'rbf', 'C': 1e3, 'gamma': 0.1},
        #{'kernel': 'linear', 'C': 1e3},
        #{'kernel': 'poly', 'C': 1e3, 'degree': 2},
    ]
    worker = SVRWorker(dependent=dependent, ignore=ignore, kernels=kernels)
    data = worker.create_plots(predict=True)
    print datetime.now() - start
