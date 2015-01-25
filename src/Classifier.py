#!/usr/bin/python2.7
import pickle

import numpy as np
import scipy
import sklearn
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.signal import butter, lfilter
from matplotlib import ticker


#Base Classifier class. Contains most of the functions used for preprocessing data, training, predicting and validating.
from network import preproc


class Classifier(object):
    """Has functions for classifying EEG data. Functions take as input the data and events in the same format as gatherdata from
    bufhelp returns.

    Attributes:
      clf: The classifier.
      fsample(float): sample frequency of the eeg device used to record the data.
      channel_idxs(int[]): Array of used channels.
      optimal_model: The optimal model found through grid search.

    """
    def __init__(self, fsample, channel_idxs):
        """Constructs a classifier object which can be used to train and classify EEG data."""
        self.clf = None
        self.fsample = fsample
        self.channel_idxs = channel_idxs

    def convert_data(self, data):
        """Converts the given data into a compatible format that can be used by the classifier and performs preprocessing.
        For preprocessing the welch method is done with window size of 50% and an overlap of 50%, converting the time domain
        to the frequency domain. Then frequencies and unused channels are filtered out.

        """
        X = np.array(data, dtype=np.float32)
        freqs, X = scipy.signal.welch(X, fs=self.fsample, axis=1, scaling='spectrum', detrend="linear",
                                      window='hann', )
        self._freqs = freqs
        if(self.channel_idxs is not None):
            X = np.ascontiguousarray(X[:, :, self.channel_idxs])
        freqs_idxs = np.logical_and(freqs >= 3, freqs <=45 )
        freqs = freqs[freqs_idxs]
        X = np.ascontiguousarray(X[:, freqs_idxs, :])
        X.shape = (X.shape[0], X.shape[1] * X.shape[2])
        return X

    def convert_events(self, events):
        """Converts the given events into a compatible format that can be used by the classifier."""
        y = np.zeros(len(events), dtype=np.int32)
        for i in range(len(events)):
            y[i] = events[i].value
        return y

    def train(self, data, events):
        """Trains the classifier on the given data and events."""
        X = self.convert_data(data)
        Y = self.convert_events(events)
        self.clf.fit(X, Y)
        self.optimal_model = self.clf.best_estimator_

    def predict(self, data):
        """Returns a prediction from the given data."""
        X = self.convert_data(data)
        return self.clf.predict(X)


    def plot_model(self, data):
        """Plots the model in frequency domain for each class with x-axis displaying frequency and y-axis the channels."""
        from matplotlib import pyplot as plt
        coefs_matrix = self.optimal_model.coef_.reshape(3, self.optimal_model.coef_.shape[1] / len(classifier.channel_idxs),  len(classifier.channel_idxs))
        for i in range(3):
            ax = plt.subplot(3, 1, i+1)
            plt.gray()
            plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
            plt.imshow(coefs_matrix[i].T, interpolation='nearest')
            skip = 2
            plt.xticks(range(self._freqs.size), self._freqs[::skip].astype(np.int32)-2)
            loc = ticker.MultipleLocator(base=skip) # this locator puts ticks at regular intervals
            ax.xaxis.set_major_locator(loc)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('channels')
            if i ==0:
                plt.title('Left LED (11hz)')
            if i ==1:
                plt.title('Forward LED (13hz)')
            if i ==2:
                plt.title('Right LED (9hz)')
            plt.colorbar()
            plt.tight_layout()
        plt.show()

    def plot_data(self, data, events):
        """Plots the data in frequency domain for each class with x-axis displaying frequency and y-axis the power."""
        X = np.array(data, dtype=np.float32)
        if(self.channel_idxs is not None):
            X = np.ascontiguousarray(X[:, :, self.channel_idxs])
        y = self.convert_events(events)
        freqs, X = scipy.signal.welch(X, fs=self.fsample, axis=1, scaling='spectrum', detrend='linear',
                                      window='hann', )
        freqs_idxs = np.logical_and(freqs >= 7, freqs <= 28)
        freqs = freqs[freqs_idxs]
        X = np.ascontiguousarray(X[:, freqs_idxs, :])

        from matplotlib import pyplot as plt

        for i in range(3):
            X_avg = np.mean(X[y == i], axis=2)
            X_avg = np.mean(X_avg, axis=0)
            plt.subplot(1, 3, i+1)
            plt.semilogy(freqs, X_avg)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('Linear spectrum [V]')
            plt.title('Class '+str(i))
        plt.show()

    def validate(self,data,events):
        """Returns the accuracy of the classifier on the given data."""
        predictions = classifier.predict(data)
        print(predictions)
        y = classifier.convert_events(events)
        print(y)
        assert len(predictions) == len(y)
        print("test accuracy: ", np.sum(predictions == y) / float(y.shape[0]))


class LRClassifier(Classifier):
    """Extends Classifier, uses Logistic regression for classification."""
    def __init__(self, fsample=256.0, channel_idxs=np.arange(13, 17)):
        super(LRClassifier, self).__init__(fsample, channel_idxs)
        n_cv_folds = 3
        c_grid = [0.001, 0.01, 1, 10]
        model = LogisticRegression(C=1.0, penalty='l1', random_state=0)
        self.clf = GridSearchCV(model, param_grid={'C': c_grid}, cv=n_cv_folds,
                                scoring='accuracy')
        self.events = None


class SVMClassifier(Classifier):
    """Extends Classifier, uses Support vector machine for classification."""
    def __init__(self, fsample=256.0, channel_idxs=np.arange(9, 22)):
        super(SVMClassifier, self).__init__(fsample, channel_idxs)
        c_grid = [0.001, 0.01, 1, 10]
        self.clf= GridSearchCV(sklearn.svm.LinearSVC(penalty="l1",dual=False),param_grid={'C': c_grid},cv = 3, scoring="accuracy")

if __name__ == '__main__':
    with open("subject_data_fran_1", "rb") as f:
        data_tuple = np.load(f)
        data = data_tuple['data']
        events = data_tuple['events']

    #Test the SVM classifier
    classifier = SVMClassifier()
    classifier.plot_data(data, events)

    classifier.train(data[0:len(data) * 2/3], events[0:len(data) * 2/3])
    classifier.validate(data[len(data) * 2/3:len(data)],events[len(events)*2/3:len(events)])
    classifier.plot_model(data)
    #Test the logistic regression classifier
    classifier = LRClassifier()
    classifier.train(data[0:len(data) * 2/3], events[0:len(data) * 2/3])
    classifier.validate(data[len(data) * 2/3:len(data)],events[len(events)*2/3:len(events)])
    classifier.plot_model(data)
    data_array = np.array(data)

    with open("subject2_classifier", "wb") as f:
        pickle.dump({"model": classifier.clf}, f)

    with open("subject2_classifier", "rb") as f:
        model = np.load(f)['model']
