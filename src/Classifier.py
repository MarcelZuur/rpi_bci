import pickle

import numpy as np
import scipy
import sklearn.svm
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

from network import preproc


class Classifier(object):
    def __init__(self, fsample=250.0, channel_idxs=np.arange(9, 22)):
        self.clf = None
        self.fsample = fsample
        self.channel_idxs = channel_idxs

    def convert_data(self, data):
        X = np.array(data, dtype=np.float32)
        freqs, X = scipy.signal.welch(X, fs=self.fsample, axis=1, scaling='spectrum', detrend='linear',
                                      window='hanning', )
        freqs_idxs = np.logical_and(freqs > 1, freqs < 80)
        X = np.ascontiguousarray(X[:, freqs_idxs, :])
        X = np.ascontiguousarray(X[:, :, self.channel_idxs])
        X.shape = (X.shape[0], X.shape[1] * X.shape[2])
        return X

    def convert_events(self, events):
        y = np.zeros(len(events), dtype=np.int32)
        for i in range(len(events)):
            y[i] = events[i].value
        return y

    def train(self, data, events):
        X = self.convert_data(data)
        Y = self.convert_events(events)
        self.clf.fit(X, Y)

    def predict(self, data):
        X = self.convert_data(data)
        return self.clf.predict(X)

    def plot_model(self, data, classifier):
        from matplotlib import pyplot as plt

        for i in range(4):
            ax = plt.subplot(2, 2, i)
            plt.gray()
            spectrum = preproc.spectrum(data, classifier.fsample)
            spectrum = np.array(spectrum).astype(np.int32)
            # indexes = np.array(np.where(np.logical_and(spectrum > 9, spectrum < 29))[0]).astype(np.int32)
            plt.imshow(coefs_matrix[i].T, interpolation='nearest')
        plt.show()

    def validate(self,data,events):
        predictions = classifier.predict(data)
        print(predictions)
        y = classifier.convert_events(events)
        print(y)
        assert len(predictions) == len(y)
        print("test accuracy: ", np.sum(predictions == y) / float(y.shape[0]))

class LRClassifier(Classifier):
    def __init__(self, fsample=250.0, channel_idxs=np.arange(9, 22)):
        super(LRClassifier, self).__init__(fsample, channel_idxs)
        n_cv_folds = 3
        c_grid = [0.001, 0.01, 1, 10]
        model = LogisticRegression(C=1.0, penalty='l1', random_state=0)
        self.clf = GridSearchCV(model, param_grid={'C': c_grid}, cv=n_cv_folds,
                                scoring='accuracy')
        self.events = None


class SVMClassifier(Classifier):
    def __init__(self, fsample=250.0, channel_idxs=np.arange(9, 22)):
        super(SVMClassifier, self).__init__(fsample, channel_idxs)
        self.clf = sklearn.svm.LinearSVC()


if __name__ == '__main__':
    with open("subject_data_marcel2101_2", "rb") as f:
        data_tuple = np.load(f)
        data = data_tuple['data']
        events = data_tuple['events']

    #Test the SVM classifier
    classifier = SVMClassifier()
    #classifier.train(data[0:len(data) / 2], events[0:len(data) / 2])
    classifier.train(data[0:len(data)-30],events[0:len(data)-30])
    classifier.validate(data[len(data)-30:len(data)],events[len(data)-30:len(data)])
    #predictions = classifier.predict(data[len(data) / 2:len(data)])

    #Test the logistic regression classifier
    classifier = LRClassifier()
    #   classifier.train(data[0:len(data) / 2], events[0:len(data) / 2])
    #predictions = classifier.predict(data[len(data) / 2:len(data)])
    classifier.train(data[0:len(data)-30],events[0:len(data)-30])
    classifier.validate(data[len(data)-30:len(data)],events[len(data)-30:len(data)])

    data_array = np.array(data)

    with open("subject2_classifier", "wb") as f:
        pickle.dump({"model": classifier.clf}, f)

    with open("subject2_classifier", "rb") as f:
        model = np.load(f)['model']

    assert (model.coef_ == classifier.optimal_model.coef_).all()
    coefs_matrix = model.coef_.reshape(4, model.coef_.shape[1] / 10, 10)
    classifier.plot_model(data, classifier)
