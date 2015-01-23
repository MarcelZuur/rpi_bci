import pickle

import numpy as np
import scipy
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

from network import preproc


class Classifier():
    def __init__(self, fsample=250.0, channel_idxs=np.arange(9, 22)):
        n_cv_folds = 3
        c_grid = [0.001, 0.01, 1, 10]
        model = LogisticRegression(C=1.0, penalty='l1', random_state=0)
        # model = SGDClassifier(loss='log', penalty='l1', random_state=0)
        #model = svm.SVC(kernel='linear', verbose=False, max_iter=1000000)
        self.clf = GridSearchCV(model, param_grid={'C': c_grid}, cv=n_cv_folds,
                                scoring='accuracy')
        self.events = None
        self.fsample = fsample
        self.optimal_model = None
        self.channel_idxs = channel_idxs

    def _convert_data(self, data):
        X = np.array(data, dtype=np.float32)
        freqs, X = scipy.signal.welch(X, fs=self.fsample, axis=1, scaling='spectrum', detrend='linear',
                                      window='hanning', )
        freqs_idxs = freqs > 0

        X = np.ascontiguousarray(X[:, freqs_idxs, :])
        X = np.ascontiguousarray(X[:, :, self.channel_idxs])

        X.shape = (X.shape[0], X.shape[1] * X.shape[2])
        return X

    def _convert_events(self, events):
        y = np.zeros(len(events), dtype=np.int32)
        for i in range(len(events)):
            y[i] = events[i].value
        return y

    def train(self, data, events):
        X = self._convert_data(data)
        y = self._convert_events(events)
        X = X[y != 3]
        y = y[y != 3]
        self.clf.fit(X, y)
        print(self.clf.best_params_, self.clf.best_score_)
        self.optimal_model = self.clf.best_estimator_
        return self.clf.best_score_

    def predict(self, data):
        X = self._convert_data(data)
        return self.optimal_model.predict(X)

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

    def plot_data(self, data, events):
        X = np.array(data, dtype=np.float32)
        y = self._convert_events(events)
        freqs, X = scipy.signal.welch(X, fs=self.fsample, axis=1, scaling='spectrum', detrend='linear',
                                      window='hanning', )
        freqs_idxs = np.logical_and(freqs > 7, freqs < 15)
        freqs = freqs[freqs_idxs]
        X = np.ascontiguousarray(X[:, freqs_idxs, :])
        X = np.ascontiguousarray(X[:, :, self.channel_idxs])
        from matplotlib import pyplot as plt

        for i in range(3):
            X_avg = np.mean(X[y == i], axis=2)
            X_avg = np.mean(X_avg, axis=0)
            plt.subplot(2, 2, i)
            plt.semilogy(freqs, np.sqrt(X_avg))
            plt.xlabel('frequency [Hz]')
            plt.ylabel('Linear spectrum [V RMS]')
        plt.show()


if __name__ == '__main__':
    with open("data\\subject_data_marcel_monday2", "rb") as f:
        data_tuple = np.load(f)
        data = data_tuple['data']
        events = data_tuple['events']

    for i in range(1):
        print(i)
        classifier = Classifier()
        classifier.plot_data(data, events)
        classifier.train(data[0:len(data) / 2], events[0:len(data) / 2])

    predictions = classifier.predict(data[len(data) / 2:len(data)])
    print(predictions)

    # training accuracy
    y = classifier._convert_events(events[len(data) / 2:len(data)])
    print(y)
    assert len(predictions) == len(y)
    print("test accuracy: ", np.sum(predictions == y) / float(y.shape[0]))

    data_array = np.array(data)

    with open("subject2_classifier", "wb") as f:
        pickle.dump({"model": classifier.optimal_model}, f)

    with open("subject2_classifier", "rb") as f:
        model = np.load(f)['model']

    assert (model.coef_ == classifier.optimal_model.coef_).all()
    coefs_matrix = model.coef_.reshape(4, model.coef_.shape[1] / 10, 10)
    classifier.plot_model(data, classifier)
