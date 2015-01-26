import pickle

from matplotlib import ticker
import numpy as np
import scipy
from scipy import signal
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression


class Classifier():
    def __init__(self, fsample=256.0, channel_idxs=np.arange(9, 22)):
        n_cv_folds = 3
        c_grid = [0.001, 0.01, 1, 100]
        model = LogisticRegression(C=1.0, penalty='l1', random_state=0)
        # model = svm.SVC(kernel='linear', verbose=False, max_iter=10000000)
        self.clf = GridSearchCV(model, param_grid={'C': c_grid}, cv=n_cv_folds,
                                scoring='accuracy')
        self.events = None
        self.fsample = fsample
        self.optimal_model = None
        self.channel_idxs = channel_idxs
        self._freqs = None

    def _convert_data(self, data):
        X = np.array(data, dtype=np.float32)
        X = np.ascontiguousarray(X[:, :, self.channel_idxs])
        freqs, X = scipy.signal.welch(X, fs=self.fsample, axis=1, scaling='spectrum', detrend='linear',
                                      nperseg=X.shape[1] / 2, window='hanning', )
        freqs_idxs = np.logical_and(freqs >= 7, freqs <= 28)
        freqs = freqs[freqs_idxs]
        self._freqs = freqs
        X = np.ascontiguousarray(X[:, freqs_idxs, :])
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
        self.clf.fit(X, y)
        print(self.clf.best_params_, self.clf.best_score_)
        self.optimal_model = self.clf.best_estimator_
        return self.clf.best_score_

    def predict(self, data):
        X = self._convert_data(data)
        return self.optimal_model.predict(X)

    def plot_model(self, data):
        from matplotlib import pyplot as plt

        coefs_matrix = self.optimal_model.coef_.reshape(3, model.coef_.shape[1] / len(classifier.channel_idxs),
                                                        len(classifier.channel_idxs))
        for i in range(3):
            ax = plt.subplot(3, 1, i + 1)
            plt.gray()
            plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
            plt.imshow(coefs_matrix[i].T, interpolation='nearest')
            skip = 2
            plt.xticks(range(self._freqs.size), self._freqs[::skip].astype(np.int32) - 2)
            loc = ticker.MultipleLocator(base=skip)  # this locator puts ticks at regular intervals
            ax.xaxis.set_major_locator(loc)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('channels')
            if i == 0:
                plt.title('Left LED (11hz)')
            if i == 1:
                plt.title('Forward LED (13hz)')
            if i == 2:
                plt.title('Right LED (9hz)')
            plt.colorbar()
            plt.tight_layout()
        plt.show()

    def plot_data(self, data, events):
        X = np.array(data, dtype=np.float32)
        X = np.ascontiguousarray(X[:, :, self.channel_idxs])
        y = self._convert_events(events)
        freqs, X = scipy.signal.welch(X, fs=self.fsample, axis=1, scaling='spectrum', detrend='linear',
                                      window='hanning', )
        freqs_idxs = np.logical_and(freqs >= 7, freqs <= 45)
        freqs = freqs[freqs_idxs]
        X = np.ascontiguousarray(X[:, freqs_idxs, :])
        from matplotlib import pyplot as plt

        for i in range(3):
            X_avg = np.mean(X[y == i], axis=2)
            X_avg = np.mean(X_avg, axis=0)
            ax = plt.subplot(1, 3, i + 1)
            plt.plot(freqs, X_avg)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('Linear spectrum [V]')
            if i == 0:
                plt.title('Left LED (11hz)')
            if i == 1:
                plt.title('Forward LED (13hz)')
            if i == 2:
                plt.title('Right LED (9hz)')
            ax.set_ylim([0, 3])
        plt.show()


    def plot_data2(self, data, data2, events, events2):
        X = np.array(data, dtype=np.float32)
        X = np.ascontiguousarray(X[:, :, self.channel_idxs])
        y = self._convert_events(events)
        freqs, X = scipy.signal.welch(X, fs=self.fsample, axis=1, scaling='spectrum', detrend='linear',
                                      window='hanning', )
        freqs_idxs = np.logical_and(freqs >= 7, freqs <= 45)
        freqs = freqs[freqs_idxs]
        X = np.ascontiguousarray(X[:, freqs_idxs, :])

        X2 = np.array(data2, dtype=np.float32)
        X2 = np.ascontiguousarray(X2[:, :, self.channel_idxs])
        y2 = self._convert_events(events2)
        freqs2, X2 = scipy.signal.welch(X2, fs=self.fsample, axis=1, scaling='spectrum', detrend='linear',
                                        window='hanning', )
        X2 = np.ascontiguousarray(X2[:, freqs_idxs, :])
        from matplotlib import pyplot as plt

        for i in range(3):
            X_avg = np.mean(X[y == i], axis=2)
            X_avg = np.mean(X_avg, axis=0)
            X2_avg = np.mean(X2[y2 == i], axis=2)
            X2_avg = np.mean(X2_avg, axis=0)
            ax = plt.subplot(3, 1, i + 1)
            plt.plot(freqs, X_avg, 'b', freqs, X2_avg, 'g', linewidth=1.5)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('Linear spectrum [V]')
            if i == 0:
                plt.title('Left LED (11hz)')
            if i == 1:
                plt.title('Forward LED (13hz)')
            if i == 2:
                plt.title('Right LED (9hz)')
            ax.set_ylim([0, 3.0])
            ax.legend(['close', 'far'])
        plt.show()


if __name__ == '__main__':
    with open("data2\\subject_data_marcel_table", "rb") as f:
        data_tuple = np.load(f)
        data = data_tuple['data']
        events = data_tuple['events']

    with open("data2\\subject_data_marcel_floor", "rb") as f:
        data_tuple = np.load(f)
        data2 = data_tuple['data']
        events2 = data_tuple['events']

    for i in range(1):
        print(i)
        classifier = Classifier()
        classifier.plot_data2(data, data2, events, events2)
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
    classifier.plot_model(data)
