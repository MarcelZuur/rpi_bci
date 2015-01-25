import pickle

import numpy as np
import scipy
import sklearn.svm
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.signal import butter, lfilter
from matplotlib import ticker

from network import preproc


class Classifier(object):
    def __init__(self, fsample, channel_idxs):
        self.clf = None
        self.fsample = fsample
        self.channel_idxs = channel_idxs

    def convert_data(self, data):
        X = np.array(data, dtype=np.float32)
        freqs, X = scipy.signal.welch(X, fs=self.fsample, axis=1, scaling='spectrum', detrend="linear",
                                      window='hann', )
        X = self.butter_bandpass_filter(X,7,45,self.fsample,order=1)
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

    def plot_model(self, data, coefs_matrix):
        from matplotlib import pyplot as plt

        for i in range(3):
            ax = plt.subplot(1, 3, i+1)
            plt.gray()
            plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
            plt.imshow(coefs_matrix[i].T, interpolation='nearest')
            skip = 2
            plt.xticks(range(self._freqs.size), self._freqs[::skip].astype(np.int32))
            loc = ticker.MultipleLocator(base=skip) # this locator puts ticks at regular intervals
            ax.xaxis.set_major_locator(loc)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('channels')
            plt.title('Class '+str(i))
        plt.show()

    def plot_data(self, data, events):
        X = np.array(data, dtype=np.float32)
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

    def butter_bandpass(self,lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a


    def butter_bandpass_filter(self,data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def validate(self,data,events):
        predictions = classifier.predict(data)
        print(predictions)
        y = classifier.convert_events(events)
        print(y)
        assert len(predictions) == len(y)
        print("test accuracy: ", np.sum(predictions == y) / float(y.shape[0]))


class LRClassifier(Classifier):
    def __init__(self, fsample=256.0, channel_idxs=np.arange(12, 17)):
        super(LRClassifier, self).__init__(fsample, channel_idxs)
        n_cv_folds = 3
        c_grid = [0.001, 0.01, 1, 10]
        model = LogisticRegression(C=1.0, penalty='l1', random_state=0)
        self.clf = GridSearchCV(model, param_grid={'C': c_grid}, cv=n_cv_folds,
                                scoring='accuracy')
        self.events = None


class SVMClassifier(Classifier):
    def __init__(self, fsample=256.0, channel_idxs=np.arange(12, 17)):
        super(SVMClassifier, self).__init__(fsample, channel_idxs)
        self.clf = sklearn.svm.LinearSVC()


if __name__ == '__main__':
    with open("subject_data_marcel1", "rb") as f:
        data_tuple = np.load(f)
        data = data_tuple['data']
        events = data_tuple['events']

    #Test the SVM classifier
    classifier = SVMClassifier()
    classifier.plot_data(data, events)
    
    classifier.train(data[0:len(data) * 2/3], events[0:len(data) * 2/3])
    classifier.validate(data[len(data) * 2/3:len(data)],events[len(events)*2/3:len(events)])
    #Test the logistic regression classifier
    classifier = LRClassifier()
    classifier.train(data[0:len(data) * 2/3], events[0:len(data) * 2/3])
    classifier.validate(data[len(data) * 2/3:len(data)],events[len(events)*2/3:len(events)])

    data_array = np.array(data)

    with open("subject2_classifier", "wb") as f:
        pickle.dump({"model": classifier.clf}, f)

    with open("subject2_classifier", "rb") as f:
        model = np.load(f)['model']
