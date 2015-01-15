import StringIO
import numpy as np
import scipy
from sklearn import svm, preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import metrics
from network import preproc
import pickle


class Classifier():

    def __init__(self, fsample=250.0, channel_idxs=np.arange(0,10)):
        n_cv_folds = 3
        c_grid = [0.001, 0.01, 1, 10]
        model = LogisticRegression(C = 1.0, penalty = 'l1',  random_state=0)
        #model = SGDClassifier(loss='log', penalty='l1', random_state=0)
        #model = svm.SVC(kernel='linear', verbose=False, max_iter=1000000)
        self.clf = GridSearchCV(model, param_grid={'C': c_grid}, cv=n_cv_folds,
                           scoring = 'accuracy')
        self.events = None
        self.fsample = fsample
        self.optimal_model = None
        self.channel_idxs = channel_idxs

    def _preprocess(self, data, events=None):
        data = preproc.detrend(data)
        data, badch = preproc.badchannelremoval(data)
        for i in range(len(data)):
            data[i]=(preproc.spatialfilter(data[i]))

        data = preproc.fouriertransform(data, self.fsample)

        if events is not None:
            data, events, badtrials = preproc.badtrailremoval(data, events) #FIXME: check outlierdetection iterValue
            self.events = events
        return data, events

    def _convert_data(self, data):
        spectrum = preproc.spectrum(data, self.fsample)
        spectrum = np.array(spectrum)
        boolean_indexes = spectrum > 6
        indexes = np.array(np.where(boolean_indexes)[0]).astype(np.int32)

        X = np.array(data, dtype=np.float32)
        X = np.ascontiguousarray(X[:, indexes, :]) #FIXME: focus on the pre-defined frequencies?
        X = np.ascontiguousarray(X[:, :, self.channel_idxs])
        X.shape=(X.shape[0], X.shape[1] * X.shape[2])
        return X

    def _convert_events(self, events):
        y = np.zeros(len(events), dtype=np.int32)
        for i in range(len(events)):
            y[i] = events[i].value
        return y

    def train(self, data, events): #TODO: welch window
        data, events = self._preprocess(data, events)
        X = self._convert_data(data)
        y = self._convert_events(events)
        self.clf.fit(X,y)
        print(self.clf.best_params_, self.clf.best_score_)
        self.optimal_model = self.clf.best_estimator_
        return self.clf.best_score_

    def predict(self, data):
        data, events = self._preprocess(data)
        X = self._convert_data(data)
        return self.optimal_model.predict(X)

    def plot_model(self, data, classifier):
        from matplotlib import pyplot as plt
        for i in range(4):
            ax=plt.subplot(2, 2, i)
            plt.gray()
            spectrum = preproc.spectrum(data, classifier.fsample)
            spectrum = np.array(spectrum).astype(np.int32)
            #indexes = np.array(np.where(np.logical_and(spectrum > 9, spectrum < 29))[0]).astype(np.int32)
            plt.imshow(coefs_matrix[i].T, interpolation='nearest')
        plt.show()


if __name__ == '__main__':
    with open("subject2_data", "rb") as f:
        data_tuple = np.load(f)
        data = data_tuple['data']
        events = data_tuple['events']

    for i in range(1):
        print(i)
        classifier = Classifier()
        classifier.train(data[0:len(data)/2], events[0:len(data)/2])

    predictions = classifier.predict(data[len(data)/2:len(data)])
    print(predictions)

    #training accuracy
    y = classifier._convert_events(events[len(data)/2:len(data)])
    print(y)
    assert len(predictions)==len(y)
    print("test accuracy: ", np.sum(predictions==y)/float(y.shape[0]))

    data_array = np.array(data)

    with open("subject2_classifier", "wb") as f:
        pickle.dump({"model":classifier.optimal_model}, f)

    with open("subject2_classifier", "rb") as f:
        model = np.load(f)['model']

    assert (model.coef_==classifier.optimal_model.coef_).all()
    coefs_matrix=model.coef_.reshape(4, model.coef_.shape[1]/10, 10)
    classifier.plot_model(data, classifier)
