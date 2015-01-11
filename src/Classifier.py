import StringIO
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import metrics
from network import bufhelp, preproc
import pickle


class Classifier():

    def __init__(self):
        n_cv_folds = 3
        c_grid = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        #model = LogisticRegression(C = 1.0, penalty = 'l1',  random_state=0)
        model = SGDClassifier(loss='log', penalty='l1', random_state=0)
        self.clf = GridSearchCV(model, param_grid={'alpha': c_grid}, cv=n_cv_folds,
                           score_func = metrics.zero_one_loss)
        self.events = None

    def _preprocess(self, data, events=None):
        data = preproc.detrend(data)
        data, badch = preproc.badchannelremoval(data)
        for i in range(len(data)):
            data[i]=(preproc.spatialfilter(data[i]))
        if hasattr(bufhelp, 'fSample'):
            fsample = bufhelp.fSample
        else:
            fsample = 100.0
            print("OVERWRITING FSAMPLE")

        #data = preproc.spectralfilter(data, (0, .1, 10, 12), bufhelp.fSample) #TODO: focus on the pre-defined frequencies.
        data = preproc.fouriertransform(data, fsample)
        if events is not None:
            data, events, badtrials = preproc.badtrailremoval(data, events) #FIXME: check outlierdetection iterValue
            self.events = events
        #else:
            #data, badtrials = preproc.badtrailremoval(data)
        return data, events

    def _convert_data(self, data):
        X=np.array(data, dtype=np.float64)
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

    def predict(self, data):
            data, events = self._preprocess(data)
            X = self._convert_data(data)
            return self.clf.predict(X)


if __name__ == '__main__':
    classifier = Classifier()
    with open("subject2_data", "rb") as f:
        data_tuple = np.load(f)
        data = data_tuple['data']
        events = data_tuple['events']
    classifier.train(data[0:60], events[0:60])
    predictions = classifier.predict(data[61:90])
    print(predictions)

    #training accuracy
    y = classifier._convert_events(events[61:90])
    print(y)
    assert len(predictions)==len(y)
    print("test accuracy: ", np.sum(predictions==y)/float(y.shape[0]))
    print("test accuracy excluding stop: ", np.sum(predictions[predictions!=3]==y[predictions!=3])/float(y[predictions!=3].shape[0]))