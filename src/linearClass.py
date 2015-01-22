import numpy as np
from network import bufhelp, preproc
import sklearn.svm


def _convert_data(data):
    X = np.array(data, dtype=np.float64)
    X.shape = (X.shape[0], X.shape[1] * X.shape[2])
    return X

def _convert_events(events):
    y = np.zeros(len(events), dtype=np.int32)
    for i in range(len(events)):
        y[i] = events[i].value
    return y

def train(data, events):
    X = _convert_data(data)
    Y = _convert_events(events)
    lin_clf = sklearn.svm.LinearSVC()
    lin_clf.fit(X, Y)
    return lin_clf


with open("subject_data", "rb") as f:
    data_tuple = np.load(f)
    data = data_tuple['data']
    events = data_tuple['events']
    data, badch = preproc.badchannelremoval(data,range(23,47,1))
    data, badch = preproc.badchannelremoval(data,range(1,8,1))
    data = preproc.spectralfilter(data,[1,5,80,85], 250.0)
    clf = train(data[0:len(data)-30],events[0:len(data)-30])
    X2 = _convert_data(data[len(data)-30:len(data)])
    print clf.predict(X2)
    print _convert_events(events[len(data)-30:len(data)])



