#!/usr/bin/python
import sys

from network import bufhelp, preproc


sys.path.append("../../dataAcq/buffer/python")
sys.path.append("../signalProc")
#import linear
import pickle
from sklearn import linear_model

#connect
bufhelp.connect()

#model init
logreg = linear_model.LogisticRegression()

#param
trlen_ms = 600

run = True
while run:
    print "Waiting for startPhase.cmd event."
    e = bufhelp.waitforevent("startPhase.cmd",1000, False)

    if e is not None:
        if e.value == "calibration":
            print "Calibration phase"
            data, events, stopevents = bufhelp.gatherdata("stimulus.visible",trlen_ms,("stimulus.training","end"), milliseconds=True)
            pickle.dump({"events":events,"data":data}, open("subject_data", "w"))

        elif e.value == "train":
            print "Training classifier"
            data = preproc.detrend(data)
            data, badch = preproc.badchannelremoval(data)
            data = preproc.spatialfilter(data)
            #data = preproc.spectrum(data, bufhelp.fSample, dim=1)
            #data = preproc.spectralfilter(data, (0, .1, 10, 12), bufhelp.fSample) #TODO: focus on the pre-defined frequencies.
            data, events, badtrials = preproc.badtrailremoval(data, events)
            mapping = {('stimulus.visible', '0'): 0, ('stimulus.visible', '1'): 1} #TODO: no mapping?, include all classes.
            logreg.fit(data,events) #TODO: proper format
            predictions = logreg.predict(data)
            #linear.fit(data,events,mapping)
            bufhelp.update()
            bufhelp.sendevent("sigproc.training","done")

        elif e.value =="testing": #TODO: check all, move different file?
            print "Feedback phase"
            while True:
                data, events, stopevents = bufhelp.gatherdata(["stimulus.columnFlash","stimulus.rowFlash"],trlen_ms,[("stimulus.feedback","end"), ("stimulus.sequence","end")], milliseconds=True)

                if isinstance(stopevents, list):
                    if any(map(lambda x: "stimulus.feedback" in x.type, stopevents)):
                        break
                else:
                    if "stimulus.feedback" in stopevents.type:
                        break

                data = preproc.detrend(data)
                data, badch = preproc.badchannelremoval(data)
                data = preproc.spatialfilter(data)
                data = preproc.spectralfilter(data, (0, .1, 10, 12), bufhelp.fSample)#TODO: check
                data2, events, badtrials = preproc.badtrailremoval(data, events)

                predictions = logreg.predict(data,events) #TODO: proper format
                #predictions = linear.predict(data)

                bufhelp.sendevent("classifier.prediction",predictions)

        elif e.value =="exit":
            run = False
