#!/usr/bin/python
import sys

from network import bufhelp
from Classifier import Classifier
sys.path.append("../../dataAcq/buffer/python")
sys.path.append("../signalProc")
import pickle
import numpy as np

#connect
bufhelp.connect()

#model init
classifier = Classifier(250)

#param
trlen_ms = 2000

run = True
while run:
    print "Waiting for startPhase.cmd event."
    e = bufhelp.waitforevent("startPhase.cmd", 1000, False)

    if e is not None:
        if e.value == "calibration":
            print "Calibration phase"
            data, events, stopevents = bufhelp.gatherdata("stimulus.visible",trlen_ms,("stimulus.training","end"), milliseconds=True)
            print "end"
            with open("subject_data", "wb") as f:
                pickle.dump({"events":events,"data":data}, f)

        elif e.value == "train":
            print "Training classifier"
            with open("subject_data", "rb") as f:
                data_tuple = np.load(f)
                data = data_tuple['data']
                events = data_tuple['events']

            classifier.train(data,events)

            with open("subject_classifier", "wb") as f:
                pickle.dump({"model":classifier.optimal_model}, f)

            bufhelp.update()
            bufhelp.sendevent("sigproc.training","end")

        elif e.value =="feedback":
            print "Feedback phase"
            while True:
                data, events, stopevents = bufhelp.gatherdata("robot.start", trlen_ms,[("robot.stop",1), ("stimulus.feedback","end")], milliseconds=True, time_trigger=True)

                if isinstance(stopevents, list):
                    if any(map(lambda x: "stimulus.feedback" in x.type, stopevents)):
                        bufhelp.sendevent("sigproc.feedback","end")
                        break
                else:
                    if "stimulus.feedback" in stopevents.type:
                        bufhelp.sendevent("sigproc.feedback","end")
                        break

                print "get prediction"
                if classifier.optimal_model is None:
                    with open("subject_classifier", "rb") as f:
                        classifier.optimal_model = np.load(f)['model']

                if data:
                    predictions = classifier.predict(data).tolist()
                    bufhelp.update()
                    bufhelp.sendevent("classifier.predictions", predictions)
                else:
                    print("ERROR: NO DATA")

        elif e.value =="exit":
            bufhelp.sendevent("sigproc","exit")
            run = False
