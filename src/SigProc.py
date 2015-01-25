#!/usr/bin/python2.7
"""Signal processing script.

This script takes care of the signal processing for the calibration phase, classifier training phase
and the feedback phase. It listens to the buffer for an event indicating which phase should be started.
In the calibration phase it gathers and labels data. In the classifier training phase it trains the
classifier. In the feedback phase it gathers data and generates predictions using the trained classifier,
which then get send to the buffer.


Attributes:
  classifier(Classifier): The classifier that is trained and used for generating predictions.
  trlen_ms(int): Duration for which data is gathered in ms.
"""
import sys
import ConfigParser
import json

from network import bufhelp
from Classifier import SVMClassifier, LRClassifier


sys.path.append("../../dataAcq/buffer/python")
sys.path.append("../signalProc")
import pickle
import numpy as np

Config = ConfigParser.ConfigParser()
Config.read("settings.ini")


def ConfigSectionMap(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1


connectionOptions = ConfigSectionMap("Connection")
# connect to buffer
hostname = connectionOptions["hostname"]
port = int(connectionOptions["port"])
bufhelp.connect(adress=hostname, port=port)

#model init
sigProcOption = ConfigSectionMap("SignalProcessing")
fsample = float(sigProcOption["fsample"])
channels = freqs = json.loads(sigProcOption["channels"])
if (sigProcOption["classifier"].lower() == "lr"):
    classifier = LRClassifier(fsample, channels)
else:
    classifier = SVMClassifier(fsample, channels)

#param
trlen_ms = 2000

run = True
while run:
    print "Waiting for startPhase.cmd event."
    e = bufhelp.waitforevent("startPhase.cmd", 1000, False)

    if e is not None:
        if e.value == "calibration":
            print "Calibration phase"
            data, events, stopevents = bufhelp.gatherdata("stimulus.visible", trlen_ms, ("stimulus.training", "end"),
                                                          milliseconds=True)
            print "end"
            with open("subject_data", "wb") as f:
                pickle.dump({"events": events, "data": data}, f)

        elif e.value == "train":
            print "Training classifier"
            with open("subject_data", "rb") as f:
                data_tuple = np.load(f)
                data = data_tuple['data']
                events = data_tuple['events']

            classifier.train(data, events)

            with open("subject_classifier", "wb") as f:
                pickle.dump({"model": classifier.optimal_model}, f)

            bufhelp.update()
            bufhelp.sendevent("sigproc.training", "end")

        elif e.value == "feedback":
            print "Feedback phase"
            while True:
                data, events, stopevents = bufhelp.gatherdata("robot.start", trlen_ms,
                                                              [("robot.stop", 1), ("stimulus.feedback", "end")],
                                                              milliseconds=True, time_trigger=True)

                if isinstance(stopevents, list):
                    if any(map(lambda x: "stimulus.feedback" in x.type, stopevents)):
                        bufhelp.sendevent("sigproc.feedback", "end")
                        break
                else:
                    if "stimulus.feedback" in stopevents.type:
                        bufhelp.sendevent("sigproc.feedback", "end")
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

        elif e.value == "exit":
            bufhelp.sendevent("sigproc", "exit")
            run = False
