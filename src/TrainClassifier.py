#!/usr/bin/python2.7
"""Classifier training script.

This script sends an event to the buffer signalling SigProc to start training the classifier,
it then waits for SigProc to send an event to the buffer that it has finished before it exits.
SigProc has to have gathered data before from the Calibration phase to be able to train the classifier,
so CalibrationStimulus should have been run at least once before this.

"""

import ConfigParser
import time

from network import bufhelp


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

hostname = connectionOptions["hostname"]
port = int(connectionOptions["port"])
# connect to buffer
bufhelp.connect(adress=hostname, port=port)
print("connected")

time.sleep(1)
print("train classifier START")
bufhelp.sendevent('startPhase.cmd', 'train')
while True:
    print("wait for classifier")
    e = bufhelp.waitforevent(("sigproc.training", "end"), 1000, False)
    if e is not None:
        break
print("training END")
