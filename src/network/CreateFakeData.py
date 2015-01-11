import threading
import numpy as np
import time
import bufhelp

#connect to buffer
bufhelp.connect()
print("connected")

#init stimuli
nSymbols = 4
nSequences = 4
nEpochs = 5
interEpochDelay = 1
stimulusDuration = 3
stimulusEventDelay = 1
interSequenceDelay = 1
np.random.seed(0)
stimuli = np.random.randint(nSymbols, size=(nSequences, nEpochs)) #TODO: less random

#init frequencies
frequencies = np.zeros(nSymbols)
frequencies[0] = 13
frequencies[1] = 17
frequencies[2] = 21
frequencies[3] = 25

#start stimuli
print("calibration START")
bufhelp.sendevent('startPhase.cmd', 'calibration')
time.sleep(2)
bufhelp.sendevent('stimulus.training', 'start')
for i in xrange(nSequences):
    for j in xrange(nEpochs):
        symbol = stimuli[i, j]
        frequencies_led=[0,0,0]
        if symbol < 3:
            frequencies_led[symbol] = frequencies[symbol]
        time.sleep(stimulusEventDelay)
        bufhelp.sendevent('stimulus.visible', symbol)
        time.sleep(stimulusDuration)
        frequencies_led=[0,0,0]
        bufhelp.sendevent('stimulus.epoch', 'end')
        time.sleep(interEpochDelay)
    bufhelp.sendevent('stimulus.sequence', 'end')
    time.sleep(interSequenceDelay)

bufhelp.sendevent('stimulus.training', 'end')
print("calibration END")