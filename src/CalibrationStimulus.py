#import blink
import numpy as np

#connect to buffer
import time
from network import bufhelp

bufhelp.connect()
print("connected")
#init stimuli
nSymbols = 4
nSequences = 2
nEpochs = 2
interEpochDelay = 1
interSequenceDelay = 2
stimuli = np.random.randint(nSymbols, size=(nSequences, nEpochs)) #TODO: less random

#init frequencies
frequencies = np.zeros(nSymbols)
frequencies[0] = 1
frequencies[1] = 2
frequencies[2] = 3
frequencies[3] = 4

#start stimuli
print("calibration START")
bufhelp.sendevent('startPhase.cmd', 'calibration')
time.sleep(2)
bufhelp.sendevent('stimulus.training', 'start')
for i in xrange(nSequences):
    for j in xrange(nEpochs):
        symbol = stimuli[i, j]
        bufhelp.sendevent('stimulus.visible', symbol)
        freq = frequencies[symbol]
        #blink.blink(symbol, duration, freq)
        bufhelp.sendevent('stimulus.epoch', 'end')
        time.sleep(interEpochDelay)
    bufhelp.sendevent('stimulus.sequence', 'end')
    time.sleep(interSequenceDelay)

bufhelp.sendevent('stimulus.training', 'end')
print("calibration END")

time.sleep(1)
print("train classifier")
bufhelp.sendevent('startPhase.cmd', 'train')
print("wait for classifier")
#bufhelp.waitnewevents(("sigproc.training","done"))
#bufhelp.sendevent('startPhase.cmd', 'exit')