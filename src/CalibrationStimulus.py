#import blink
import numpy as np

#connect to buffer
from network import bufhelp

bufhelp.connect()
print("connected")
#init stimuli
nSymbols = 4
nSequences = 2
nEpochs = 4
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
bufhelp.sendevent('stimulus.training', 'start')
for i in xrange(nSequences):
    for j in xrange(nEpochs):
        symbol = stimuli[i, j]
        bufhelp.sendevent('stimulus.visible', symbol)
        freq = frequencies[symbol]
        #blink.blink(symbol, duration, freq)
        bufhelp.sendevent('stimulus.epoch', 'end')
    bufhelp.sendevent('stimulus.sequence', 'end')

bufhelp.sendevent('stimulus.training', 'end')
print("calibration END")


print("train classifier")
bufhelp.sendevent('startPhase.cmd', 'train')
print("wait for classifier")
bufhelp.waitnewevents(("sigproc.training","done"))
