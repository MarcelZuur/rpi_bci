import blink
import numpy as np
import bufhelp

#connect to buffer
bufhelp.connect()

#init stimuli
nSymbols = 4
nSequences = 2
nEpochs = 4
stimuli = np.random.randint(nSymbols, size=(nSequences, nEpochs))

#init frequencies
frequencies = np.zeros(nSymbols)
frequencies[0] = 1
frequencies[1] = 2
frequencies[2] = 3
frequencies[3] = 4

#start stimuli
bufhelp.sendEvent('stimulus.training','start');
for i in xrange(nSequences):
    for j in xrange(nEpochs):
        symbol = stimuli[i, j]
        bufhelp.sendEvent('stimulus.showing' ,symbol)
        freq = frequencies[symbol]
        #blink.blink(symbol, duration, freq)
        bufhelp.sendEvent('stimulus.sequence','end');

bufhelp.sendEvent('stimulus.training','end');



