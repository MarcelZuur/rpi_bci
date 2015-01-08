#import blink
import threading
import numpy as np

#connect to buffer
import time
from LEDPI import LEDPI
from network import bufhelp

bufhelp.connect()
print("connected")
#init stimuli
nSymbols = 4
nSequences = 2
nEpochs = 2
interEpochDelay = 1
stimulusDuration = 1
interSequenceDelay = 2
np.random.seed(0)
stimuli = np.random.randint(nSymbols, size=(nSequences, nEpochs)) #TODO: less random

#init frequencies
frequencies = np.zeros(nSymbols)
frequencies[0] = 15
frequencies[1] = 16
frequencies[2] = 17
frequencies[3] = 18
leds=LEDPI([11, 13, 15])
t1 = threading.Thread(target=leds.blinkLED)
t1.start()

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
        leds.changeLED(frequencies_led)
        time.sleep(stimulusDuration)
        bufhelp.sendevent('stimulus.visible', symbol)
        time.sleep(stimulusDuration)
        frequencies_led=[0,0,0]
        leds.changeLED(frequencies_led)
        bufhelp.sendevent('stimulus.epoch', 'end')
        time.sleep(interEpochDelay)
    bufhelp.sendevent('stimulus.sequence', 'end')
    time.sleep(interSequenceDelay)

bufhelp.sendevent('stimulus.training', 'end')
leds.blink = False
print("calibration END")

#time.sleep(1)
#print("train classifier")
#bufhelp.sendevent('startPhase.cmd', 'train')
#print("wait for classifier")
#bufhelp.waitnewevents(("sigproc.training","done"))
#bufhelp.sendevent('startPhase.cmd', 'exit')