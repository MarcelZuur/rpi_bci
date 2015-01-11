import threading
import numpy as np
import time
from LEDPI import LEDPI
from network import bufhelp

#connect to buffer
bufhelp.connect()
print("connected")

#init stimuli
nSymbols = 4
nSequences = 6
nEpochs = 15
interEpochDelay = 1
stimulusDuration = 3
stimulusEventDelay = 0.1
stimulusFullDuration = 1
interSequenceDelay = 1
np.random.seed(0)
stimuli = np.random.randint(nSymbols, size=(nSequences, nEpochs)) #TODO: less random

#init frequencies
frequencies = np.zeros(nSymbols)
frequencies[0] = 13
frequencies[1] = 17
frequencies[2] = 21
frequencies[3] = 25
frequency_full = 100
leds=LEDPI([11, 13, 15])
t1 = threading.Thread(target=leds.blinkLED)
t1.start()
frequencies_led=frequencies[0:3].tolist()
leds.changeLED(frequencies_led)

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
            frequencies_led=frequencies[0:3].tolist()
            frequencies_led[symbol] = frequency_full
        leds.changeLED(frequencies_led)

        time.sleep(stimulusFullDuration)

        if symbol < 3:
            frequencies_led=frequencies[0:3].tolist()
        leds.changeLED(frequencies_led)
        time.sleep(stimulusEventDelay)
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