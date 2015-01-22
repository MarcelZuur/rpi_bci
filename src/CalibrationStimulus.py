import threading
import numpy as np
import time
from LEDPI import LEDPI
from network import bufhelp

#connect to buffer
bufhelp.connect()
print(bufhelp.fSample)
print("connected")

#init stimuli
nSymbols = 3
nSequences = 6
nEpochs = nSymbols*5
interEpochDelay = 0.5
stimulusDuration = 2.5
stimulusEventDelay = 0.5
stimulusFullDuration = 2
interSequenceDelay = 5
np.random.seed(0)
stimuli = np.random.randint(nSymbols, size=(nSequences, nEpochs)) #TODO: less random

#init frequencies
frequencies = np.zeros(nSymbols)
frequencies[0] = 11
frequencies[1] = 13
frequencies[2] = 9
#frequencies[3] = 25
frequency_full = 0
leds=LEDPI([13, 15, 11])
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
        if symbol < 3:
            #frequencies_led=frequencies[0:3].tolist()
            frequencies_led[symbol] = frequency_full
        else:
            frequencies_led[0] = frequency_full
            frequencies_led[1] = frequency_full
            frequencies_led[2] = frequency_full
        leds.changeLED(frequencies_led)

        time.sleep(stimulusFullDuration)

        frequencies_led=frequencies[0:3].tolist()
        leds.changeLED(frequencies_led)
        time.sleep(stimulusEventDelay)
        bufhelp.sendevent('stimulus.visible', symbol)
        time.sleep(stimulusDuration)

        print('stimulus.epoch', 'end')
        time.sleep(interEpochDelay)

    print('stimulus.sequence', 'end')
    frequencies_led=[0,0,0]
    leds.changeLED(frequencies_led)
    time.sleep(interSequenceDelay)

bufhelp.sendevent('stimulus.training', 'end')
leds.blink = False
print("calibration END")
