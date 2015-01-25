#!/usr/bin/python2.7
"""Calibration stimulus script.

This script displays the stimulus of the calibration phase, and sends messages to the buffer used by SigProc.py
to label and gather data. The leds flicker continuously and the target is indicated by turning the target led off
for a specified time, after which the flicker resumes and an event with the target is send to the buffer.

Attributes:
  nSymbols(int): Number of symbols/leds used in the stimulus presentation.
  nSequences(int): Number of sequences used in the stimulus presentation.
  nEpochs(int): Number of epochs used in the stimulus presentation.
  interEpochDelay(float): Delay between epochs.
  stimulusDuration(float): Duration of the stimulus.
  stimulusEventDelay(float): Delay before sending an event to the buffer indicating the stimulus is active.
  stimulusFullDuration(float): Duration that the target led is off.
  interSequenceDelay(float): Delay between sequences.
  stimuli(int[][]): An array of symbols, indicating which symbol is active for each [sequence, epoch].
  frequencies(int[]): An array of frequencies, where the frequency at index i is the frequency of symbol i.
  frequency_full(int): Frequency of the target led when it's off.
  leds(LEDPI): Object to interact with the leds.
  t1(Thread): Thread that runs the leds.
  frequencies_led(int[]): List of frequencies used as parameter for leds.
"""
import json
import threading
import time
import ConfigParser

import numpy as np

from LEDPI import LEDPI
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
calibrationOptions = ConfigSectionMap("Calibration")
ledOptions = ConfigSectionMap("LED")

hostname = connectionOptions["hostname"]
port = int(connectionOptions["port"])
# connect to buffer
bufhelp.connect(adress=hostname, port=port)
print(bufhelp.fSample)
print("connected")

#init stimuli
nSymbols = 3
nSequences = int(calibrationOptions["sequences"])
nEpochs = nSymbols * 5
interEpochDelay = float(calibrationOptions["interepochdelay"])
stimulusDuration = float(calibrationOptions["stimulusduration"])
stimulusEventDelay = float(calibrationOptions["stimuluseventdelay"])
stimulusFullDuration = float(calibrationOptions["stimulusfullduration"])
interSequenceDelay = float(calibrationOptions["intersequencedelay"])
np.random.seed(0)
stimuli = np.random.randint(nSymbols, size=(nSequences, nEpochs))  #TODO: less random

#init frequencies
freqs = json.loads(ledOptions["frequencies"])
frequencies = np.zeros(nSymbols)
frequencies[0] = int(freqs[0])
frequencies[1] = int(freqs[1])
frequencies[2] = int(freqs[2])
#frequencies[3] = 25
frequency_full = 0

gpio = json.loads(ledOptions["gpio"])
leds = LEDPI(gpio)
t1 = threading.Thread(target=leds.blinkLED)
t1.start()
frequencies_led = frequencies[0:3].tolist()
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

        frequencies_led = frequencies[0:3].tolist()
        leds.changeLED(frequencies_led)
        time.sleep(stimulusEventDelay)
        bufhelp.sendevent('stimulus.visible', symbol)
        time.sleep(stimulusDuration)

        print('stimulus.epoch', 'end')
        time.sleep(interEpochDelay)

    print('stimulus.sequence', 'end')
    frequencies_led = [0, 0, 0]
    leds.changeLED(frequencies_led)
    time.sleep(interSequenceDelay)

bufhelp.sendevent('stimulus.training', 'end')
leds.blink = False
print("calibration END")

