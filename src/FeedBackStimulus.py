#!/usr/bin/python2.7
"""Feedback stimulus script.

This script displays the stimulus of the feedback phase, and sends messages to the buffer used by SigProc.py
to gather data after which it waits for predictions. A prediction buffer is used to combine multiple predictions,
the most frequent prediction in the buffer is taken, if these are all equal the most recent prediction is taken.
After a final prediction is made the robot (see Robot) is moved according to the predicted value and the buffer
is reset then new predictions are gathered again starting the same process over again.
SigProc has to have trained a classifier before to be able to generate predictions, so CalibrationStimulus and
TrainClassifier should have been run at least once before this.


Attributes:
  nSymbols(int): Number of symbols/leds used in the stimulus presentation.
  buffer_size(int): Size of the buffer.
  frequencies(int[]): An array of frequencies, where the frequency at index i is the frequency of symbol i.
  frequency_full(int): Frequency of the target led when it's off.
  leds(LEDPI): Object to interact with the leds.
  t1(Thread): Thread that runs the leds.
  frequencies_led(int[]): List of frequencies used as parameter for leds.
  buffer(int[]): Buffer array containing the predictions, which resets after prediction_count>=buffer_size.
  prediction(int): The predicted symbol.
  prediction_count(int): Current number of predictions.
"""

import ConfigParser
import json
import threading
import time

import numpy as np
from scipy.stats import mode

from LEDPI import LEDPI
from Robot import Robot
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

ledOptions = ConfigSectionMap("LED")
#init robot
robot = Robot()

#init stimuli
nSymbols = 4
buffer_size = 3

#init frequencies
freqs = json.loads(ledOptions["frequencies"])
frequencies = np.zeros(nSymbols)
frequencies[0] = int(freqs[0])
frequencies[1] = int(freqs[1])
frequencies[2] = int(freqs[2])
frequencies[3] = 25  #STOP

gpio = json.loads(ledOptions["gpio"])
leds = LEDPI(gpio)
t1 = threading.Thread(target=leds.blinkLED)
t1.start()
leds.changeLED(frequencies[0:3].tolist())

#start stimuli
print("feedback START")
bufhelp.sendevent('startPhase.cmd', 'feedback')
time.sleep(2)
bufhelp.sendevent('stimulus.feedback', 'start')
buffer = np.zeros(buffer_size, dtype=np.int32)
prediction = 3
prediction_count = 0
while True:
    bufhelp.sendevent("robot.start", 1)
    t2 = time.time()
    e = bufhelp.waitforevent("classifier.predictions", 3000, False)
    if e is not None:
        print (time.time() - t2, e.value)
        buffer = np.roll(buffer, 1)
        buffer[0] = e.value
        prediction_count += 1
        if prediction_count >= buffer_size:
            prediction = mode(buffer)[0]
            prediction_count = 0
        else:
            prediction = 3
    else:
        prediction = 3

    if prediction == 0:
        print("robot turning left")
        robot.turn_left()
    elif prediction == 1:
        print("robot moving forward")
        robot.move_forward()
    elif prediction == 2:
        print("robot turning right")
        robot.turn_right()
    elif prediction == 3:
        print("robot stopping")
        robot.beep()
        robot.stop()

    if robot.read_sensor(0):
        break

bufhelp.sendevent('stimulus.feedback', 'end')
leds.blink = False
print("feedback END")
bufhelp.sendevent('startPhase.cmd', 'exit')
