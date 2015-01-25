import time

import numpy as np

import RPi.GPIO as GPIO


# init GPIO
GPIO.setmode(GPIO.BOARD)
nSymbols = 4
pins = np.zeros(nSymbols)
pins[0] = 7
pins[1] = 8
pins[2] = 9
pins[3] = 10
GPIO.setup(pins[0], GPIO.OUT)
GPIO.setup(pins[1], GPIO.OUT)
GPIO.setup(pins[2], GPIO.OUT)
GPIO.setup(pins[3], GPIO.OUT)

#TODO: convert to frequency and duration
def blink(symbol, nTimes, delay):
    for i in range(0, nTimes):
        print "led ON"
        GPIO.output(pins[symbol], True)
        time.sleep(delay)
        GPIO.output(pins[symbol], False)
        print "led OFF"
        time.sleep(delay)
    GPIO.cleanup()


def test():
    nTimes = 5
    frequency = 10
    symbol = 0
    blink(symbol, nTimes, 1 / frequency)
