__author__ = 'PAUL'

import time

import numpy as np

import RPi.GPIO as GPIO


class Blink():
    # init GPIO
    GPIO.setmode(GPIO.BOARD)
    nSymbols = 4
    pins = np.zeros(nSymbols)
    pins[0] = 7
    pins[1] = 8
    pins[2] = 9
    pins[3] = 10
    for id in pins:
        GPIO.setup(id, GPIO.OUT)
        leds = []
        leds.append(Led(frequency, pin))

    def start(indexes, stop_event):
        while not stop_event.is_set():
            for led in leds:
                led.blinkCheck()
        GPIO.cleanup()


class Led():
    def __init__(self, frequency, pin):
        self._last_update = time.time()
        self._delay = 1 / frequency
        self._pin = pin
        self._pin_ON = False

    def blinkCheck(self):
        if time.time() >= self._last_update + self._delay:
            if not self._pin_ON:
                GPIO.output(self._pin, True)
                self._pin_ON = True
            else:
                GPIO.output(self._pin, False)
                self._pin_ON = False
            self._last_update = time.time()


