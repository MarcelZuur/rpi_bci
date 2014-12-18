import RPi.GPIO as GPIO
import time
import numpy as np

#init GPIO
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
def blink(symbol, nTimes, speed):
        for i in range(0, nTimes):
                print "led ON"
                GPIO.output(pins[symbol], True)
                time.sleep(speed)
                GPIO.output(pins[symbol], False)
                print "led OFF"
                time.sleep(speed)
        GPIO.cleanup()


def test():
        nTimes = 5
        speed = 0.5 #seconds
        symbol = 0
        blink(symbol, nTimes, speed)
