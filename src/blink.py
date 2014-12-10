import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
pin1 = 7;
GPIO.setup(pin1, GPIO.OUT)

def blink(nTimes, speed):
        for i in range(0, nTimes):
                print "led1 ON"
                GPIO.output(pin1, True)
                time.sleep(speed)
                GPIO.output(pin1, False)
                print "led1 OFF"
                time.sleep(speed)
        GPIO.cleanup()

nTimes = 5
speed = 0.5 #seconds

blink(nTimes, speed)
