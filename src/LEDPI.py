# Use the pi for leds.
import time
import RPi.GPIO as GPIO


class LEDPI():
    def blinkLED(self):
        newTime = [0] * len(self.millis)
        while (self.blink):
            millis = int(round(time.time() * 1000))
            for idx, val in enumerate(self.millis):
                if val == 0:
                    GPIO.output(self.pins[idx], False)
                    continue
                if (millis > newTime[idx]):
                    newTime[idx] = millis + val
                    if (self.flipLED[idx]):
                        self.flipLED[idx] = False
                        GPIO.output(self.pins[idx], True)
                    else:
                        GPIO.output(self.pins[idx], False)
                        self.flipLED[idx] = True
        GPIO.cleanup()


    def __init__(self, pins):
        self.pins = pins
        self.blink = True
        self.millis = [0] * len(pins)
        GPIO.setmode(GPIO.BOARD)
        # pins[3] = 10
        for idx in range(0, len(pins)):
            GPIO.setup(pins[idx], GPIO.OUT)
            GPIO.output(pins[idx], False)

    def changeLED(self, frequencies):
        self.flipLED = [False] * len(frequencies)
        for idx, freq in enumerate(frequencies):
            self.millis[idx]=(1000 / freq if freq >0 else 0)

    def update(self, model):
        pass

if __name__ == '__main__':
    program = LEDPI([11, 13, 15])
