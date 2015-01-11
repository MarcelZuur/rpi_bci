import threading
import numpy as np
import time
#from LEDPI import LEDPI
#from src.Robot import Robot
from network import bufhelp

#connect to buffer
bufhelp.connect()
print("connected")

#init robot
#robot = Robot()

#init stimuli
nSymbols = 4

#init frequencies
frequencies = np.zeros(nSymbols)
frequencies[0] = 13 #LEFT
frequencies[1] = 17 #FORWARD
frequencies[2] = 21 #RIGHT
frequencies[3] = 25 #STOP
#leds=LEDPI([11, 13, 15])
#t1 = threading.Thread(target=leds.blinkLED)
#t1.start()
#leds.changeLED(frequencies[0:3].tolist())

#start stimuli
print("feedback START")
bufhelp.sendevent('startPhase.cmd', 'feedback')
time.sleep(2)
bufhelp.sendevent('stimulus.feedback', 'start')
while True:
    bufhelp.sendevent("robot.start", 1)
    t1 = time.time()
    e = bufhelp.waitforevent("classifier.predictions", 3000, False)
    if e is not None:
        print (time.time()-t1, e.value)
    time.sleep(0.1)
    #if robot.read_sensor(0):
        #break

bufhelp.sendevent('stimulus.feedback', 'end')
#leds.blink = False
print("feedback END")
bufhelp.sendevent('startPhase.cmd', 'exit')
