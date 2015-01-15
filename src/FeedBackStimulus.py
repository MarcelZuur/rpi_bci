import threading
import numpy as np
import time
from LEDPI import LEDPI
from Robot import Robot
from scipy.stats import mode
from network import bufhelp

#connect to buffer
bufhelp.connect()
print("connected")

#init robot
robot = Robot()

#init stimuli
nSymbols = 4
buffer_size = 3

#init frequencies
frequencies = np.zeros(nSymbols)
frequencies[0] = 11 #LEFT
frequencies[1] = 17 #FORWARD
frequencies[2] = 23 #RIGHT
#frequencies[3] = 25 #STOP
leds=LEDPI([13, 15, 11])
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
    t1 = time.time()
    e = bufhelp.waitforevent("classifier.predictions", 3000, False)
    if e is not None:
        print (time.time()-t1, e.value)
        buffer = np.roll(buffer, 1)
        buffer[0] = e.value
        prediction_count+=1
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
