from network import bufhelp
import time

#connect to buffer
bufhelp.connect()
print("connected")

time.sleep(1)
print("train classifier START")
bufhelp.sendevent('startPhase.cmd', 'train')
while True:
    print("wait for classifier")
    e = bufhelp.waitforevent(("sigproc.training","end"), 1000, False)
    if e is not None:
        break
print("training END")
