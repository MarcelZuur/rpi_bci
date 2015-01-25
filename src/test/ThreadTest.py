from __future__ import print_function
import threading
import time
import sys

print = lambda x: sys.stdout.write("%s\n" % x)


def main():
    t1_stop = threading.Event()
    t1 = threading.Thread(target=thread, args=(1, t1_stop))
    t1.start()
    print("tm")
    time.sleep(5)
    print("stop")
    t1_stop.set()


def thread(arg1, stop_event):
    while not stop_event.is_set():
        # time.sleep(1)
        stop_event.wait(0.5)
        print(time.time())


main()