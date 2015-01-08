# Model View Controller structure, as the base for the RPI project.
import sys
import select
import threading
import time

from network import FieldTrip


class View():
    def update(self, model, message):
        print model.getStatus()


class Model():
    def __init__(self):
        self.observers = []
        self.status = 0
        self.statusList = {0: "wait", 1: "acquire", 2: "train", 3: "run", 4: "stop"}

    #Add the observers to the list
    def addObserver(self, item):
        self.observers.append(item)

    def __setChanged(self, message):

        if (message == "key pressed"):
            self.status = (self.status + 1) % 4
        elif (message == "stop pressed"):
            self.status = 4

    #Getter and setter for the status.
    def __setStatus(self, status):
        self.status = status

    def getStatus(self):
        return self.statusList[self.status]

    #Notify the Views the model has changed
    def __notifyObservers(self, message):
        for observer in self.observers:
            observer.update(self, message)

    def __notifyObeservers(self):
        self.__notifyObservers(None)

    def startKeyPressed(self):
        self.__setChanged("key pressed")
        self.__notifyObeservers()

    def stopKeyPressed(self):
        self.__setChanged("stop pressed")
        self.__notifyObeservers()


#Controller which can be controlled using the SignalBuffer.
class FieldTripController():
    def __init__(self, model):
        self.model = model
        self.runLoop = True
        self.ftc = FieldTrip.Client()

    def run(self):
        thread = threading.Thread(target=self.__pythonclient)
        #  thread.Daemon = True
        thread.start()

    #    thread.join()

    def __pythonclient(self, hostname='localhost', port=1972, timeout=5000):


        # Wait until the buffer connects correctly and returns a valid header
        hdr = None
        while hdr is None:
            print 'Trying to connect to buffer on %s:%i ...' % (hostname, port)
            try:
                self.ftc.connect(hostname, port)
                print '\nConnected - trying to read header...'
                hdr = self.ftc.getHeader()
            except IOError:
                pass

            if hdr is None:
                print 'Invalid Header... waiting'
                time.sleep(1)
            else:
                print hdr
                print hdr.labels

        # Now do the echo server
        nEvents = hdr.nEvents

        while self.runLoop:
            (curSamp, curEvents) = self.ftc.wait(-1, nEvents, timeout)  # Block until there are new events to process
            if curEvents > nEvents:
                evts = self.ftc.getEvents([nEvents, curEvents - 1])
                nEvents = curEvents  # update record of which events we've seen
                for evt in evts:
                    if evt.type == "echo": continue
                    if evt.type == "exit": endExpt = 1
                    if evt.value == 's':
                        self.model.startKeyPressed()
                    elif evt.value == 'q':
                        self.model.stopKeyPressed()

    def stop(self):
        self.runLoop = False


#Controller which listens to key presses. The up key changes the models status, while the down key exits the application
class KeyboardController():
    def __init__(self, model):
        self.model = model
        self.runLoop = True

    def run(self):
        thread = threading.Thread(target=self.__keyListener)
        thread.start()
        #   thread.join()

    def __keyListener(self):

        #This listens to the down key. Should be changed to one of the NXT buttons.
        while self.runLoop:
            i, o, e = select.select([sys.stdin], [], [], 0.0001)
            for s in i:
                if s == sys.stdin:
                    input = sys.stdin.readline()
                    if input[0] == 's':
                        self.model.startKeyPressed()
                    elif input[0] == 'q':
                        self.model.stopKeyPressed()

    def stop(self):
        self.runLoop = False


#Main program
class Program():
    def __init__(self):
        model = Model()
        view = View()
        model.addObserver(view)
        model.addObserver(self)
        self.keyController = KeyboardController(model)
        self.ftController = FieldTripController(model)
        self.keyController.run()
        self.ftController.run()

    def update(self, model, message):
        if (model.getStatus() == "stop"):
            self.keyController.stop()
            self.ftController.stop()


if __name__ == '__main__':
    program = Program()
