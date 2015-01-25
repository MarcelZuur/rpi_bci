# Model View Controller structure, as the base for the RPI project.
from BCIController import BCIController
from SignalProc import SignalProc
from StateControllers import KeyboardStateController
import Views

from Model import BCIModel


class Program(Views.Observer):
    def __init__(self):
        model = BCIModel({1: 12, 2: 16, 3: 18})
        # Views
        view = Views.StatusView()
        arduinoview = Views.LEDArduino()
        signalproc = SignalProc(model)

        model.add_observer(view)
        model.add_observer(arduinoview)
        model.add_observer(signalproc)

        #controllers
        bciController = BCIController(model, arduinoview)
        bciController.start()
        arduinoview.set_controller(bciController)
        # model.addObserver(self)
        self.keyController = KeyboardStateController(model)

        #        self.ftController = FieldTripController(model)
        self.keyController.run()
        #   self.ftController.run()

    # TODO see if this is actually necessary to do. Normally the program should not observe the model.
    def update(self, model):
        if (model.getStatus() == "stop"):
            self.keyController.stop()


if __name__ == '__main__':
    program = Program()
