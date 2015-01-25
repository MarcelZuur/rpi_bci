from nxt.brick import Brick
from nxt.locator import find_one_brick
from nxt.motor import Motor, PORT_A, PORT_B, PORT_C, SynchronizedMotors
from nxt.sensor import Touch
from nxt.sensor import PORT_1, PORT_2, PORT_3, PORT_4
import time


class Robot():
    """Has functions for controlling the nxt brick.

    Attributes:
      brick: The connected nxt brick.
      sensors: List of used sensors.
      motors: List of used motors.

    """

    def __init__(self):
        """Constructs a robot object which can be used to control the nxt brick."""
        self.brick = find_one_brick()
        self.sensors = [Touch(self.brick, PORT_1)]
        self.motors = [Motor(self.brick, PORT_A), Motor(self.brick, PORT_B)]
        #self.motors_both = SynchronizedMotors(self.motors[0], self.motors[1], 0)

    def move_forward(self):
        """Moves the robot forward."""
        both=SynchronizedMotors(self.motors[0], self.motors[1], 0)
        both.turn(-70, 2*250)

    def stop(self):
        """Stops the robot."""
        self.motors_both.idle()

    def turn_left(self):
        """Turns the robot left."""
        both=SynchronizedMotors(self.motors[0], self.motors[1], 20)
        both.turn(-70, 250)

    def turn_right(self):
        """Turns the robot right."""
        both=SynchronizedMotors(self.motors[1], self.motors[0], 20)
        both.turn(-70, 250)

    def beep(self, frequency=523):
        """Causes the robot to beep at the given frequency."""
        self.brick.play_tone_and_wait(frequency, 500)

    def show_message(self, message):
        self.brick.message_write(message)

    def read_sensor(self, idx=0):
        """Returns whether the touch sensor is pressed."""
        return self.sensors[idx].get_sample()