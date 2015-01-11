from nxt.brick import Brick
from nxt.locator import find_one_brick
from nxt.motor import Motor, PORT_A, PORT_B, PORT_C
from nxt.sensor import Touch
from nxt.sensor import PORT_1, PORT_2, PORT_3, PORT_4

__author__ = 'PAUL'


class Robot():

    def __init__(self):
        self.brick = find_one_brick()
        self.sensors = [Touch(self.brick, PORT_1)]
        self.motors = [Motor(self.brick, PORT_A), Motor(self.brick, PORT_C)]

    def increase_velocity(self):
        pass

    def decrease_velocity(self):
        pass

    def turn_left(self):
        pass

    def turn_right(self):
        pass

    def beep(self, frequency=523):
        self.brick.play_tone_and_wait(frequency, 500)

    def show_message(self, message):
        self.brick.message_write(message)

    def read_sensor(self, idx):
        return self.sensors[idx].get_sample()