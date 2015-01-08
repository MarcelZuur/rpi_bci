#!/usr/bin/env python

import nxt.locator
from nxt.motor import *

def spin_around(b):
    m_left = Motor(b, PORT_B)
    m_left.turn(100, 360)
    m_right = Motor(b, PORT_C)
    m_right.turn(-100, 360)

def move_forward(b):
    m_left = Motor(b, PORT_B)
    m_right = Motor(b, PORT_C)
    both = SynchronizedMotors(m_left,m_right,0)
    #both.turn(-70,1000)
    both.run(75)
    time.sleep(2)
    both.idle()
    m_left.run(75)
    time.sleep(2)
    m_left.idle()

b = nxt.locator.find_one_brick()
#spin_around(b)
move_forward(b)
print("end")