# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 3-4 a: Pendulum Simulation
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp03_oscillation/??
# But followed on screen example
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=22

from p5 import *



def setup():

    global origin, bob, angle, len, aVel, aAcc
    size(640, 360)
    len = 180
    angle = PI/4
    aVel = 0.0
    aAcc = 0.0

    origin = Vector(width/2,0)
    bob = Vector(width/2,len)


def draw():
    background(255)
    global origin, bob, angle, len, aVel, aAcc
    bob.x = origin.x + len * sin(angle)
    bob.y = origin.y + len * cos(angle)
    line(origin.x, origin.y, bob.x, bob.y)
    ellipse(bob.x, bob.y, 48, 48)

    aAcc = -0.01 * sin(angle)

    angle += aVel
    aVel += aAcc
    aVel *= 0.99 # some dampening to get it to rest


if __name__ == "__main__":
    run()