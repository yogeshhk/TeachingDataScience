# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 3-1: Angles and Angulr Motion
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp03_oscillation/NOC_3_01_angular_motion
# But followed on screen example
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=19

from p5 import *

angle = 0
aVelocity = 0
aAcceleration = 0.0001


def setup():
    size(800, 200)
    # smooth()


def draw():
    global angle, aVelocity, aAcceleration

    background(255)

    fill(127)
    stroke(0)

    translate(width / 2, height / 2)
    rectMode(CENTER)
    rotate(angle)

    stroke(0)
    strokeWeight(2)
    fill(127)

    rect(0,0,64,36)

    # line(-60, 0, 60, 0)
    # ellipse(60, 0, 16, 16)
    # ellipse(-60, 0, 16, 16)

    angle += aVelocity
    aVelocity += aAcceleration

if __name__ == "__main__":
    run()