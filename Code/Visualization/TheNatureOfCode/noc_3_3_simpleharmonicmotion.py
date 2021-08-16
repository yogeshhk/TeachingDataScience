# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 3-3: Simple Harmonic Motion
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp03_oscillation/NOC_3_05_simple_harmonic_motion
# But followed on screen example
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=21

from p5 import *


def setup():
    size(640, 360)


def draw():
    background(255)

    period = 120 # number of frames needed for a full cycle
    amplitude = 300 # traversal distance

    # Calculating horizontal position according to formula
    # for simple harmonic motion
    x = amplitude * sin(TWO_PI * frame_count / period)
    # x = amplitude * sin(angle), where angle += 0.2 will work also. [Note: angle is in radians]

    stroke(0)
    strokeWeight(2)
    fill(127)
    translate(width / 2, height / 2)
    line(0, 0, x, 0)
    ellipse(x, 0, 48, 48)


if __name__ == "__main__":
    run()