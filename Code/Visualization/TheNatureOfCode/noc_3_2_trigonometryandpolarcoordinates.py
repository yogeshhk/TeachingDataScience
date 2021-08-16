# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 3-2: Trigonometry and Polar Coordinates
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp03_oscillation/NOC_3_04_PolarToCartesian
# But followed on screen example
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=20

from p5 import *


# PolarToCartesian
# Convert a polar coordinate (r,theta) to cartesian (x,y):
# x = r * cos(theta)
# y = r * sin(theta)


def setup():
    size(640, 360)
    # Initialize all values
    global r, theta, angularVeclocity, angularAcceleration
    r = height * 0.45
    theta = 0
    angularVeclocity = 0.0
    angularAcceleration = 0.01


def draw():
    background(255)
    global r, theta, angularVeclocity, angularAcceleration

    # Translate the origin point to the center of the screen
    translate(width / 2, height / 2)

    # Convert polar to cartesian
    x = r * cos(theta)
    y = r * sin(theta)

    # Draw the ellipse at the cartesian coordinate
    ellipseMode(CENTER)
    fill(127)
    stroke(0)
    strokeWeight(2)
    line(0, 0, x, y)
    ellipse(x, y, 48, 48)

    # Increase the angle over time
    theta += angularVeclocity
    angularVeclocity += angularAcceleration
    angularVeclocity = constrain(angularVeclocity,0,0.1)

if __name__ == "__main__":
    run()