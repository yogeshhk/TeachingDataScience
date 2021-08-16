# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 1-4: Vector Math II
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp01_vectors/NOC_1_5_vector_magnitude
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp01_vectors/NOC_1_6_vector_normalize/
# But followed on screen example
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=10

from p5 import *

def setup():
    size(500, 300)

def draw():
    background(255)
    strokeWeight(2)
    stroke(0)
    noFill()
    translate(width / 2, height / 2)
    ellipse(0,0,4,4)

    mouse = Vector(mouse_x, mouse_y)
    center = Vector(width / 2, height / 2)
    mouse -= center # mouse.sub(center)
    mouse *= 0.5

    # m = mouse.magnitude
    # fill(255,0,0)
    # rect(0,0,m,20)

    # mouse.normalize()
    # mouse *= 50

    mouse.magnitude = 50 # mouse.setMag(50)

    line(0, 0, mouse.x, mouse.y)

if __name__ == "__main__":
    run()