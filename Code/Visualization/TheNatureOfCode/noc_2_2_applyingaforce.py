# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 2-2: Applying a force?
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp02_forces/NOC_2_1_forces
# But followed on screen example
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=14

from p5 import *

class Mover(object):
    def __init__(self):
        self.position = Vector(30, 30)
        self.velocity = Vector(0, 0)
        self.acceleration = Vector(0, 0)

    def applyForce(self, force):
        self.acceleration += force

    def update(self):
        self.velocity += self.acceleration
        self.position += self.velocity
        self.acceleration *= 0

    def display(self):
        stroke(0)
        strokeWeight(2)
        fill(127)
        ellipse(self.position.x, self.position.y, 48, 48)

    def checkEdges(self):
        if (self.position.x > width):
            self.position.x = width
            self.velocity.x *= -1
        elif (self.position.x < 0):
            self.position.x = 0
            self.velocity.x *= -1

        if (self.position.y > height):
            self.position.y = height
            self.velocity.y *= -1


def setup():
    size(640, 360)
    global m
    m = Mover()


def draw():
    background(255)

    gravity = Vector(0, 0.1)
    m.applyForce(gravity)

    if mouse_is_pressed:
        wind = Vector(0.01, 0)
        m.applyForce(wind)

    m.update()
    m.display()
    m.checkEdges()


if __name__ == "__main__":
    run()