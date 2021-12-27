# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 1-6: Acceleration towards mouse
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp01_vectors/NOC_1_7_motion101
# But followed on screen example
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=12

from p5 import *
import random

class Mover(object):

    def __init__(self):
        self.location = Vector(width/2,height/2) #Vector(random.uniform(0,width), random.uniform(0,height))
        self.velocity = Vector(0,0) #Vector(random.uniform(-2, 2), random.uniform(-2, 2))
        self.acceleration = Vector(0,0)

    def update(self):
        self.mouse = Vector(mouse_x,mouse_y)
        self.mouse -= self.location
        self.mouse.magnitude = 0.1

        self.acceleration = self.mouse
        self.velocity += self.acceleration
        self.location += self.velocity
        self.velocity.limit(5)

    def display(self):
        stroke(0)
        strokeWeight(2)
        fill(127)
        ellipse(self.location.x, self.location.y, 48, 48)

    def checkEdges(self):
        if self.location.x > width:
            self.location.x = 0
        elif self.location.x < 0:
            self.location.x = width

        if self.location.y > height:
            self.location.y = 0
        elif self.location.y < 0:
            self.location.y = height

def setup():
    size(640, 360)
    global mover
    mover = Mover()


def draw():
    background(255)

    mover.update()
    mover.checkEdges()
    mover.display()

if __name__ == "__main__":
    run()