# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 2-3 : Simulating with Mass
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp02_forces/NOC_2_2_forces_many/
# But followed on screen example
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=15

from p5 import *
import random

num_movers = 5

class Mover(object):
    def __init__(self, m, x, y):
        self.position = Vector(x, y)
        self.velocity = Vector(0, 0)
        self.acceleration = Vector(0, 0)
        self.mass = m

    def applyForce(self, force):
        f = force / self.mass
        self.acceleration += f

    def update(self):
        self.velocity += self.acceleration
        self.position += self.velocity
        self.acceleration *= 0

    def display(self):
        stroke(0)
        strokeWeight(2)
        fill(0, 127)
        ellipse(self.position.x, self.position.y, self.mass * 20, self.mass * 20)

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
    global movers
    movers = []
    for i in range(num_movers):
        movers.append(Mover(random.uniform(0.1, 4), random.uniform(0,640), 200))

def draw():
    background(255)

    for m in movers:
        gravity = Vector(0, 0.1)
        gravity *= m.mass # next, internally it will get divided, so as to keep gravitational acceleration constant
        m.applyForce(gravity)

        if mouse_is_pressed:
            wind = Vector(0.01, 0)
            m.applyForce(wind)

        m.update()
        m.display()
        m.checkEdges()


if __name__ == "__main__":
    run()