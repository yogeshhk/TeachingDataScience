# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 2-5: Drag Force
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp02_forces/NOC_2_5_fluidresistance
# But followed on screen example
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=17

from p5 import *
import random

class Mover(object):
    def __init__(self, m, x, y):
        self.position = Vector(x, y)
        self.velocity = Vector(0,0)
        self.acceleration = Vector(0,0)
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
        fill(127,200)
        ellipse(self.position.x, self.position.y,  self.mass*16, self.mass*16)

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

class Liquid(object):
    def __init__(self, x, y, w,  h, c):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c

    def contains(self, m):
        l = m.position
        return (l.x > self.x) and (l.x < (self.x + self.w)) and \
                (l.y > self.y) and (l.y < (self.y + self.h))

    def drag(self, m):
        """
        Calculates the drag force
        """
        speed = m.velocity.magnitude
        dragMagnitude = self.c * speed * speed

        dragForce = m.velocity.copy()
        dragForce *= -1

        # dragForce.setMag(dragMagnitude)
        dragForce.normalize()
        dragForce *= dragMagnitude
        return dragForce

    def display(self):
        noStroke()
        fill(50)
        rect(self.x, self.y, self.w, self.h)

def setup():
    size(640, 360)
    reset()

    global liquid
    liquid = Liquid(0, height/2, width, height/2, 0.1)

def draw():
    background(255)

    liquid.display()

    for mover in movers:

        # Is the Mover in the liquid?
        if liquid.contains(mover):
            # Calculate the drag force
            dragForce = liquid.drag(mover)
            # Apply the drag force
            mover.applyForce(dragForce)

        # Gravity is scaled by mass here!
        gravity = Vector(0, 0.1*mover.mass)
        # Apply gravity
        mover.applyForce(gravity)

        # update and display
        mover.update()
        mover.display()
        mover.checkEdges()

    fill(0)
    #text("click mouse to reset", 10, 30)

def mousePressed():
    reset()

def reset():
    # restart all movers randomly
    global movers
    movers = [Mover(random.uniform(0.5, 3), 40 + i*70, 0) for i in range(8)]

if __name__ == "__main__":
    run()