# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 3-5a: Springs
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp03_oscillation/??
# But followed on screen example
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=22

from p5 import *

class Mover(object):
    def __init__(self,x=width/2,y=30):
        self.position = Vector(x, y)
        self.velocity = Vector(0, 0)
        self.acceleration = Vector(0, 0)
        self.mass = 1

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

origin = Vector(width/2,0)
bob = Mover(width/2,240)
restLength = 200

def setup():
    size(640, 360)

def draw():
    background(255)
    line(origin.x,origin.y,bob.position.x,bob.position.y)
    spring = bob.position - origin
    currentLength = spring.magnitude
    spring.normalize()
    k = 0.1
    stretch = currentLength - restLength
    spring *= -1 * k * stretch
    bob.applyForce(spring)

    gravity = Vector(0,0.1)
    bob.applyForce(gravity)

    wind = Vector(0.1,0)
    if mouse_is_pressed:
        bob.applyForce(wind)
    bob.update()
    bob.display()



if __name__ == "__main__":
    run()