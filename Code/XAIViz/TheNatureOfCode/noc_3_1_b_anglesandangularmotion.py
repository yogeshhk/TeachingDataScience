# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 3-1: Angles and Angulr Motion
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp03_oscillation/NOC_3_02_forces_angular_motion
# But followed on screen example
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=19

from p5 import *
import random

class Mover(object):
    def __init__(self, mass, x, y):
        self.mass = mass
        self.position = Vector(x, y)
        self.velocity = Vector(random.uniform(-1, 1), random.uniform(-1, 1))
        self.acceleration = Vector(0, 0)

        self.angle = 0
        self.aVelocity = 0
        self.aAcceleration = 0

    def applyForce(self, force):
        f = force / self.mass
        self.acceleration += f

    def update(self):
        self.velocity += self.acceleration
        self.position += self.velocity

        self.aAcceleration = self.acceleration.x / 10.0
        self.aVelocity += self.aAcceleration
        self.aVelocity = constrain(self.aVelocity, -0.1, 0.1)
        self.angle += self.aVelocity

        self.acceleration *= 0

    def display(self):
        stroke(0)
        fill(175, 200)
        rectMode(CENTER)

        pushMatrix()
        translate(self.position.x, self.position.y)
        rotate(self.angle)
        rect(0, 0, self.mass * 16, self.mass * 16)
        # ellipse(0,0,self.mass*25,self.mass*25)
        popMatrix()


class Attractor(object):
    """A class for a draggable attractive body in our world"""

    def __init__(self, position=Vector(0, 0),
                 mass=20, g=0.4):
        self.position = position
        self.mass = mass
        self.g = g

    def attract(self, m):
        # Calculate the direction of force.
        force = self.position - m.position

        # Get the distance between the bodies using the force magnitude
        distance = force.magnitude

        # Limit the distance to eliminate "extreme" results for very close
        # or very far objects
        distance = constrain(distance, 5.0, 25.0)

        # We are only interested in the direction, so normalize.
        force.normalize()

        # Calculate the gravitional force magnitude
        strength = (self.g * self.mass * m.mass) / (distance * distance)

        # Get force vector.
        force *= strength

        return force

    def display(self):
        """Method to display"""
        stroke(0)
        strokeWeight(2)
        fill(127)
        ellipse(self.position.x, self.position.y, 48, 48)


max_movers = 20


def setup():
    size(640, 360)

    global movers, a

    movers = [Mover(random.uniform(0.1, 2), random.uniform(0,width), random.uniform(0,height)) \
              for i in range(max_movers)]
    a = Attractor(position=Vector(width / 2, height / 2))

    background(255)


def draw():
    global a, movers

    background(255)
    a.display()

    for mv in movers:
        force = a.attract(mv)
        mv.applyForce(force)

        mv.update()
        mv.display()


if __name__ == "__main__":
    run()