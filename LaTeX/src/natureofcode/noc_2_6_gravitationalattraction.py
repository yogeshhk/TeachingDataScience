# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 2-6: Gravitational Attraction
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp02_forces/NOC_2_6_attraction
# But followed on screen example
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=18

from p5 import *
import random

class Mover(object):
    def __init__(self):
        self.position = Vector(400, 50)
        self.velocity = Vector(1,0)
        self.acceleration = Vector(0,0)
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


class Attractor(object):

    def __init__(self):
        self.position = Vector(width / 2, height / 2)  # Position
        self.mass = 20  # Mass, tied to size
        self.G = 1;  # Gravitational Constant

        # holds the offset when the object is clicked on
        self.dragOffset = Vector(0.0, 0.0)

        self.dragging = False  # Is the object being dragged?
        self.rollover = False  # Is the mouse over the ellipse?

    def attract(self, m):
        # Calculate the direction of force
        force = self.position - m.position

        # Distance between objects
        d = force.magnitude

        # Limiting the distance to eliminate "extreme" results for
        # very close or very far objects
        d = constrain(d, 5.0, 25.0)

        # Normalize vector (distance doesn't matter here, we just
        # want this vector for direction)
        force.normalize()

        # Calculate gravitional force magnitude
        strength = (self.G * self.mass * m.mass) / float(d * d)

        # Get force vector --> magnitude * direction
        force *= strength

        return force

    def display(self):
        ellipseMode(CENTER)
        strokeWeight(4)
        stroke(0)
        if (self.dragging):
            fill(50)
        elif (self.rollover):
            fill(100)
        else:
            fill(175, 200)
        ellipse(self.position.x, self.position.y, self.mass * 2, self.mass * 2);

    def clicked(self, mx, my):
        d = dist(mx, my, self.position.x, self.position.y)
        if (d < self.mass):
            self.dragging = True
            self.dragOffset.x = self.position.x - mx
            self.dragOffset.y = self.position.y - my

    def hover(self, mx, my):
        d = Vector(mx,my).distance(Vector(self.position.x,self.position.y))
        # d = dist(mx, my, self.position.x, self.position.y)
        if (d < self.mass):
            rollover = True
        else:
            rollover = False

    def stopDragging(self):
        self.dragging = False

    def drag(self):
        if (self.dragging):
            self.position.x = mouse_x + self.dragOffset.x
            self.position.y = mouse_y + self.dragOffset.y


def setup():
    size(640, 360)

    global m
    m = Mover()

    global a
    a = Attractor()


def draw():
    background(255)

    force = a.attract(m)
    m.applyForce(force)
    m.update()

    a.drag()
    a.hover(mouse_x, mouse_y)

    a.display()
    m.display()


def mousePressed():
    a.clicked(mouse_x, mouse_y)


def mouseReleased():
    a.stopDragging()

if __name__ == "__main__":
    run()