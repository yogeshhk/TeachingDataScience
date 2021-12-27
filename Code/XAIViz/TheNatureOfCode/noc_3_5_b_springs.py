# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 3-5b: Springs
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp03_oscillation/NOC_3_11_spring
# But followed on screen example
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=22

from p5 import *


class Bob(object):
    """
    Bob class, just like our regular
    Mover(position, velocity, acceleration, mass)
    """

    def __init__(self, x, y):
        self.position = Vector(x, y)
        self.velocity = Vector(0,0)
        self.acceleration = Vector(0,0)

        self.mass = 24

        # Arbitrary self.damping to simulate friction / drag
        self.damping = 0.98

        # For mouse interaction
        self.dragOffset = Vector(0,0)
        self.dragging = False

    def update(self):
        """
        Standard Euler integration
        """
        self.velocity += self.acceleration
        self.velocity *= self.damping
        self.position += self.velocity
        self.acceleration *= 0

    def applyForce(self, force):
        # Newton's law: F = M * A
        f = force.copy()
        f /= self.mass
        self.acceleration += f

    def display(self):
        """Draw the bob"""
        stroke(0)
        strokeWeight(2)
        fill(175)
        if (self.dragging):
            fill(50)
        ellipse(self.position.x, self.position.y, self.mass * 2, self.mass * 2)

    # The methods below are for mouse interaction

    def clicked(self, mx, my):
        """This checks to see if we clicked on the mover"""
        d = dist(Vector(mx, my), self.position)
        if d < self.mass:
            self.dragging = True
            self.dragOffset.x = self.position.x - mx
            self.dragOffset.y = self.position.y - my

    def stopDragging(self):
        self.dragging = False

    def drag(self, mx, my):
        if self.dragging:
            self.position.x = mx + self.dragOffset.x
            self.position.y = my + self.dragOffset.y


class Spring(object):
    """
    Class to describe an anchor point that can connect to "Bob" objects via
    a spring.
    Thank you: http://www.myphysicslab.com/spring2d.html
    """

    def __init__(self, x, y, l):
        # position
        self.anchor = Vector(x, y)

        # Rest length and spring constant
        self.length = l
        self.k = 0.2

    def connect(self, b):
        """Calculate spring force"""

        # Vector pointing from anchor to bob position
        force = b.position - self.anchor

        # What is distance
        d = force.magnitude

        # Stretch is difference between current distance and rest length
        stretch = d - self.length

        # Calculate force according to Hooke's Law
        # F = k * stretch
        force.normalize()
        force *= -1 * self.k * stretch
        b.applyForce(force)

    def constrainLength(self, b, minlen, maxlen):
        """
        Constrain the distance between bob and anchor between min and max
        """

        direction = b.position - self.anchor
        d = direction.magnitude

        # Is it too short?
        if d < minlen:
            direction.normalize()
            direction *= minlen

            # Reset position and stop from moving (not realistic physics)
            b.position = self.anchor + direction
            b.velocity *= 0

        # Is it too long?
        elif d > maxlen:
            direction.normalize()
            direction *= maxlen

            # Reset position and stop from moving (not realistic physics)
            b.position = self.anchor + direction
            b.velocity *= 0

    def display(self):
        stroke(0)
        fill(175)
        strokeWeight(2)
        rectMode(CENTER)
        rect(self.anchor.x, self.anchor.y, 10, 10)

    def displayLine(self, b):
        strokeWeight(2)
        stroke(0)
        line(b.position.x, b.position.y, self.anchor.x, self.anchor.y)


def setup():
    size(640, 360)

    # Create objects at starting position
    # Note third argument in Spring constructor is "rest length"
    global bob, spring
    spring = Spring(width / 2, 10, 100)
    bob = Bob(width / 2, 100)


def draw():
    background(255)
    global bob, spring

    # Apply a gravity force to the bob
    gravity = Vector(0, 2)
    bob.applyForce(gravity)

    # Connect the bob to the spring (this calculates the force)
    spring.connect(bob)

    # Constrain spring distance between min and max
    spring.constrainLength(bob, 30, 200)

    # Update bob
    bob.update()
    # If it's being dragged
    bob.drag(mouse_x, mouse_y)

    # Draw a line between spring and bob
    spring.displayLine(bob)

    # Draw everything else
    bob.display()
    spring.display()

    fill(0)
    text("click on bob to drag", 10, height - 5)


# For mouse interaction with bob
def mousePressed():
    global bob
    bob.clicked(mouse_x, mouse_y)


def mouseReleased():
    global bob
    bob.stopDragging()

if __name__ == "__main__":
    run()