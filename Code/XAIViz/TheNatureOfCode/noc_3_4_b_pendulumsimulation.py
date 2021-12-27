# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 3-4 b: Pendulum Simulation
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp03_oscillation/NOC_3_10_PendulumExample
# But followed on screen example
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=22

# Pendulum
#
# A simple pendulum simulation
# Given a pendulum with an angle theta (0 being the pendulum at rest) and
# a radius r we can use sine to calculate the angular component of the
# gravitational force.
#
# Gravity Force = Mass * Gravitational Constant;
# Pendulum Force = Gravity Force * sine(theta)
# Angular Acceleration =
#       Pendulum Force / Mass = gravitational acceleration * sine(theta);
#
# Note this is an ideal world scenario with no tension in the pendulum arm,
# a more realistic formula might be:
#       Angular Acceleration = (g / R) * sine(theta)
#
# For a more substantial explanation, visit:
# http://www.myphysicslab.com/pendulum1.html

from p5 import *


class Pendulum(object):
    """
    A Simple Pendulum Class
    Includes functionality for user can click and drag the pendulum
    """

    def __init__(self, origin, r):
        """
        This constructor could be improved to allow a greater variety of
        pendulums
        """

        # position of pendulum ball
        self.position = Vector(0,0)

        # position of arm origin
        self.origin = origin.copy()

        # Length of arm
        self.r = r

        # Pendulum arm angle
        self.angle = PI / 4

        # Angle velocity
        self.aVelocity = 0.0

        # Angle acceleration
        self.aAcceleration = 0.0

        # Arbitrary ball radius
        self.ballr = 48

        # Arbitary damping amount
        self.damping = 0.995

        self.dragging = False

    def go(self):
        self.update()
        self.drag()  # for user interaction
        self.display()

    def update(self):
        """
        Function to update position
        """
        # As long as we aren't dragging the pendulum, let it swing!
        if not self.dragging:
            # Arbitrary constant
            gravity = 0.4

            # Calculate acceleration
            # (see: http://www.myphysicslab.com/pendulum1.html)
            self.aAcceleration = (-1 * gravity / self.r) * sin(self.angle)

            # Increment velocity
            self.aVelocity += self.aAcceleration

            # Arbitrary damping
            self.aVelocity *= self.damping

            # Increment angle
            self.angle += self.aVelocity

    def display(self):
        # Polar to cartesian conversion
        self.position = Vector(self.r * sin(self.angle), self.r * cos(self.angle), 0)

        #  Make sure the position is relative to the pendulum's origin
        self.position += self.origin

        stroke(0)
        strokeWeight(2)

        # Draw the arm
        line(self.origin.x, self.origin.y, self.position.x, self.position.y)
        ellipseMode(CENTER)
        fill(175)

        if self.dragging:
            fill(0)

        # Draw the ball
        ellipse(self.position.x, self.position.y, self.ballr, self.ballr)

    # The methods below are for mouse interaction

    def clicked(self, mx, my):
        """
        This checks to see if we clicked on the pendulum ball
        """
        m = Point(mx,my)
        pos = Point(self.position.x,self.position.y)
        d = distance(m, pos)
        if d < self.ballr:
            self.dragging = True

    def stopDragging(self):
        """
        This tells us we are not longer clicking on the ball.
        """
        # No velocity once you let go
        self.aVelocity = 0
        self.dragging = False

    def drag(self):
        # If we are draging the ball, we calculate the angle between the
        # pendulum origin and mouse position we assign that angle to the
        # pendulum
        if self.dragging:
            # Difference between 2 points
            diff = self.origin - Vector(mouse_x, mouse_y)
            # Angle relative to vertical axis
            angle = atan2(-1 * diff.y, diff.x) - radians(90)

def setup():
    size(640, 360)

    # Make a new Pendulum with an origin position and armlength
    global p
    p = Pendulum(Vector(width / 2, 0), 175)


def draw():
    background(255)
    p.go()


def mousePressed():
    p.clicked(mouse_x, mouse_y)


def mouseReleased():
    p.stopDragging()

if __name__ == "__main__":
    run()