# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example 1-2: PVector Class
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/chp01_vectors/NOC_1_2_bouncingball_vectors
# But made Object Oriented as per video.
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=8

from p5 import *

class Ball(object):

    def __init__(self):
        self.location = Vector(width/2, height/2)
        self.velocity = Vector(2.5, 5)

    def move(self):
        self.location += self.velocity  # location.add(velocity)

    def bounce(self):
        if (self.location.x > width) or (self.location.x < 0):
            self.velocity.x = self.velocity.x * -1
        if (self.location.y > height) or (self.location.y < 0):
            self.velocity.y = self.velocity.y * -1

    def display(self):
        # Display circle at x location
        stroke(0)
        strokeWeight(2)
        fill(175)
        ellipse(self.location.x, self.location.y, 48, 48)

def setup():
    size(400, 300)
    global b
    b = Ball()

def draw():
    background(255)
    b.move()
    b.bounce()
    b.display()

if __name__ == "__main__":
    run()