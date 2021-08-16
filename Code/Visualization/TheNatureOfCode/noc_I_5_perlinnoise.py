# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example I-5: Perlin Noise
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/introduction/NOC_I_5_NoiseWalk
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=6

from p5 import *
import random

class Walker(object):

    def __init__(self):
        self.location = Vector(width / 2, height / 2)
        self.noff = Vector(random.random(), random.random())

    def display(self):
        strokeWeight(2)
        fill(127)
        stroke(0)
        ellipse(self.location.x, self.location.y, 48, 48)

    # Randomly move up, down, left, right, or stay in one place
    def walk(self):
        self.location.x = noise(float(self.noff.x)) * width #map(noise(self.noff.x), 0, 1, 0, width)
        self.location.y = noise(float(self.noff.y)) * height  #map(noise(self.noff.y), 0, 1, 0, height)
        self.noff = self.noff + Vector(0.01, 0.01, 0)


def setup():
    size(800, 200)
    # frameRate(30) # Not Implemnted in p5py

    # Create a walker object
    global w
    w = Walker()


def draw():
    background(255)
    # Run the walker object
    w.walk()
    w.display()

if __name__ == "__main__":
    run()