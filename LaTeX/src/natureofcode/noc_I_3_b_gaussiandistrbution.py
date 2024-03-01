# The Nature of Code - Daniel Shiffman http://natureofcode.com
# Example I-3-b: Gaussian Distribution
# PyP5 port by: Yogesh Kulkarni
# Adopted from processing.py based implementation at:
# https://github.com/nature-of-code/noc-examples-python/blob/master/introduction/NOC_I_4_Gaussian
# Reference Youtube Video: https://www.youtube.com/watch?v=rqecAdEGW6I&list=PLRqwX-V7Uu6aFlwukCmDf0-1-uSR7mklK&index=4

from p5 import *

def setup():
    size(640, 360)
    background(255)


def draw():
    # Get a gaussian random number w/ mean of 0 and standard deviation of 1.0
    xloc = randomGaussian()
    sd = 60  # Define a standard deviation
    # Define a mean value (middle of the screen along the x-axis)
    mean = width / 2
    # Scale the gaussian random number by standard deviation and mean
    xloc = (xloc * sd) + mean
    fill(0, 10)
    noStroke()
    # Draw an ellipse at our "normal" random location
    ellipse(xloc, height / 2, 16, 16)

if __name__ == "__main__":
    run()