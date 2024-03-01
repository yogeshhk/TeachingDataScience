# Reference : https://github.com/p5py/p5/issues/199

from p5 import *

def setup():
    size(200, 200)

def draw():
    background(204)
    bezier((85, 20), (10, 10), (90, 90), (15, 80))

if __name__ == "__main__":
    run()