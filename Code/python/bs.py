from typing import List


def bubblesort(xs: List[int]):
    for i in range(len(xs)):
        for j in range(len(xs)-1):
            if xs[j] > xs[j+1]:
                xs[j+1], xs[j] = xs[j], xs[j+1]
