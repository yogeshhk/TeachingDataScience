from typing import List

from decorators import empty
from .library import PriorityQueue, Task


@empty
# You should implement your solution here
# You receive a list of tasks and a number of computers
def solve(l: List[Task], n: int) -> int:
    if n == 0:
        return 0
    if len(l) > 0:
        # Use a priority queue to keep track of which
        # computer is freed up next (we store this as the timestamp
        # at which the computer becomes free)
        computers = PriorityQueue()
        # Put the first n tasks in the queue
        for task in l[:n]:
            computers.enqueue(task.duration)
        # Based on the next available computer assign the next task to it
        # and add the computer back in the PQ, with a new timestamp at which it becomes free
        for task in l[n:]:
            computers.enqueue(computers.dequeue() + task.duration)
        for i in range(len(computers) - 1):
            computers.dequeue()
        # Return the last finishing task
        return computers.dequeue()
    else:
        return 0
