from collections import deque

from decorators import empty, remove
from .library import Package, PriorityPackage


class Warehouse:

    @empty
    def __init__(self):
        self.use_list = False  # Switch easily from regular list to deque
        if self.use_list:
            self.init_list()
        else:
            self.init_deque()

    @remove
    def init_deque(self):
        # Using a deque is similar to a doubly linked list.
        # We can insert and remove from both sides in O(1).
        # We should therefore prefer this option.
        self.terminal_1 = deque()
        self.terminal_2 = deque()
        self.terminal_3 = deque()

    @remove
    def init_list(self):
        # With regular lists adding and removing from the back is O(1),
        # but removing a package from the front is an O(n) operation.
        # This is not as efficient as possible, therefore not what we want.
        self.terminal_1 = []
        self.terminal_2 = []
        self.terminal_3 = []

    @remove
    def get_terminal(self, n: int):
        if n == 1:
            return self.terminal_1
        elif n == 2:
            return self.terminal_2
        else:
            return self.terminal_3

    @empty
    # Store the given package in the warehouse
    def store(self, package: Package):
        if self.use_list:
            self.store_list(package)
        else:
            self.store_deque(package)

    @remove
    def store_deque(self, package: Package):
        if isinstance(package, PriorityPackage):
            self.get_terminal(package.terminal).appendleft(package)  # This is O(1)
        else:
            self.get_terminal(package.terminal).append(package)

    @remove
    def store_list(self, package: Package):
        if isinstance(package, PriorityPackage):
            self.get_terminal(package.terminal).insert(0, package)  # This is O(n)
        else:
            self.get_terminal(package.terminal).append(package)

    @empty
    # Collect the first available package from the given terminal
    def collect(self, terminal: int):
        if self.use_list:
            return self.collect_list(terminal)
        else:
            return self.collect_deque(terminal)

    @remove
    def collect_deque(self, terminal: int):
        t = self.get_terminal(terminal)
        if len(t) == 0:
            return None
        else:
            return t.popleft()  # This is O(1)

    @remove
    def collect_list(self, terminal: int):
        t = self.get_terminal(terminal)
        if len(t) == 0:
            return None
        else:
            return t.pop(0)  # This is O(n)
