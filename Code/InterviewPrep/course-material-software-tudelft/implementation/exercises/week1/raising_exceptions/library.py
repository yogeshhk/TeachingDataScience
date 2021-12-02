class Clock:
    def __init__(self):
        self.time = 0

    # Adds the given time to the clock
    def add_time(self, time: int):
        self.time += time

    # Decreases time
    def tick(self):
        if self.time > 0:
            self.time -= 1

    # Resets time
    def reset(self):
        self.time = 0

    # Returns true if the timer is at 0
    def is_ready(self) -> bool:
        return self.time == 0
