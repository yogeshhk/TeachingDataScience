from decorators import remove
from .library import Clock


@remove
# Custom exception that is thrown when an attempt is made to decrease time manually.
class NegativeTimeError(Exception):
    pass


@remove
# Clock that does not allow decreasing time manually
class DigitalClock(Clock):
    def __init__(self):
        Clock.__init__(self)

    def add_time(self, time: int):
        if time >= 0:
            Clock.add_time(self, time)
        else:
            raise NegativeTimeError("The clock time cannot be decreased manually, entered time: " + str(time))
