import decorators
from .library import Polygon


@decorators.remove
class Triangle(Polygon):

    def __init__(self):
        Polygon.__init__(self, 3)

    def find_area(self) -> float:
        sides = self.get_sides()
        s = (sides[0] + sides[1] + sides[2]) / 2
        area = (s * (s - sides[0]) * (s - sides[1]) * (s - sides[2])) ** 0.5
        return round(area, 1)
