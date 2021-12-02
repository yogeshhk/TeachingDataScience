import unittest

from decorators import spectest, yourtest, timeout, remove, solution_only
from weblabTestRunner import TestRunner
from .solution import Playlist

if solution_only:
    songs = ["Raise Your Banner - Within Temptation", "Pearl in a World of Dirt - Beyond the Black",
             "Bring Me to Life - Evanescence", "I Am the Fire - Halestorm", "All Star - Smash Mouth",
             "Never Gonna Give You Up - Rick Astley", "That's mathematics - Tom Lehrer",
             "Finite simple group of order two - Klein Group", "Labochevsky - Tom Lehrer",
             "A song about a circle constant - Vihart", "An awful lot of running - Chameleon Circuit",
             "The Element Song - Tom Lehrer", "Wonderwall - Oasis", "One - U2",
             "Californication - Red Hot Chili Peppers"]


@remove
def helper(p):
    for i in range(13):
        p.next()
    for i in range(13):
        p.previous()


class TestSuite(unittest.TestCase):
    @yourtest
    @spectest(1)
    def test_play(self):
        x = ["That's mathematics - Tom Lehrer", "Finite simple group of order two - Klein Group",
             "Labochevsky - Tom Lehrer", "A song about a circle constant - Vihart",
             "An awful lot of running - Chameleon Circuit", "The Element Song - Tom Lehrer"]
        playlist = Playlist(x)
        self.assertEqual(playlist.play(), "That's mathematics - Tom Lehrer")

    @spectest(1)
    def test_number_attributes(self):
        p = Playlist(songs)
        self.assertEqual(len(p.__dict__), 3)

    @spectest(1)
    def test_no_play(self):
        p = Playlist(songs)
        self.assertEqual(p.current, None)

    @spectest(1)
    def test_play_next(self):
        p = Playlist(songs)
        p.play()
        self.assertEqual(p.next(), "Pearl in a World of Dirt - Beyond the Black")

    @spectest(1)
    def test_next_twice(self):
        p = Playlist(songs)
        p.play()
        p.next()
        self.assertEqual(p.next(), "Bring Me to Life - Evanescence")

    @spectest(1)
    def test_next_previous(self):
        p = Playlist(songs)
        p.play()
        p.next()
        p.next()
        p.next()
        self.assertEqual(p.previous(), "Bring Me to Life - Evanescence")

    @spectest(1)
    def test_next_previous_next(self):
        p = Playlist(songs)
        p.play()
        p.next()
        p.previous()
        self.assertEqual(p.next(), "Pearl in a World of Dirt - Beyond the Black")

    @spectest(1)
    def test_previous_twice(self):
        p = Playlist(songs)
        p.play()
        for i in range(5):
            p.next()
        p.previous()
        self.assertEqual(p.previous(), "I Am the Fire - Halestorm")

    @spectest(1)
    @timeout(1)
    def test_big_efficiency(self):
        p = Playlist(songs)
        p.play()
        for i in range(23000):
            helper(p)
        for i in range(13):
            p.next()
        self.assertEqual(p.next(), "Californication - Red Hot Chili Peppers")


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
