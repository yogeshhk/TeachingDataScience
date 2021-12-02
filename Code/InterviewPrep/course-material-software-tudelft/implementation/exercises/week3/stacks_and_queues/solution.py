import decorators
from .library import Stack, SpecialQueue


class Playlist:

    @decorators.empty
    # The constructor of the class. You receive a list of songs and your playlist
    # should be able to play then in the given order. Hint: you need 3
    # attributes to keep track of the current song (name it current), and the lists of songs that
    # you can play and have already played. When creating a playlist initially
    # you don't have a song playing.
    def __init__(self, songs):
        self.listenedTo = Stack()
        self.current = None
        self.toListen = SpecialQueue()
        for song in songs:
            self.toListen.enqueue_back(song)

    @decorators.empty
    # To start your application you need to hit play and the first song will
    # start playing. If you are already playing a song, hitting play should
    # not change anything
    def play(self) -> str:
        if self.current is None:
            self.current = self.toListen.dequeue()
            return self.current

    @decorators.empty
    # Once you hit next your app will start playing the next song on the list
    def next(self) -> str:
        self.listenedTo.push(self.current)
        self.current = self.toListen.dequeue()
        return self.current

    @decorators.empty
    # Once you hit previous your app should start playing the song that was before
    # the current one on the list
    def previous(self) -> str:
        self.toListen.enqueue_front(self.current)
        self.current = self.listenedTo.pop()
        return self.current
