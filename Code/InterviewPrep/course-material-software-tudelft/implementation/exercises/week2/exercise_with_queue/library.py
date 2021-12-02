class Package:
    def __init__(self, content: str, terminal: int):
        self.terminal = terminal
        self.content = content


class PriorityPackage(Package):
    def __init__(self, content: str, terminal: int):
        super().__init__(content, terminal)
