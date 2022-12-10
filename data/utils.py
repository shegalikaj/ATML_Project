class StringFileInterface:
    def __init__(self):
        self.data = ''
    def write(self, str):
        self.data += str
