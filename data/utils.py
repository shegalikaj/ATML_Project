class StringFileInterface:
    def __init__(self, filename=''):
        self.toFile = bool(filename)
        if self.toFile:
            self.f = open(filename, "w")
        self.data = ''
    def write(self, str):
        if self.toFile:
            self.f.write(str)
        self.data += str

    def close(self):
        if self.toFile:
            self.f.close()
