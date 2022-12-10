class StringFileInterface:
    def __init__(self, filename=''):
        if filename:
            self.toFile = True
            self.f = open(filename, "w")
        else:
            self.data = ''
    def write(self, str):
        if self.toFile:
            self.f.write(str)
        else:
            self.data += str
    def close(self):
        self.f.close()
