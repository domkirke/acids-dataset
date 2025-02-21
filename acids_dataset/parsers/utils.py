
class FileNotReadException(Exception):
    def __init__(self, file, backend):
        self.file = file
        self.backend = backend
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return f"FileNotReadException(backend={self.backend.__name__}, file={self.file})"
    