from random import shuffle
import numpy as np

class Ring:
    def __init__(self, data):
        self._data = data

    def __repr__(self):
        return repr(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    def add(self, d):
        self._data.append(d)

    def turn(self):
        last = self._data.pop(-1)
        self._data.insert(0, last)

    def last(self):
        last = self._data[-1]
        self.turn()

        return last