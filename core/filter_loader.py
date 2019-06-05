import numpy as np
import pickle

class filter_loader:
    def __init__(self, file_name):
        self.file_name = file_name
        self.filter = None
    def __call__(self):
        if self.filter is None:
            with open(self.file_name, "rb") as f:
                self.filter = pickle.load(f)
        return self.filter
