import numpy as np

class env(object):
    def __init__(self) -> None:
        pass

class Antenna(object):
    def __init__(self,n_H,n_V) -> None:
        self.steering_vetor = np.zeros(n_H, n_V)

class BS():
    def __init__(self) -> None:
        pass

class UE():
    pass

if __name__ == "__main__":
    ...