# Optimizer.py

class Optimizer_Base:
    def __init__(self):
        pass

    def update(self):
        pass

class GradientDescent(Optimizer_Base):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

    def update(self,gradient):
        return -self.learning_rate*gradient
