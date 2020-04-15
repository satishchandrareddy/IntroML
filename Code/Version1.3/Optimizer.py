'''
Class Optimizer
'''
class Optimizer_Base:
    def __init__(self):
        pass
    def update_params(self):
        pass

class GradientDescent(Optimizer_Base):
    def __init__(self,learning_rate):
        self.label = "GradientDescent"
        self.learning_rate = learning_rate

    def update_params(self,model):
        for layer in range(model.n_layer):
            model.update_params(layer,"W",-self.learning_rate*model.get_params(layer,"params_der","W"))
            model.update_params(layer,"b",-self.learning_rate*model.get_params(layer,"params_der","b"))

if __name__ == "__main__":
    pass