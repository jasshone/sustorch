import nn
from nn import gradient
class SGD:
    def __init__(self, parameters,lr=0.001):
        self.lr=lr
        self.parameters = parameters
    def zero_grad(self):
        for param in self.parameters:
            param.gradient=0
            gradient[0]=0
    def step(self):
        last_step = gradient[0]
        #go from last layer to first
        for param in self.parameters[::-1]:
            last_step = param.backwards(last_step,self.lr)
class Module:
  def __call__(self,x):
    for parameter in self.parameters:
      x=parameter.forward(x)
    return x