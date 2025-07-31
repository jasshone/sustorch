import sklearn
from sklearn.datasets import fetch_california_housing
import nn
import optim
from nn import MSELoss, Module, Linear
from optim import SGD
import numpy as np
class Model(Module):
  def __init__(self,in_dim,out_dim):
    super().__init__()
    self.parameters=[Linear(in_dim,out_dim,True)]


  def parameters(self):
    return self.parameters
housing = fetch_california_housing()
a,b=len(housing["data"][0]),1
model = Model(a,b)
loss_fn=MSELoss()
optim=SGD(model.parameters)
housing["data"]=housing["data"]/np.max(housing["data"])
housing["target"]=housing["target"]/np.max(housing["target"])
epochs=100
lr=0.01
B=64
X,y = (housing["data"],housing["target"])
for i in range(epochs):
    for batchidx in range(0, len(X), B):
      xx,yy=X[batchidx:batchidx+B],y[batchidx:batchidx+B]
      #print(xx,yy)
      #print(xx@model.parameters[0].ref)
      #print(model(xx).shape)
      loss = loss_fn(model(xx),yy)


      optim.zero_grad()
      loss.backwards()
      optim.step()
      #print(gradient[0].shape)
      #break
    if i%10==0:
        total_loss=0
        for x,correct in zip(housing["data"][:B],housing["target"][:B]):
            total_loss+= loss_fn(model(x),correct)
        print(i,total_loss)