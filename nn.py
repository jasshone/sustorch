
import numpy as np
class Module:
  def __call__(self,x):
    for parameter in self.parameters:
      x=parameter.forward(x)
    return x
class Parameter:
  def __init__(self):
    return
  def forward(self,x):
    return
class Linear(Parameter):
  def __init__(self, in_dim, out_dim,bias=True):
    super().__init__()
    self.in_dim=in_dim
    self.out_dim=out_dim
    self.bias=bias
    if bias:
      self.ref = np.concatenate([np.zeros((1,out_dim)),np.random.rand(in_dim, out_dim)*0.01])
    else:
      self.ref=np.random.rand(in_dim, out_dim)*0.01

  def forward(self, x):
    if x.shape[-1]<self.in_dim+int(self.bias):
      x= np.concatenate([np.ones((*x.shape[:-1],1)),x],axis=-1)
    self.inp=x
    return x@self.ref
  def backwards(self, grad, lr):
    dxdw=self.inp
    self.gradient=np.transpose(dxdw)@grad
    self.ref=self.ref-lr*np.mean(self.gradient, axis=-1)[:,None]
    return self.gradient

gradient = [0]

class Loss:
  def __init__(self,value,inp,error,back):
    self.value=value
    self.inp=inp
    self.error=error
    self.back=back
  def __str__(self):
    return str(self.value)
  def __repr__(self):
    return self.value
  def __add__(self,other):
    return other+self.value
  def __radd__(self,other):
    return other+self.value
  def __iadd__(self,other):
    return other+self.value
  def __sub__(self,other):
    return other-self.value
  def backwards(self):
    gradient[0]= self.back
    return self.back
class MSELoss():
  def __call__(self,preds,correct):
    return Loss(np.mean((preds-correct)**2),preds,(preds-correct)/len(preds),(preds-correct)/len(preds))