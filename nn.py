
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
            self.bias_param=np.zeros((out_dim))
            self.ref=np.random.randn(in_dim,out_dim)

    def forward(self, x):
        self.inp=x
        out=x@self.ref
        if self.bias:
            out=(out+self.bias_param)
        return out
    def backwards(self, grad, lr):
        dxdw=self.inp
        #print(self.ref.shape)
        self.gradient=np.transpose(dxdw)@grad
        self.ref=self.ref-lr*np.mean(self.gradient, axis=-1)[:,None]
        if self.bias:
            self.bias_param=self.bias_param-lr*np.mean(grad)
        #print(self.ref.shape)
        return self.gradient
gradient = [0]
class ReLU(Parameter):
  def __init__(self):
    self.fn = lambda out: np.where(out>=0, out, 0)
  def forward(self, x):
    self.inp=x
    return self.fn(x)
  def backwards(self,grad,lr):
    dxdw=self.inp
    gradient=np.where(dxdw==0, np.transpose(grad),0)
    return gradient
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