import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

"""
Function for quantization
reference: https://github.com/zzzxxxttt/pytorch_DoReFaNet/tree/master

Here, I implemented deterministic quantization, which quantizes a given input to the nearest.
I tired to implement the stochastic version, but it took too much time for training (I was not able to vertorize stochastic operations in python). 
According to "I.Hubara, (2016)," these two quantization schemes eventually work in the almost same way (the stochastic version generalizes a bit better).
Therefore, for fast training, I adopted the deterministic version for this proof of concept implementation.
"""
def uniform_quantize(n_bit):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if n_bit == 32:
        out = input
      elif n_bit == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** (n_bit - 1))
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, n_bit):
    super(weight_quantize_fn, self).__init__()
    # assert w_bit <= 8 or w_bit == 32
    self.n_bit = n_bit
    self.uniform_q = uniform_quantize(n_bit = n_bit)

  def forward(self, x):
    if self.n_bit == 32:
      weight_q = x
    else:
      weight = torch.clamp(x, min =-1, max = 1) #It clips an input to [-1, 1]
      weight_q = self.uniform_q(weight)
    return weight_q


class activation_quantize_fn(nn.Module):
  def __init__(self, n_bit):
    super(activation_quantize_fn, self).__init__()
    self.n_bit = n_bit
    self.uniform_q = uniform_quantize(n_bit=n_bit)

  def forward(self, x):
    if self.n_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(F.leaky_relu(x))
    return activation_q


class Conv2d_Q(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True):
    super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                    padding, dilation, groups, bias)
    self.n_bit = None
    self.quantize_fn = None

  def set_quantization_level(self, n_bit):
    self.n_bit = n_bit
    self.quantize_fn = weight_quantize_fn(n_bit=n_bit)

  def forward(self, input, order=None):
    weight_q = self.quantize_fn(self.weight)
    return F.conv2d(input, weight_q, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)


class Linear_Q(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super(Linear_Q, self).__init__(in_features, out_features, bias)
    self.n_bit = None
    self.quantize_fn = None

  def set_quantization_level(self, n_bit):
    self.n_bit = n_bit
    self.quantize_fn = weight_quantize_fn(n_bit=n_bit)
    
  def forward(self, input):
    weight_q = self.quantize_fn(self.weight)
    # print(np.unique(weight_q.detach().numpy()))
    return F.linear(input, weight_q, self.bias)