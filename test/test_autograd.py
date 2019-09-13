import torch
from torch import autograd

# this file shows that auto grad will fail to compute the grad at negative xs

x = torch.tensor(-800.0,requires_grad=True)
beta = 1
# y = torch.exp(-beta*x)
# z = (1 + y) ** (-1)
if x >=0:
    y = torch.exp(-beta*x)
    z = (1 + y) ** (-1)
else:
    y = torch.exp(beta*x)
    z = 1 - (1 + y) ** (-1)

y.retain_grad()

with autograd.detect_anomaly():
    z.backward()
    print(z)
    print(y.grad)
    print(x.grad)