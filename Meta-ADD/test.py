import torch
a = torch.zeros(5,4,800,1)
b = torch.zeros(5,4,800,16,1).squeeze(-1)
s = (a*b).sum(dim = 2)
uv = torch.matmul(b,s.unsqueeze(-1))
a+=uv
print(b.shape)
print(s.shape)
print(uv.shape)