from net_s3fd import s3fd
import torch
from collections import defaultdict

m = s3fd().float().eval()
x = torch.zeros((1, 3, 512, 512)).float()
#x = torch.autograd.Variable(x, volatile=True)
print(m(x))
t = torch.jit.trace(m, x)
g = t.graph
op_count = defaultdict(int)
for n in g.nodes():
    op_count[n.kind()] += 1
for k, v in op_count.items():
    print(f'{k}: {v}')
