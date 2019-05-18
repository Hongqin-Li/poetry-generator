import torch
from torch import nn

# (vocab_size, embed_size)
embedding = nn.Embedding(1000,128)
# 3, 4, 5 is vocab idx
x = embedding(torch.LongTensor([[3, 4], [3, 5]]))

print(x.shape) 
