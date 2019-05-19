import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable

from batched_train import model, vocab, provider

num_epochs = 1


for e in range(num_epochs):

    for input_batch, target_batch, sorted_lengths in provider.padded_batches():

        input_batch = Variable(input_batch)
        target_batch = Variable(target_batch)

        model.zero_grad()
        model.init_hidden(input_batch.shape[1])

        output_batch = model(input_batch, sorted_lengths) 
        output_idx = torch.argmax(output_batch, dim=2).t().tolist()
        output_seqs = [''.join([vocab.to_word(idx) for idx in batch]) for batch in output_idx]

        for s in output_seqs:
            print (s)
        
        print (vocab.to_word(vocab.padding_idx))
        print (vocab.padding_idx)
        input()



        








