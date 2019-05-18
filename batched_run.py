import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable
from batched_model import RecurrentNetwork as CharRNN
from utils import DataProvider, get_vocab_and_dataset

# Hyperparameters
embedding_dim = 256
hidden_size = 32
num_layers = 1
batch_size = 50
num_epochs = 100

vocab, dataset = get_vocab_and_dataset()
vocab_size = len(vocab)
print ('Vocab size: ', vocab_size)

provider = DataProvider(dataset, batch_size=batch_size, padding_value=vocab.padding_idx)


checkpoint_path = './checkpoints/poetry_gen_batch.pt'

model = CharRNN(vocab_size=vocab_size,
                target_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

for e in range(num_epochs):

    for input_batch, target_batch, sorted_lengths in provider.padded_batches():

        input_batch = Variable(input_batch)
        target_batch = Variable(target_batch)

        model.zero_grad()
        model.init_hidden(batch_size)
        output_batch = model(input_batch, sorted_lengths) 
        output_idx = torch.argmax(output_batch, dim=2).t().tolist()
        output_seqs = [''.join([vocab.to_word(idx) for idx in batch]) for batch in output_idx]


        for s in output_seqs:
            print (s)
        
        print (vocab.to_word(vocab.padding_idx))
        print (vocab.padding_idx)
        input()



        








