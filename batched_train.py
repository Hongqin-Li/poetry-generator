import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable
from batched_model import RecurrentNetwork as CharRNN
from utils import DataProvider, get_vocab_and_dataset

# Hyperparameters
learning_rate = 0.01
embedding_dim = 256
hidden_size = 32
num_layers = 1
num_epochs = 100000
train_size = 0.8
batch_size = 50

vocab, dataset = get_vocab_and_dataset()
vocab_size = len(vocab)
print ('Vocab size: ', vocab_size)

provider = DataProvider(dataset, batch_size=batch_size, padding_value=vocab.padding_idx)


checkpoint_path = './checkpoints/poetry_gen_batch1.pt'

model = CharRNN(vocab_size=vocab_size,
                target_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers)
criterion = nn.NLLLoss(ignore_index=vocab.padding_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

e = 0
try:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    e = checkpoint['epoch']

    model.eval()
    print ('Load previous model and optimizer!')
except:
    print ('No saved model found!')


for epoch in range(num_epochs):

    print ('Epoch: {}'.format(epoch))

    for input_batch, target_batch, sorted_lengths in provider.padded_batches():

        input_batch = Variable(input_batch)
        target_batch = Variable(target_batch)

        model.zero_grad()
        model.init_hidden(batch_size)
        output_batch = model(input_batch, sorted_lengths) 

        loss = model.loss(output_batch, target_batch, vocab.padding_idx)
        loss.backward()
        optimizer.step()
        print ('Loss: {}'.format(loss))

    torch.save(model, checkpoint_path)     
    torch.save({
        'epoch': e + epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print ('Save model: total epochs ', e + epoch)

        








