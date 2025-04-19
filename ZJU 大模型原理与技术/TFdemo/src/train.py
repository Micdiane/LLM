import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.rnn import RNN  # Assuming RNN model is implemented in rnn.py
from utils.data_utils import TextDataset  # Assuming TextDataset is implemented in data_utils.py
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Configuration
train_file = "../data/wiki.train.txt"
valid_file = "../data/wiki.valid.txt"
test_file = "../data/wiki.test.txt"
seq_len = 32
batch_size = 64
embedding_dim = 32
hidden_dim = 64
num_epochs = 10
learning_rate = 0.001

# Tokenizer and Vocabulary
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(load_data(train_file)), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
vocab.set_default_index(vocab["<unk>"])

# Load Data
train_dataset = TextDataset(load_data(train_file), vocab, tokenizer, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize Model
model = RNN(vocab_size=len(vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim)
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        output = model(inputs)  # [batch_size, seq_len, vocab_size]
        loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "language_model.pth")