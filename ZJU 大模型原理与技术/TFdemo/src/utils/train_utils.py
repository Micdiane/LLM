import os
import torch

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss
    else:
        raise FileNotFoundError(f"No checkpoint found at '{filepath}'")

def log_training_progress(epoch, total_epochs, loss):
    print(f"Epoch [{epoch}/{total_epochs}], Loss: {loss:.4f}")