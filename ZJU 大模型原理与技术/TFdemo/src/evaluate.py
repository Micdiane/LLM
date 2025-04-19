import torch
from torch.utils.data import DataLoader
from src.utils.data_utils import TextDataset
from src.models.rnn import RNN  # or import your model of choice
from src.models.transformer import Transformer  # or import your model of choice
import json

def load_model(model_path, model_class, *args):
    model = model_class(*args)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, data_loader, device):
    total_loss = 0
    total_correct = 0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, dim=-1)
            total_correct += (predicted.view(-1) == targets.view(-1)).sum().item()
            total_samples += targets.numel()

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def main():
    # Load configuration
    with open('config/default.json', 'r') as f:
        config = json.load(f)

    # Load data
    valid_dataset = TextDataset(load_data(config['valid_file']), vocab, tokenizer, config['seq_len'])
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(config['model_path'], RNN, vocab_size=len(vocab), embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim']).to(device)

    # Evaluate model
    avg_loss, accuracy = evaluate_model(model, valid_loader, device)
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()