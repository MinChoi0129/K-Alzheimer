import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    return checkpoint
