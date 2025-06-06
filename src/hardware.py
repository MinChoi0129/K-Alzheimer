import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    print("체크포인트 이용 >", path)
    return checkpoint
