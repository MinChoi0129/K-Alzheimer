import torch
from config import config
from src import hardware, dataloader, model, train_and_eval

torch.backends.cudnn.benchmark = True


def main():
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = hardware.get_device()
    loaders = dataloader.get_dataloaders(config)

    if config.mode == "A":
        print("사전학습을 시작합니다.")
        train_loader, test_loader = loaders

        classifier = model.ResNet3DClassifier(config)
        classifier.to(device)
        train_and_eval.train_and_eval(classifier, train_loader, test_loader, config, device)

    elif config.mode == "B":
        print("전이학습을 시작합니다.")
        (train_loaders, train_fracs), test_loaders = loaders

        for i, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
            print("-----------------------------------------------------------------")
            print(f"전이학습 데이터 비율 : {train_fracs[i]}")
            classifier = model.ResNet3DClassifier(config)
            classifier.to(device)
            checkpoint = config.pretrained_checkpoint
            classifier.load_state_dict(hardware.load_checkpoint(checkpoint, device))
            train_and_eval.train_and_eval(classifier, train_loader, test_loader, config, device)


if __name__ == "__main__":
    main()
