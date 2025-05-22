import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.transform import transform_volume
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, root_dir, config, transform=None):
        self.samples = []
        self.classes = ['AD', 'CN', 'MCI']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.config = config
        self.transform = transform
        
        class_counts = {"AD": 0, "CN": 0, "MCI": 0}
        for label in self.classes:
            class_dir = os.path.join(root_dir, label)
            if os.path.exists(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith(".npz"):
                        full_path = os.path.join(class_dir, file)
                        self.samples.append((full_path, self.class_to_idx[label]))
                        class_counts[label] += 1
        
        print(class_counts)
        print("*" * 80)
    
    def __len__(self):
        return len(self.samples)
    
    def load_volume(self, file_path):
        data = np.load(file_path)
        volume = data['volume']  # shape: (224, 224, 224)
        if self.transform:
            volume = self.transform(volume)
        return volume
    
    def __getitem__(self, index):
        file_path, label = self.samples[index]
        volume = self.load_volume(file_path)
        return volume, label


def get_dataloaders(config):
    # 1) 데이터셋 준비
    root = config.root_dir_A if config.mode == 'A' else config.root_dir_B
    ds   = MRIDataset(root, config, transform=transform_volume)

    # ── 디버그 모드 ───────────────────────────────────────────
    if config.debug:
        print("디버깅 용으로 데이터셋을 감소합니다.")
        total = len(ds)
        subset_size = int(total * 0.3)
        ds, _ = random_split(ds, [subset_size, total - subset_size])
    
    # 2) A 모드: 그대로
    if config.mode == 'A':
        total_len = len(ds)
        train_len = int(0.7 * total_len)
        test_len = total_len - train_len

        train_set, test_set = random_split(ds, [train_len, test_len])

        train_loader = DataLoader(
           train_set,
            batch_size  = config.batch_size,
            shuffle     = True,
            num_workers = config.num_workers,
            pin_memory  = True
        )
        test_loader  = DataLoader(
            test_set,
            batch_size  = config.batch_size,
            shuffle     = False,
            num_workers = config.num_workers,
            pin_memory  = True
        )

        return train_loader, test_loader

    else:
        total_len = len(ds)
        test_len = int(0.7 * total_len)
        pool_len = total_len - test_len

        test_set, pool_set = random_split(ds, [test_len, pool_len])
        

        # 고정 테스트 로더
        test_loader_fixed = DataLoader(
            test_set,
            batch_size  = config.batch_size,
            shuffle     = False,
            num_workers = config.num_workers,
            pin_memory  = True
        )

        # 후보 중에서 사용할 학습 비율
        train_fracs = [1.0, 0.75, 0.5, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025, 0.0125, 0.0125 / 2, 0.0]
        train_loaders, test_loaders = [], []

        pool_indices = torch.randperm(pool_len).tolist()

        for i, frac in enumerate(train_fracs):
            train_size = int(frac * pool_len)

            if train_size == 0:
                train_loaders.append(None)
            else:
                sel_idx = pool_indices[:train_size]
                train_subset = Subset(pool_set, sel_idx)
                train_loaders.append(DataLoader(
                    train_subset,
                    batch_size  = config.batch_size,
                    shuffle     = True,
                    num_workers = config.num_workers,
                    pin_memory  = True
                ))

            # 동일한 테스트 로더를 7번 넣어 둡니다.
            test_loaders.append(test_loader_fixed)

        return (train_loaders, train_fracs), test_loaders
