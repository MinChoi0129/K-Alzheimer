import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.transform import transform_volume
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
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
        rng = np.random.RandomState(42)
        selected_idxs = rng.permutation(total)[:subset_size]
        ds = Subset(ds, selected_idxs)

    N    = len(ds)
    idxs = np.arange(N)

    # 2) A 모드: 그대로
    if config.mode == 'A':
        rng  = np.random.RandomState(42)
        perm = rng.permutation(idxs)
        split = int(0.7 * N)
        train_idx, test_idx = perm[:split], perm[split:]

        train_loader = DataLoader(
            Subset(ds, train_idx),
            batch_size  = config.batch_size,
            shuffle     = True,
            num_workers = config.num_workers,
            pin_memory  = True
        )
        test_loader  = DataLoader(
            Subset(ds, test_idx),
            batch_size  = config.batch_size,
            shuffle     = False,
            num_workers = config.num_workers,
            pin_memory  = True
        )
        return train_loader, test_loader

    # 3) B 모드 ────────────────────────────────────────────────
    rng  = np.random.RandomState(42)
    perm = rng.permutation(idxs)

    test_size       = int(0.80 * N)          # 80 % 고정
    test_idx        = perm[:test_size]
    train_pool_idx  = perm[test_size:]       # 20 % 학습 후보
    M               = len(train_pool_idx)

    # 고정 테스트 로더
    test_loader_fixed = DataLoader(
        Subset(ds, test_idx),
        batch_size  = config.batch_size,
        shuffle     = False,
        num_workers = config.num_workers,
        pin_memory  = True
    )

    # 후보 중에서 사용할 학습 비율
    train_fracs = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
    train_loaders, test_loaders = [], []

    for i, frac in enumerate(train_fracs):
        rng_i  = np.random.RandomState(42 + i)
        perm_i = rng_i.permutation(train_pool_idx)
        train_size = int(frac * M)

        if train_size == 0:
            train_loaders.append(None)
        else:
            t_idx = perm_i[:train_size]
            train_loaders.append(DataLoader(
                Subset(ds, t_idx),
                batch_size  = config.batch_size,
                shuffle     = True,
                num_workers = config.num_workers,
                pin_memory  = True
            ))

        # 동일한 테스트 로더를 7번 넣어 둡니다.
        test_loaders.append(test_loader_fixed)

    return (train_loaders, train_fracs), test_loaders
