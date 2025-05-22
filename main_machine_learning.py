#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from config import config
from src.dataloader import MRIDataset
from src.transform import transform_volume


def batch_indices(arr, batch_size):
    """인덱스 배열을 배치 단위로 분할"""
    for i in range(0, len(arr), batch_size):
        yield arr[i : i + batch_size]


def load_batch(dataset, idxs):
    """
    주어진 인덱스 리스트로부터 데이터 배치를 로드하고 flatten한 ndarray 반환
    """
    batch_size = len(idxs)
    # 첫 볼륨 크기로 초기화
    vol0, _ = dataset[idxs[0]]
    vol_np0 = vol0.numpy() if hasattr(vol0, "numpy") else vol0
    flat_size = int(np.prod(vol_np0.shape))
    X = np.zeros((batch_size, flat_size), dtype=np.float32)
    y = np.zeros(batch_size, dtype=int)
    for j, idx in enumerate(idxs):
        vol, label = dataset[idx]
        vol_np = vol.numpy() if hasattr(vol, "numpy") else vol
        X[j] = vol_np.astype(np.float32).flatten()
        y[j] = int(label)
    return X, y


def main():
    # 데이터셋 로드
    dataset_dir = config.root_dir_A if config.mode == "A" else config.root_dir_B
    dataset = MRIDataset(dataset_dir, config, transform=transform_volume)
    N = len(dataset)

    # 전체 레이블 목록 추출
    labels = np.array([dataset.samples[i][1] for i in range(N)])

    # train/test split
    indices = np.arange(N)
    train_idx, test_idx = train_test_split(indices, stratify=labels, test_size=0.2, random_state=42)
    n_train, n_test = len(train_idx), len(test_idx)

    # IncrementalPCA 학습
    n_components = 50
    print("🔍 IncrementalPCA 학습 중...")
    ipca = IncrementalPCA(n_components=n_components)
    for batch_idx in tqdm(batch_indices(train_idx, batch_size=100)):
        X_batch, _ = load_batch(dataset, batch_idx)
        ipca.partial_fit(X_batch)

    # PCA 변환 결과를 메모리 내 ndarray로 생성
    X_train = np.zeros((n_train, n_components), dtype=np.float32)
    X_test = np.zeros((n_test, n_components), dtype=np.float32)
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    print("🔄 PCA 변환 중...")
    # train 변환
    for i, batch_idx in enumerate(tqdm(batch_indices(train_idx, batch_size=100))):
        X_batch, _ = load_batch(dataset, batch_idx)
        X_train[i * 100 : (i * 100 + len(batch_idx))] = ipca.transform(X_batch)
    # test 변환
    for i, batch_idx in enumerate(tqdm(batch_indices(test_idx, batch_size=100))):
        X_batch, _ = load_batch(dataset, batch_idx)
        X_test[i * 100 : (i * 100 + len(batch_idx))] = ipca.transform(X_batch)

    # 모델 정의
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SGDClassifier": SGDClassifier(loss="log_loss", max_iter=1000, random_state=42),
    }

    # 학습 및 평가
    for name, model in models.items():
        print(f"🧠 {name} 학습 및 평가 중...")
        if hasattr(model, "partial_fit") and name == "SGDClassifier":
            model.partial_fit(X_train, y_train, classes=np.unique(y_train))
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=["치매(0)", "정상(1)", "경증치매(2)"])
        print(f"\n📌 {name} 성능:\n{report}")


if __name__ == "__main__":
    main()
