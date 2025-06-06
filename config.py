import os


class Config:
    def __init__(self):
        # 학습 모드: 'A'는 처음부터 학습, 'B'는 전이학습(few-shot)
        self.mode = "B"
        self.debug = False

        # 데이터셋 경로 설정
        self.root_dir_A = "/home/workspace/K-Alzheimer/ALL_DATASETS/dataset_adni_segmented"  # 사전학습용
        self.root_dir_B = "/home/workspace/K-Alzheimer/ALL_DATASETS/dataset_korean_processed"  # 전이학습용

        if self.debug:
            print("DEBUGGING MODE in Config.py")
            self.root_dir_B = self.root_dir_A  # DEBUGGING!!!

        # 3D ResNet 18 분류 모델 관련 파라미터
        self.num_classes = 3

        # 학습 파라미터 (RTX 3090, 24GB VRAM, 16코어 CPU, 64GB RAM 기준)
        self.num_epochs = 40
        self.transfer_num_peochs = 40
        self.batch_size = 4
        self.num_workers = 2
        self.learning_rate = 0.002
        self.stride = 2

        # 체크포인트 경로
        self.train_best_folder = os.path.join("experiments", "best_models(train)")
        self.transfer_best_folder = os.path.join("experiments", "best_models(transfer)")

        self.train_best_f1_checkpoint = os.path.join(self.train_best_folder, "best_test_f1_model.pth")
        self.train_best_loss_checkpoint = os.path.join(self.train_best_folder, "best_test_loss_model.pth")

        self.transfer_best_f1_checkpoint = os.path.join(self.transfer_best_folder, "best_test_f1_model.pth")
        self.transfer_best_loss_checkpoint = os.path.join(self.transfer_best_folder, "best_test_loss_model.pth")

        self.using_transfer_checkpoint = (
            "/home/workspace/K-Alzheimer/ALL_MODELS/ADNI_skull_stripped_2mm/best_models(train)/best_test_f1_model.pth"
        )

        # 시각화 저장
        self.visualize_umap = True
        self.visualize_confusion = True

        # 재현성을 위한 seed
        self.seed = 42


config = Config()
