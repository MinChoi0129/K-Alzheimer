import os


class Config:
    def __init__(self):
        # 학습 모드: 'A'는 처음부터 학습, 'B'는 전이학습(few-shot)
        self.mode = "B"
        self.debug = True

        # 데이터셋 경로 설정
        self.root_dir_A = "/home/k_alz/dataset"  # 사전학습용
        self.root_dir_B = "/home/ubuntu/alz/MRI_B"  # 전이학습용

        if self.debug:
            print("DEBUGGING MODE in Config.py")
            self.root_dir_B = self.root_dir_A  # DEBUGGING!!!

        # 3D ResNet 18 분류 모델 관련 파라미터
        self.num_classes = 3

        # 학습 파라미터 (RTX 3090, 24GB VRAM, 16코어 CPU, 64GB RAM 기준)
        self.num_epochs = 40
        self.transfer_num_peochs = 5
        self.batch_size = 16
        self.num_workers = 8
        self.learning_rate = 0.002

        # 체크포인트 경로
        self.pretrained_checkpoint = os.path.join("best_models(train)", "best_test_f1_model.pth")
        self.transfer_checkpoint = os.path.join("best_models(transfer)", "best_transfer_model.pth")

        # 시각화 저장
        self.visualize_umap = True
        self.visualize_confusion = True

        # 재현성을 위한 seed
        self.seed = 42


config = Config()
