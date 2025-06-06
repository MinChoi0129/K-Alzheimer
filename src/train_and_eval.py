import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)


def train_and_eval(model, train_loader, test_loader, config, device):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_test_loss = float("inf")
    best_test_f1 = 0.0

    print(
        "train/test 개수 :",
        len(train_loader.dataset) if train_loader is not None else 0,
        len(test_loader.dataset),
    )
    epochs = config.num_epochs if config.mode == "A" else config.transfer_num_peochs
    for epoch in range(epochs):
        # 🔹 학습 루프 건너뛰기 예외 처리
        if train_loader is None:
            print(f"Epoch {epoch+1}/{epochs}: 학습 데이터 없음 → 학습 건너뜁니다.")
        else:
            model.train()
            train_loss_epoch = 0.0
            all_train_preds = []
            all_train_labels = []
            num_samples = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for volumes, labels, sexs, ages in pbar:
                volumes = volumes.float().to(device)
                labels = labels.to(device)
                sexs = sexs.to(device)
                ages = ages.float().to(device)

                class_logits, combined_emb = model(volumes, sexs, ages)  # (batch, num_classes)
                loss = criterion(class_logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bsz = volumes.size(0)
                train_loss_epoch += loss.item() * bsz
                num_samples += bsz

                preds = torch.argmax(class_logits, dim=1)
                all_train_preds.append(preds.cpu().numpy())
                all_train_labels.append(labels.cpu().numpy())

                batch_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
                batch_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="weighted")
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{batch_acc:.4f}",
                        "f1": f"{batch_f1:.4f}",
                    }
                )

            avg_train_loss = train_loss_epoch / num_samples
            print(f"\nEpoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # ✅ 평가
        if config.mode == "A":
            if epoch == 1 or epoch >= 5:
                os.makedirs(config.train_best_folder, exist_ok=True)
                test_results, print_str = evaluate(model, test_loader, config, device, criterion, epoch=epoch + 1)
                if test_results["test_loss"] < best_test_loss:
                    best_test_loss = test_results["test_loss"]
                    torch.save(model.state_dict(), config.train_best_loss_checkpoint)
                    print("Best test loss model saved on 'Training'.")
                if test_results["test_f1"] > best_test_f1:
                    print(print_str)
                    best_test_f1 = test_results["test_f1"]
                    torch.save(model.state_dict(), config.train_best_f1_checkpoint)
                    print("Best test f1 model saved on 'Training'.")
            else:
                print("초기 단계이므로 Evaluation을 건너뜁니다.")
        else:
            test_results, print_str = evaluate(model, test_loader, config, device, criterion, epoch=epoch + 1)
            os.makedirs(config.transfer_best_folder, exist_ok=True)
            print("test_loss :", test_results["test_loss"])
            if test_results["test_loss"] < best_test_loss:
                best_test_loss = test_results["test_loss"]
                torch.save(model.state_dict(), config.transfer_best_loss_checkpoint)
                print("Best test loss model saved on 'Transfer'.")
            if test_results["test_f1"] > best_test_f1:
                print(print_str)
                best_test_f1 = test_results["test_f1"]
                torch.save(model.state_dict(), config.transfer_best_f1_checkpoint)
                print("Best test f1 model saved on 'Transfer'.")

    print(f"{'사전' if config.mode == 'A' else '전이'} 학습 종료.")


def evaluate(model, eval_loader, config, device, criterion, epoch=None):
    model.eval()
    all_outputs = []
    all_labels = []
    all_embeddings = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        pbar = tqdm(eval_loader, desc="Evaluating")
        for volumes, labels, sexs, ages in pbar:
            volumes = volumes.float().to(device)
            labels = labels.to(device)
            sexs = sexs.to(device)
            ages = ages.float().to(device)

            class_logits, combined_emb = model(volumes, sexs, ages)  # (batch, num_classes)

            loss = criterion(class_logits, labels)

            total_loss += loss.item() * volumes.size(0)
            num_samples += volumes.size(0)
            probs = F.softmax(class_logits, dim=1)
            all_outputs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_embeddings.append(combined_emb.cpu().numpy())

    avg_loss = total_loss / num_samples
    all_outputs = np.concatenate(all_outputs, axis=0)  # shape: (N, num_classes)
    all_labels = np.concatenate(all_labels, axis=0)  # shape: (N,)
    preds = np.argmax(all_outputs, axis=1)

    pred_counts = {0: 0, 1: 0, 2: 0}
    for p in preds:
        pred_counts[p] += 1
    print("Prediction Counts :", pred_counts)

    overall_acc = accuracy_score(all_labels, preds)
    overall_f1 = f1_score(all_labels, preds, average="weighted", zero_division=0)
    # confusion matrix 및 classification report
    conf_matrix = confusion_matrix(all_labels, preds)
    class_report = classification_report(
        all_labels, preds, target_names=["AD", "CN", "MCI"], output_dict=True, zero_division=0
    )

    # ROC AUC (multiclass; one-hot encoding 사용)
    try:
        all_labels_onehot = np.eye(3)[all_labels]  # assuming 3 classes
        roc_auc = roc_auc_score(all_labels_onehot, all_outputs, multi_class="ovr")
    except Exception as e:
        roc_auc = None

    precision = precision_score(all_labels, preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, preds, average="weighted", zero_division=0)

    # 결과 딕셔너리 구성
    results = {
        "test_loss": avg_loss,
        "test_acc": overall_acc,
        "test_f1": overall_f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
    }

    # UMAP visualization (만약 config.visualize_umap == True 인 경우)
    if config.visualize_umap:
        try:
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(np.concatenate(all_embeddings, axis=0))

            plt.figure(figsize=(8, 6))

            # 고유 클래스 목록
            classes = np.unique(all_labels)
            colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

            for i, cls in enumerate(classes):
                indices = all_labels == cls
                plt.scatter(
                    embedding[indices, 0],
                    embedding[indices, 1],
                    color=colors[i],
                    label=f"Class {cls} ({'AD' if cls==0 else 'CN' if cls==1 else 'MCI'})",
                    s=5,
                )

            plt.legend(title="Class", loc="best", markerscale=3)

            title = f"UMAP of outputs at epoch {epoch}" if epoch is not None else "UMAP of outputs"
            plt.title(title)

            metric_image_path = (
                "experiments/metric_images(train)" if config.mode == "A" else "experiments/metric_images(transfer)"
            )
            os.makedirs(f"{metric_image_path}/umap_plots", exist_ok=True)
            filename = (
                os.path.join(f"{metric_image_path}/umap_plots", f"umap_epoch_{epoch}.png")
                if epoch is not None
                else f"{metric_image_path}/umap_plots/umap_final.png"
            )
            plt.savefig(filename)
            plt.close()
            results["umap_plot"] = filename

        except Exception as e:
            print("UMAP visualization skipped due to error:", e)

    if config.visualize_confusion:
        try:
            confusion_matric_path = (
                "experiments/metric_images(train)" if config.mode == "A" else "experiments/metric_images(transfer)"
            )
            os.makedirs(f"{confusion_matric_path}/confusion_plots", exist_ok=True)
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["AD", "CN", "MCI"],
                yticklabels=["AD", "CN", "MCI"],
            )
            title = f"Confusion Matrix - epoch {epoch}" if epoch is not None else "Confusion Matrix - final"
            plt.title(title)
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            filename = (
                os.path.join(
                    f"{confusion_matric_path}/confusion_plots",
                    f"confusion_epoch_{epoch}.png",
                )
                if epoch is not None
                else f"{confusion_matric_path}/confusion_plots/confusion_final.png"
            )
            plt.savefig(filename)
            plt.close()
            results["confusion_plot"] = filename
        except Exception as e:
            print("Confusion matrix visualization skipped due to error:", e)

    print_str = "*******************************************************************************"
    print_str += f"\nEpoch: {epoch} Test Results:\n"
    print_str += f"Loss: {avg_loss:.4f}, Acc: {overall_acc:.4f}, F1: {overall_f1:.4f}\n"
    print_str += f"Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {roc_auc:.4f}\n"
    print_str += "\nClassification Report per class:\n"
    for cls in ["AD", "CN", "MCI"]:
        print_str += f"{cls}: {class_report.get(cls)}\n"

    return results, print_str
