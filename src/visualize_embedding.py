# src/visualize_embedding.py
import torch
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_embeddings(model, data_loader, config, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for anchor_vol, batch_labels in data_loader:
            anchor_vol = anchor_vol.to(device)
            emb = model(anchor_vol)
            embeddings.append(emb.cpu())
            labels.extend(batch_labels)
    embeddings = torch.cat(embeddings, dim=0).numpy()

    # 2D UMAP 시각화
    reducer2d = umap.UMAP(n_components=2, random_state=config.seed)
    embedding_2d = reducer2d.fit_transform(embeddings)

    plt.figure()
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(embedding_2d[idxs, 0], embedding_2d[idxs, 1], label=label)
    plt.title("2D UMAP of Embeddings")
    plt.legend()
    plt.savefig("umap_2d.png")
    plt.show()

    # 3D UMAP 시각화
    reducer3d = umap.UMAP(n_components=3, random_state=config.seed)
    embedding_3d = reducer3d.fit_transform(embeddings)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(
            embedding_3d[idxs, 0],
            embedding_3d[idxs, 1],
            embedding_3d[idxs, 2],
            label=label,
        )
    ax.set_title("3D UMAP of Embeddings")
    plt.legend()
    plt.savefig("umap_3d.png")
    plt.show()

    model.train()
