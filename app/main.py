import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# 2次元のデータを生成
np.random.seed(0)
X = np.random.randn(300, 2)

clauter_counts = [3, 4, 5]

for _i, clauter_count in enumerate(clauter_counts):
    # K-means法を適用
    kmeans = KMeans(n_clusters=clauter_count, random_state=0).fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 結果をプロット
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x")
    plt.title(f"K-means Clustering (cluster_count={clauter_count})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.savefig(f"./public/kmeans-clustering-{clauter_count}.png")
    plt.clf()
