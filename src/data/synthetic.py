# src/data/synthetic.py
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

#generate synthetic dataset of Gaussian blobs
def generate_blob_dataset(n_samples=200, n_features=2, n_clusters=2, cluster_std=1.0, random_state=42, to_csv=None):
    # Returns: X (pd.DataFrame): Feature matrix with cluster labels as y.

    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])
    df["label"] = y

    if to_csv:
        df.to_csv(to_csv, index=False)

    return df

#generate synthetic dataset of Gaussian blobs WITH anomalies injected
def generate_blob_dataset_with_anomalies(n_samples=200, n_features=2, n_clusters=2, cluster_std=1.0, random_state=11, to_csv=None, n_anomalies=10, anomaly_range=(-10, 10)):
    # Returns: X (pd.DataFrame): Feature matrix with cluster labels as y.

    rng = np.random.RandomState(random_state)

    # Normal clustered data
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    labels = np.zeros(n_samples, dtype=int)  # 0 = normal

    # Anomaly data (spread out uniformly)
    anomalies = rng.uniform(
        low=anomaly_range[0],
        high=anomaly_range[1],
        size=(n_anomalies, n_features),
    )
    anomaly_labels = np.ones(n_anomalies, dtype=int)  # 1 = anomaly

    # Combine normal + anomalies
    X_full = np.vstack([X, anomalies])
    y_full = np.hstack([labels, anomaly_labels])

    # Build DataFrame
    df = pd.DataFrame(X_full, columns=[f"feature_{i+1}" for i in range(n_features)])
    df["label"] = y_full

    if to_csv:
        df.to_csv(to_csv, index=False)

    return df


if __name__ == "__main__":
    df = generate_blob_dataset(
        n_samples=300,
        n_features=2,
        n_clusters=2,
        cluster_std=0.8,
        to_csv="data/synthetic_blobs.csv"
    )
    print("Synthetic dataset saved to data/synthetic_blobs.csv")
    print(df.head())

    df_anomalies = generate_blob_dataset_with_anomalies(
        n_samples=300,
        n_features=2,
        n_clusters=2,
        cluster_std=0.8,
        to_csv="data/synthetic_blobs_anomalies.csv"
    )
    print("Synthetic dataset with anomalies saved to data/synthetic_blobs_anomalies.csv")
    print(df_anomalies.head())
