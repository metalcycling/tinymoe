# %% Modules

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from data.projection import find_projection

# %% Constants

DEFAULT_COEFFICIENTS = [
    np.array([0.3, -0.5, 0.0, 1.0]),
    np.array([-0.3, 0.5, 0.0, -1.0]),
    np.array([0.5, 0.0, -1.5]),
    np.array([-0.5, 0.0, 1.5]),
]

# %% Classes

class PolynomialMoEDataset(Dataset):
    def __init__(self, num_samples=2000, coefficients=None, x_range=(-3.0, 3.0), y_range=(-3.0, 3.0), threshold=0.5, seed=42):
        super().__init__()

        if coefficients is None:
            coefficients = DEFAULT_COEFFICIENTS

        num_experts = len(coefficients)
        rng = np.random.default_rng(seed)

        xs = rng.uniform(x_range[0], x_range[1], size=num_samples)
        ys = rng.uniform(y_range[0], y_range[1], size=num_samples)

        points = np.stack([xs, ys], axis=1)
        distances = np.zeros((num_samples, num_experts))
        projections = np.zeros((num_samples, num_experts, 2))

        for i in range(num_samples):
            point = points[i]

            for j, coeff in enumerate(coefficients):
                px, py = find_projection(point, coeff, guess=point[0])
                projections[i, j] = [px, py]
                distances[i, j] = np.sqrt((point[0] - px) ** 2 + (point[1] - py) ** 2)

            if (i + 1) % 500 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")

        expert_labels = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(num_samples), expert_labels]
        best_projections = projections[np.arange(num_samples), expert_labels]

        self.points = torch.tensor(points, dtype=torch.float32)
        self.expert_labels = torch.tensor(expert_labels, dtype=torch.long)
        self.projections = torch.tensor(best_projections, dtype=torch.float32)
        self.distances = torch.tensor(min_distances, dtype=torch.float32)

        close_count = int((min_distances < threshold).sum())
        print(f"  Dataset: {num_samples} samples, {close_count} close ({100 * close_count / num_samples:.1f}%), {num_samples - close_count} far ({100 * (num_samples - close_count) / num_samples:.1f}%)")

        for j in range(num_experts):
            count = int((expert_labels == j).sum())
            print(f"  Expert {j}: {count} samples ({100 * count / num_samples:.1f}%)")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.expert_labels[idx], self.projections[idx], self.distances[idx]


def create_dataloader(num_samples=2000, batch_size=32, shuffle=True, **kwargs):
    dataset = PolynomialMoEDataset(num_samples=num_samples, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# %% Testing

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data.functions import polynomial

    coefficients = DEFAULT_COEFFICIENTS
    threshold = 0.5
    dataset = PolynomialMoEDataset(coefficients=coefficients, threshold=threshold)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot polynomial curves

    x_curve = np.linspace(-3.0, 3.0, 300)

    for j, coeff in enumerate(coefficients):
        y_curve = polynomial(x_curve, coeff)
        ax.plot(x_curve, y_curve, color=colors[j], linewidth=2, label=f"Expert {j}")

    # Plot sampled points: colored if close, gray if not

    points = dataset.points.numpy()
    experts = dataset.expert_labels.numpy()
    dists = dataset.distances.numpy()
    close = dists < threshold

    far_mask = ~close
    ax.scatter(points[far_mask, 0], points[far_mask, 1], c="lightgray", s=10, alpha=0.6, label="Not close")

    for j in range(len(coefficients)):
        mask = (experts == j) & close
        ax.scatter(points[mask, 0], points[mask, 1], c=colors[j], s=10, alpha=0.6)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Synthetic MoE Data")
    plt.savefig("data/synthetic_data.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% End of script
