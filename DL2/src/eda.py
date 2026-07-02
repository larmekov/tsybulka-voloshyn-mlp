# b) Exploratory Data Analysis
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from data import get_raw_dataset


def plot_class_distribution(dataset):
    labels = [label for _, label in dataset]
    counts = Counter(labels)

    plt.figure(figsize=(14, 5))
    plt.bar(
        [dataset.classes[i] for i in range(len(dataset.classes))],
        [counts[i] for i in range(len(dataset.classes))],
    )
    plt.xticks(rotation=90)
    plt.ylabel("Anzahl Bilder")
    plt.title("Klassenverteilung (Oxford-IIIT Pet)")
    plt.tight_layout()
    plt.savefig("eda_class_distribution.png")
    plt.show()

    print("Kleinste Klasse:", min(counts.values()))
    print("Größte Klasse:", max(counts.values()))
    print("Durchschnitt pro Klasse:", round(np.mean(list(counts.values())), 1))


def plot_image_sizes(dataset):
    sizes = [img.size for img, _ in dataset]  # (width, height), PIL Image
    widths, heights = zip(*sizes)

    print("Breite: min", min(widths), "max", max(widths), "mean", round(np.mean(widths), 1))
    print("Höhe:   min", min(heights), "max", max(heights), "mean", round(np.mean(heights), 1))

    plt.figure(figsize=(6, 6))
    plt.scatter(widths, heights, alpha=0.3, s=10)
    plt.xlabel("Breite (px)")
    plt.ylabel("Höhe (px)")
    plt.title("Verteilung der Bildgrößen")
    plt.tight_layout()
    plt.savefig("eda_image_sizes.png")
    plt.show()


def plot_sample_images(dataset, n_rows=2, n_cols=4, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(dataset), size=n_rows * n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 7))
    for ax, idx in zip(axes.flatten(), indices):
        img, label = dataset[idx]
        ax.imshow(img)
        ax.set_title(dataset.classes[label], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("eda_sample_images.png")
    plt.show()


def run_eda():
    dataset = get_raw_dataset(split="trainval")

    print("Anzahl Bilder (trainval):", len(dataset))
    print("Anzahl Klassen:", len(dataset.classes))
    print()

    print("== Klassenverteilung ==")
    plot_class_distribution(dataset)
    print()

    print("== Bildgrößen ==")
    plot_image_sizes(dataset)
    print()

    print("== Beispielbilder ==")
    plot_sample_images(dataset)


if __name__ == "__main__":
    run_eda()
