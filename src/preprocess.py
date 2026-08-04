import argparse
import glob
import json
import os
import random
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from PIL import Image
from torch.utils.data import ConcatDataset
from torchvision.transforms import v2

from src.generate import load_pipeline, generate_images, create_image_grid

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_imbalanced_dataset(dataset, target_class, keep_fraction):

    target_indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]

    num_to_keep = int(len(target_indices) * keep_fraction)
    kept_target_indices = random.sample(target_indices, num_to_keep)

    other_indices = [i for i, (_, label) in enumerate(dataset) if label != target_class]

    final_indices = kept_target_indices + other_indices

    return torch.utils.data.Subset(dataset, final_indices)

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, images, label, transform):
        self.images = images
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.label

def train_and_evaluate(trainset, testloader, device, target_class, num_epochs=5, lr=1e-4, train_batch_size=64, checkpoint_path=None):
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=0)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    # Evaluate per-class accuracy on cat class specifically
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            mask = labels == target_class
            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    return correct / total if total > 0 else 0

def train_and_evaluate_avg(trainset, testloader, device, target_class, num_epochs=5, num_runs=3, lr=1e-4, train_batch_size=64, seed=42, checkpoint_dir=None, run_label="run"):
    accuracies = []
    for run in range(num_runs):
        set_seed(seed + run)
        print(f"  Run {run+1}/{num_runs}...")
        checkpoint_path = os.path.join(checkpoint_dir, f"{run_label}_run{run+1}.pt") if checkpoint_dir else None
        acc = train_and_evaluate(trainset, testloader, device, target_class, num_epochs, lr=lr, train_batch_size=train_batch_size, checkpoint_path=checkpoint_path)
        accuracies.append(acc)
    mean = sum(accuracies) / len(accuracies)
    std = statistics.stdev(accuracies)
    print(f"  Mean: {mean:.3f} ± {std:.3f}")
    return mean, std

@dataclass
class ExperimentConfig:
    target_class: int = 3
    first_fraction: float = 0.1
    target_fractions: list = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0, 2.0])
    test_batch_size: int = 4
    train_batch_size: int = 64
    repo_id: str = "Ketansomewhere/cifar10_conditional_diffusion1"
    save_dir: str = "generated_images"
    data_dir: str = "./data"
    num_epochs: int = 5
    num_runs: int = 3
    lr: float = 1e-4
    seed: int = 42
    output_dir: str = "results"
    save_checkpoints: bool = False


def parse_args(argv=None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Diffusion-based data augmentation for a class-imbalanced CIFAR-10 subset."
    )
    parser.add_argument("--target-class", type=int, default=3, help="CIFAR-10 class index to imbalance (default: 3, cat)")
    parser.add_argument("--first-fraction", type=float, default=0.1, help="Fraction of the target class kept in the imbalanced baseline")
    parser.add_argument("--target-fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 2.0], help="Synthetic data fractions to evaluate")
    parser.add_argument("--test-batch-size", type=int, default=4, help="Batch size for the evaluation dataloader")
    parser.add_argument("--train-batch-size", type=int, default=64, help="Batch size for the training dataloader")
    parser.add_argument("--repo-id", type=str, default="Ketansomewhere/cifar10_conditional_diffusion1", help="Hugging Face repo id for the class-conditional DDPM")
    parser.add_argument("--save-dir", type=str, default="generated_images", help="Directory for cached/generated synthetic images")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for the CIFAR-10 dataset")
    parser.add_argument("--num-epochs", type=int, default=5, help="Training epochs per run")
    parser.add_argument("--num-runs", type=int, default=3, help="Independent runs averaged per condition")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save each run's config/metrics/plot artifacts")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save trained model weights for every run under checkpoints/")
    args = parser.parse_args(argv)
    return ExperimentConfig(
        target_class=args.target_class,
        first_fraction=args.first_fraction,
        target_fractions=args.target_fractions,
        test_batch_size=args.test_batch_size,
        train_batch_size=args.train_batch_size,
        repo_id=args.repo_id,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        num_epochs=args.num_epochs,
        num_runs=args.num_runs,
        lr=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
        save_checkpoints=args.save_checkpoints,
    )


def build_transform() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def build_dataloaders(config: ExperimentConfig, transform: v2.Compose):
    trainset = torchvision.datasets.CIFAR10(root=config.data_dir, train=True,
                                             download=True, transform=transform)
    imbalanced_trainset = create_imbalanced_dataset(trainset, config.target_class, config.first_fraction)
    testset = torchvision.datasets.CIFAR10(root=config.data_dir, train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size,
                                              shuffle=False, num_workers=0)
    return trainset, imbalanced_trainset, testloader


def load_or_prepare_synthetic_source(config: ExperimentConfig, device):
    """Returns cached image paths if generated_images/ is already populated,
    otherwise loads the diffusion pipeline so new images can be generated."""
    cached_files = sorted(glob.glob(os.path.join(config.save_dir, "*.png")))
    if cached_files:
        print(f"Loading {len(cached_files)} cached images from disk...")
        return cached_files, None
    pipeline = load_pipeline(config.repo_id, device)
    return [], pipeline


def run_experiment(config: ExperimentConfig, device) -> dict:
    set_seed(config.seed)
    transform = build_transform()

    trainset, imbalanced_trainset, testloader = build_dataloaders(config, transform)
    cached_files, pipeline = load_or_prepare_synthetic_source(config, device)
    skip_generation = pipeline is None

    checkpoint_dir = "checkpoints" if config.save_checkpoints else None

    results = {}

    print("Training upper bound (full balanced data)...")
    upper_bound_acc, upper_bound_std = train_and_evaluate_avg(
        trainset, testloader, device, config.target_class, config.num_epochs,
        num_runs=config.num_runs, lr=config.lr, train_batch_size=config.train_batch_size, seed=config.seed,
        checkpoint_dir=checkpoint_dir, run_label="upper_bound",
    )
    results[-1] = (upper_bound_acc, upper_bound_std)
    print(f"Upper bound cat accuracy (full data): {upper_bound_acc:.3f}")

    print("Training baseline (no synthetic data)...")
    baseline_acc, baseline_std = train_and_evaluate_avg(
        imbalanced_trainset, testloader, device, config.target_class, config.num_epochs,
        num_runs=config.num_runs, lr=config.lr, train_batch_size=config.train_batch_size, seed=config.seed,
        checkpoint_dir=checkpoint_dir, run_label="baseline",
    )
    results[0.0] = (baseline_acc, baseline_std)
    print(f"Baseline cat accuracy: {baseline_acc:.3f}")

    all_synthetic_images = []
    prev_fraction = config.first_fraction
    prev_count = 0
    for target_fraction in config.target_fractions:
        num_to_load = int((target_fraction - prev_fraction) * 5000)

        if not skip_generation:
            print(f"\nGenerating images up to fraction: {target_fraction}")
            new_images = generate_images(pipeline, config.target_class, config.save_dir, target_fraction, prev_fraction)
        else:
            new_files = cached_files[prev_count:prev_count + num_to_load]
            new_images = [Image.open(p) for p in new_files]
            prev_count += num_to_load

        all_synthetic_images.extend(new_images)
        prev_fraction = target_fraction

        synthetic_dataset = SyntheticDataset(all_synthetic_images, config.target_class, transform)
        combined = ConcatDataset([imbalanced_trainset, synthetic_dataset])

        print(f"Training classifier with {len(all_synthetic_images)} synthetic images...")
        acc, std = train_and_evaluate_avg(
            combined, testloader, device, config.target_class,
            num_runs=config.num_runs, lr=config.lr, train_batch_size=config.train_batch_size, seed=config.seed,
            checkpoint_dir=checkpoint_dir, run_label=f"fraction_{target_fraction}",
        )
        results[target_fraction] = (acc, std)
        print(f"Cat accuracy at fraction {target_fraction}: {acc:.3f}")

    return results


def plot_results(results: dict, output_path: str = "results.png") -> None:
    fractions = sorted(k for k in results.keys() if k >= 0)
    means = [results[f][0] for f in fractions]
    stds = [results[f][1] for f in fractions]
    upper_bound_acc = results[-1][0]

    plt.figure(figsize=(8, 5))
    plt.errorbar(fractions, means, yerr=stds, marker='o', capsize=5, capthick=2, linewidth=2)
    plt.axhline(y=upper_bound_acc, color='green', linestyle='--', label=f'Upper bound ({upper_bound_acc:.3f})')
    plt.xlabel("Synthetic data fraction")
    plt.ylabel("Cat class accuracy")
    plt.title("Effect of Diffusion Augmentation on Minority Class Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)


def save_run_artifacts(config: ExperimentConfig, results: dict) -> Path:
    """Persist this run's config and metrics to a timestamped directory under
    config.output_dir, and refresh the top-level results.png used by the README."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    serializable_results = {
        str(fraction): {"mean": mean, "std": std}
        for fraction, (mean, std) in results.items()
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(serializable_results, f, indent=2)

    plot_results(results, str(run_dir / "results.png"))
    plot_results(results, "results.png")

    return run_dir


def main(argv=None) -> None:
    config = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run_experiment(config, device)
    run_dir = save_run_artifacts(config, results)
    print(f"Saved run config, metrics, and plot to {run_dir}")


if __name__ == "__main__":
    main()
