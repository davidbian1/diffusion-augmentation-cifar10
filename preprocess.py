
from sched import scheduler
import torch
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import DDIMScheduler, DDPMPipeline
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import torch.nn.functional as F
import random
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
from generate import load_pipeline, generate_images, create_image_grid
import os
import torchvision.models as models
from torchvision.transforms import v2
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
import sys
import statistics
from PIL import Image
import glob

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

def train_and_evaluate(trainset, testloader, device, target_class, num_epochs=5):
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

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
    return correct / total if total > 0 else 0

def train_and_evaluate_avg(trainset, testloader, device, target_class, num_epochs=5, num_runs=3):
    accuracies = []
    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}...")
        acc = train_and_evaluate(trainset, testloader, device, target_class, num_epochs)
        accuracies.append(acc)
    mean = sum(accuracies) / len(accuracies)
    std = statistics.stdev(accuracies)
    print(f"  Mean: {mean:.3f} ± {std:.3f}")
    return mean, std

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":

    target_class = 3 # cat
    first_fraction = 0.1  # Starting fraction of cat images
    target_fractions = [0.25, 0.5, 0.75, 1, 2]
    batch_size = 4
    repo_id = "Ketansomewhere/cifar10_conditional_diffusion1"
    # transform = v2.Compose([
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),  # add this
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    save_dir = "generated_images"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    imbalanced_trainset = create_imbalanced_dataset(trainset, target_class, first_fraction)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)
    

    cache_path = os.path.join(save_dir, "*.png")
    cached_files = glob.glob(cache_path)

    results = {}
    all_synthetic_images = []
    prev_fraction = first_fraction

    all_cached_files = sorted(glob.glob(os.path.join(save_dir, "*.png")))

    if cached_files:
        print(f"Loading {len(cached_files)} cached images from disk...")
        skip_generation = True
    else:
        pipeline = load_pipeline(repo_id, device)
        skip_generation = False

    all_cached_files = sorted(glob.glob(os.path.join(save_dir, "*.png")))

    print("Training upper bound (full balanced data)...")
    upper_bound_acc, upper_bound_std = train_and_evaluate_avg(trainset, testloader, device, target_class, 5)
    results[-1] = (upper_bound_acc, upper_bound_std)
    print(f"Upper bound cat accuracy (full data): {upper_bound_acc:.3f}")

    # sys.exit()

    print("Training baseline (no synthetic data)...")
    baseline_acc, baseline_std = train_and_evaluate_avg(imbalanced_trainset, testloader, device, target_class, 5)
    results[0.0] = (baseline_acc, baseline_std)
    print(f"Baseline cat accuracy: {baseline_acc:.3f}")

    prev_count = 0
    for target_fraction in target_fractions:
        num_to_load = int((target_fraction - prev_fraction) * 5000)

        if not skip_generation:
            print(f"\nGenerating images up to fraction: {target_fraction}")
            new_images = generate_images(pipeline, target_class, save_dir, target_fraction, prev_fraction)
            all_synthetic_images.extend(new_images)
            prev_fraction = target_fraction
        else:
            new_files = all_cached_files[prev_count:prev_count + num_to_load]
            new_images = [Image.open(p) for p in new_files]
            all_synthetic_images.extend(new_images)
            prev_count += num_to_load
            prev_fraction = target_fraction  # update prev_fraction here too

        synthetic_dataset = SyntheticDataset(all_synthetic_images, target_class, transform)
        combined = ConcatDataset([imbalanced_trainset, synthetic_dataset])

        print(f"Training classifier with {len(all_synthetic_images)} synthetic images...")
        acc, std = train_and_evaluate_avg(combined, testloader, device, target_class)
        results[target_fraction] = (acc, std)
        print(f"Cat accuracy at fraction {target_fraction}: {acc:.3f}")

    # Plot results
# Plot results with error bars
# Plot results with error bars
    fractions = sorted([k for k in results.keys() if k >= 0])
    means = [results[f][0] for f in fractions]
    stds = [results[f][1] for f in fractions]

    plt.figure(figsize=(8, 5))
    plt.errorbar(fractions, means, yerr=stds, marker='o', capsize=5, capthick=2, linewidth=2)
    plt.axhline(y=upper_bound_acc, color='green', linestyle='--', label=f'Upper bound ({upper_bound_acc:.3f})')
    plt.xlabel("Synthetic data fraction")
    plt.ylabel("Cat class accuracy")
    plt.title("Effect of Diffusion Augmentation on Minority Class Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("results.png")
    plt.show()

# # get some random training images
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# model_id = "google/ddpm-cifar10-32"

# ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
# ddpm.to(device)

# scheduler = DDIMScheduler.from_pretrained(model_id)
# scheduler.set_timesteps(num_inference_steps=40)

# image = ddpm().images[0]
# image.save("ddpm_generated_image.png")

# x = torch.randn(4, 3, 32, 32).to(device)  

# for i, t in tqdm(enumerate(scheduler.timesteps)):

#     model_input = scheduler.scale_model_input(x, t)
#     with torch.no_grad():
#         noise_pred = ddpm.unet(model_input, t)["sample"]

#     scheduler_output = scheduler.step(noise_pred, t, x)
#     x = scheduler_output.prev_sample

    # if i % 10 == 0 or i == len(scheduler.timesteps) - 1:
    #     fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    #     grid = torchvision.utils.make_grid(x, nrow=4).permute(1, 2, 0)
    #     axs[0].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
    #     axs[0].set_title(f"Current x (step {i})")

    #     pred_x0 = (
    #         scheduler_output.pred_original_sample
    #     )  
    #     grid = torchvision.utils.make_grid(pred_x0, nrow=4).permute(1, 2, 0)
    #     axs[1].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
    #     axs[1].set_title(f"Predicted denoised images (step {i})")
    #     plt.show()





