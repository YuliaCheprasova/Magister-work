 
"""
Evaluator for learning rate scheduler search.
Tests the scheduler with different parameter configurations to find robust solutions.
"""

import os
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch
import torchvision
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import sys
import time
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import math


from shinka.core import run_shinka_eval
from models.av_MNIST import av_Classifier

NUM_EPOCHES = 50
def get_experiment_kwargs(run_idx: int) -> Dict[str, Any]:
    """
    Returns different parameter configurations for each run.
    This tests the scheduler's robustness across different settings.

    Args:
        run_idx: Index of the run (0-based)

    Returns:
        Dictionary of parameters for compute_lr
    """

    return {"total_steps": NUM_EPOCHES,}


def validate_fn(lr_schedule: List[float]) -> Tuple[bool, Optional[str]]:
    """
    Validates that the LR schedule is well-formed.
    Checks that all LR values are positive and finite.
    """
    if not lr_schedule:
        return False, "Empty schedule"
    if len(lr_schedule) != NUM_EPOCHES:
        return False, f"Schedule length {len(lr_schedule)} != expected {NUM_EPOCHES}"

    for i, lr in enumerate(lr_schedule):
        if not np.isfinite(lr):
            return False, f"Step {i}: non-finite LR {lr}"
        if lr <= 0:
            return False, f"Step {i}: non-positive LR {lr}"

    return True, "Valid schedule"


def train_with_scheduler(lr_schedule: List[float]):
    """
    Trains MLP on MNIST using the given LR schedule.

    Args:
        lr_schedule: List of learning rates for each step

    Returns:
        Dictionary with training metrics
    """

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device('cuda')
    torch.set_float32_matmul_precision("medium")# снижение точности вычислений
    torch.backends.cudnn.benchmark = True
    print('start evaluation')
    batch_size = 128
    num_workers = 0
    generator = torch.Generator(device=device)

    transform = transforms.Compose(
        [transforms.ToTensor(), ])  # transforms.ToTensor() автоматически нормализует данные в случае картинок

    # Use small subset for fast evaluation
    train_dataset = MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    # Use only 5000 samples for fast training
    train_dataset.data = train_dataset.data[:5000]
    train_dataset.targets = train_dataset.targets[:5000]

    test_dataset = MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    # Use only 1000 samples for validation
    test_dataset.data = test_dataset.data[:1000]
    test_dataset.targets = test_dataset.targets[:1000]

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, generator=generator, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, generator=generator, num_workers=num_workers)

    time_individ = time.time()

    # Create model and optimizer
    model = av_Classifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_schedule[0])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Training loop
    model.train()

    for epoch in range(1, NUM_EPOCHES + 1):
        time_epoch = time.time()
        lr = lr_schedule[epoch-1]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            torch.cuda.synchronize()
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        running_loss /= len(train_loader)
        print('Epoch [{}/{}], lr:{:.4f}, Loss:{:.4f}, Time_for_epoch: {:.4f}'.format(
            epoch, NUM_EPOCHES, lr, running_loss, time.time() - time_epoch))

    # Validation
    model.eval()
    test_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # получаем индекс
            correct += torch.sum(predicted == labels).item()

    test_loss /= len(test_loader)  # делим на количество батчей
    test_acc = correct / len(test_dataset)  # делим на количество наблюдений в тестовой выборке

    print(
                    'TestLoss: {:.4f}, TestAccuracy: {:.4f}, Time_for_individ: {:.4f}'.format(test_loss,
                                                                                                       test_acc, time.time() - time_individ))

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def aggregate_fn(results: List[List[float]]) -> Dict[str, Any]:
    """
    Aggregates metrics by training the network with the scheduler.
    Score is the validation accuracy (higher = better).
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results"}

    lr_schedule = results[0]

    # Train the network
    metrics = train_with_scheduler(lr_schedule)

    test_loss = metrics["test_loss"]

    combined_score = 1.0 / (1.0 + test_loss)

    # Analyze LR schedule properties
    lr_values = np.array(lr_schedule)
    max_lr = float(np.max(lr_values))
    min_lr = float(np.min(lr_values))
    lr_ratio = max_lr / min_lr if min_lr > 1e-10 else float('inf')
    smoothness = float(np.sum(np.abs(np.diff(lr_values))))

    return {
        "combined_score": round(combined_score, 4),
        "public": {
            "test_acc": round(metrics["test_acc"], 4),
            "test_loss": round(test_loss, 6),
        },
        "private": {
            "lr_ratio": round(min(lr_ratio, 1e6), 2),
            "smoothness": round(smoothness, 4), #
        },
        "extra_data": {
            "lr_schedule": lr_schedule,
        },

    }


def main(program_path: str, results_dir: str):
    """Runs the learning rate scheduler evaluation."""
    print(f"Evaluating scheduler: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    metrics, correct, err = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=1,  # Single run (no need for multiple configs)
        run_workers=1,
        get_experiment_kwargs=get_experiment_kwargs,
        aggregate_metrics_fn=aggregate_fn,
        validate_fn=validate_fn,
    )

    if correct:
        print("✓ Scheduler is valid")
    else:
        print(f"✗ Scheduler validation failed: {err}")

    print(f"\nCombined Score (Val Accuracy): {metrics.get('combined_score', 'N/A')}")
    if "public" in metrics:
        print("\nPublic Metrics:")
        for k, v in metrics["public"].items():
            print(f"  {k}: {v}")

    if "text_feedback" in metrics:
        print(f"\nFeedback for LLM: {metrics['text_feedback']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learning rate scheduler evaluator"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        required=True,
        help="Path to scheduler program",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save results",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
