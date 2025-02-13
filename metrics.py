import torch
import numpy as np
from typing import Dict, List, Tuple


def calculate_per_task_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_tasks: int = 5,
    classes_per_task: int = 2,
    current_task: int = 0,
) -> Dict[str, float]:
    """
    Calculate accuracy for each previously seen task in class incremental continual learning.

    Args:
        predictions: Model predictions tensor (N, num_classes)
        labels: Ground truth labels tensor (N)
        num_tasks: Total number of tasks
        classes_per_task: Number of classes per task
        current_task: Current task index (0-based)

    Returns:
        Dictionary containing per-task accuracies and average accuracy
    """
    # Convert predictions to class indices if they're not already
    if predictions.dim() > 1:
        predictions = torch.argmax(predictions, dim=1)

    # Initialize dictionary to store accuracies
    per_task_accuracy = {}
    total_per_task_accuracy = 0
    total_tasks = 0

    # Calculate accuracy for each previously seen task (including current task)
    for task_id in range(current_task + 1):
        # Calculate class range for this task
        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task

        # Create mask for samples belonging to current task's classes
        task_mask = (labels >= start_class) & (labels < end_class)

        if task_mask.sum() == 0:
            continue

        # Get predictions and labels for current task's classes
        task_preds = predictions[task_mask]
        task_labels = labels[task_mask]

        # Calculate accuracy for current task
        correct = (task_preds == task_labels).sum().item()
        total = task_mask.sum().item()
        task_accuracy = (correct / total) * 100 if total > 0 else 0

        # Store task accuracy
        per_task_accuracy[f"valid/task_{task_id}_accuracy"] = task_accuracy

        # Update counts for average accuracy
        total_per_task_accuracy += task_accuracy
        total_tasks += 1

    # Calculate average accuracy across all seen tasks
    if total_per_task_accuracy > 0:
        per_task_accuracy["valid/average_accuracy"] = total_per_task_accuracy / total_tasks
    else:
        per_task_accuracy["valid/average_accuracy"] = 0.0

    return per_task_accuracy


def get_task_samples(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    task_id: int,
    classes_per_task: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter predictions and labels for a specific task.

    Args:
        predictions: Model predictions tensor
        labels: Ground truth labels tensor
        task_id: Task ID to filter for
        classes_per_task: Number of classes per task

    Returns:
        Tuple of filtered predictions and labels for the specified task
    """
    start_class = task_id * classes_per_task
    end_class = start_class + classes_per_task

    # Create mask for samples belonging to task's classes
    task_mask = (labels >= start_class) & (labels < end_class)

    # Filter predictions and labels
    task_preds = predictions[task_mask]
    task_labels = labels[task_mask]

    return task_preds, task_labels


# Example usage:
def example_usage():
    # Create sample predictions and labels
    num_samples = 100
    num_classes = 10
    current_task = 3  # Third task (0-based indexing)

    # Generate random predictions and labels
    predictions = torch.randint(0, num_classes, (num_samples,))
    labels = torch.randint(0, num_classes, (num_samples,))

    # Calculate per-task accuracy
    accuracies = calculate_per_task_accuracy(
        predictions=predictions,
        labels=labels,
        num_tasks=5,
        classes_per_task=2,
        current_task=current_task,
    )

    print("Per-task accuracies:", accuracies)

    # Get samples for a specific task
    task_preds, task_labels = get_task_samples(
        predictions=predictions, labels=labels, task_id=1, classes_per_task=2
    )

    print(f"Number of samples for task 1: {len(task_labels)}")
