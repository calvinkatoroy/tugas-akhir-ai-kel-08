import random
import csv
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def log_experiment(record: dict, csv_path='results/metrics_summary.csv'):
    """Append one experiment record (dict) to the metrics CSV."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def plot_confusion_matrix(cm, class_names, title, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_loss_curves(train_losses, val_losses, title, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label='Train loss')
    ax.plot(val_losses, label='Val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_overlay(roc_data: dict, save_path=None):
    """roc_data: {'ModelName': (fpr_array, tpr_array, auc_score), ...}"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
