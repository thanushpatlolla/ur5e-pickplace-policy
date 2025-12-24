import os
import json
import logging
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Any, Optional


def setup_logging(log_dir: str) -> tuple[logging.Logger, str]:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger('robot_training')
    logger.setLevel(logging.INFO)

    logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    log_path = os.path.join(log_dir, f'train_log_{timestamp}.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    metrics_path = os.path.join(log_dir, f'metrics_{timestamp}.json')

    logger.info(f"Logging to {log_path}")
    logger.info(f"Metrics will be saved to {metrics_path}")

    return logger, metrics_path


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   train_loss: float,
                   val_loss: float,
                   input_stats: Dict[str, np.ndarray],
                   output_stats: Dict[str, np.ndarray],
                   checkpoint_path: str,
                   metrics: Optional[Dict[str, float]] = None) -> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'input_stats': input_stats,
        'output_stats': output_stats,
    }

    if metrics is not None:
        checkpoint['metrics'] = metrics

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str,
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'train_loss': checkpoint.get('train_loss', 0.0),
        'val_loss': checkpoint.get('val_loss', 0.0),
        'input_stats': checkpoint.get('input_stats', None),
        'output_stats': checkpoint.get('output_stats', None),
        'metrics': checkpoint.get('metrics', None),
    }


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        joint_vel_outputs = outputs[:, 0:6]
        joint_vel_targets = targets[:, 0:6]
        joint_vel_mse = torch.mean((joint_vel_outputs - joint_vel_targets) ** 2).item()
        joint_vel_max_error = torch.max(torch.abs(joint_vel_outputs - joint_vel_targets)).item()

        gripper_outputs = outputs[:, 6]
        gripper_targets = targets[:, 6]

        gripper_pred = (gripper_outputs > 0.5).float()
        gripper_true = (gripper_targets > 0.5).float()
        gripper_accuracy = (gripper_pred == gripper_true).float().mean().item()

        total_mse = torch.mean((outputs - targets) ** 2).item()

        metrics = {
            'total_mse': total_mse,
            'joint_vel_mse': joint_vel_mse,
            'joint_vel_max_error': joint_vel_max_error,
            'gripper_accuracy': gripper_accuracy,
        }

    return metrics


def log_metrics(logger: logging.Logger,
               metrics_path: str,
               epoch: int,
               train_loss: float,
               val_loss: float,
               metrics: Dict[str, float],
               epoch_time: float) -> None:

    logger.info(f"Epoch {epoch:3d} | "
               f"Train Loss: {train_loss:.6f} | "
               f"Val Loss: {val_loss:.6f} | "
               f"Time: {epoch_time:.2f}s")
    logger.info(f"         | "
               f"Joint Vel MSE: {metrics['joint_vel_mse']:.6f} | "
               f"Gripper Acc: {metrics['gripper_accuracy']:.4f}")

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics[f'epoch_{epoch}'] = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epoch_time': epoch_time,
        **metrics
    }

    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)


def denormalize_outputs(normalized_outputs: torch.Tensor,
                       output_stats: Dict[str, np.ndarray]) -> torch.Tensor:
    mean = torch.tensor(output_stats['mean'], dtype=normalized_outputs.dtype, device=normalized_outputs.device)
    std = torch.tensor(output_stats['std'], dtype=normalized_outputs.dtype, device=normalized_outputs.device)

    denormalized = normalized_outputs * std + mean

    return denormalized


def outputs_to_commands(outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    joint_velocities = outputs[..., 0:6]

    gripper_logits = outputs[..., 6]
    gripper_commands = (gripper_logits > 0.5).float() * 255.0

    return joint_velocities, gripper_commands
