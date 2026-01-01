import os
import json
import logging
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, List


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
                   metrics: Optional[Dict[str, float]] = None,
                   chunk_size: int = 1,
                   action_dim: int = 7) -> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'input_stats': input_stats,
        'output_stats': output_stats,
        'chunk_size': chunk_size,
        'action_dim': action_dim,
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


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor, action_dim: int = 7) -> Dict[str, float]:
    """
    Compute metrics for model outputs.

    Args:
        outputs: Model predictions. Shape: (batch_size, chunk_size * action_dim)
        targets: Ground truth. Shape: (batch_size, chunk_size * action_dim)
        action_dim: Dimension of a single action (default: 7)
    """
    with torch.no_grad():
        # Check if we're using action chunks
        if outputs.shape[1] % action_dim == 0:
            chunk_size = outputs.shape[1] // action_dim
        else:
            chunk_size = 1

        # Reshape to (batch_size, chunk_size, action_dim) for easier indexing
        batch_size = outputs.shape[0]
        outputs_reshaped = outputs.view(batch_size, chunk_size, action_dim)
        targets_reshaped = targets.view(batch_size, chunk_size, action_dim)

        # Extract joint velocities (first 6 dims of each action) and gripper (7th dim)
        joint_vel_outputs = outputs_reshaped[:, :, 0:6]  # (batch_size, chunk_size, 6)
        joint_vel_targets = targets_reshaped[:, :, 0:6]
        joint_vel_mse = torch.mean((joint_vel_outputs - joint_vel_targets) ** 2).item()
        joint_vel_max_error = torch.max(torch.abs(joint_vel_outputs - joint_vel_targets)).item()

        gripper_outputs = outputs_reshaped[:, :, 6]  # (batch_size, chunk_size)
        gripper_targets = targets_reshaped[:, :, 6]

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


def plot_loss_curves(train_losses: list[float],
                     val_losses: list[float],
                     save_path: str,
                     logger: Optional[logging.Logger] = None) -> None:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path where the plot will be saved
        logger: Optional logger for logging messages
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if logger:
            logger.warning("matplotlib not installed. Skipping loss curve plotting.")
        return

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)  # Ensure y=0 is always visible

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"Loss curves saved to {save_path}")


def setup_live_plot():
    """
    Set up a live plot for training and validation losses.

    Returns:
        tuple: (fig, ax, train_line, val_line) for updating the plot
    """
    try:
        import matplotlib.pyplot as plt
        plt.ion()  # Enable interactive mode

        fig, ax = plt.subplots(figsize=(10, 6))
        train_line, = ax.plot([], [], 'b-', label='Training Loss', linewidth=2)
        val_line, = ax.plot([], [], 'r-', label='Validation Loss', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)  # Ensure y=0 is always visible

        return fig, ax, train_line, val_line
    except ImportError:
        return None, None, None, None


def update_live_plot(fig, ax, train_line, val_line, train_losses, val_losses):
    """
    Update the live plot with new loss values.

    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
        train_line: Training loss line object
        val_line: Validation loss line object
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    if fig is None:
        return

    try:
        import matplotlib.pyplot as plt

        epochs = range(1, len(train_losses) + 1)

        train_line.set_data(epochs, train_losses)
        val_line.set_data(epochs, val_losses)

        # Set x-axis limits
        ax.set_xlim(0, max(len(train_losses) + 1, 10))

        # Set y-axis limits to show all data while keeping y=0 visible
        if train_losses and val_losses:
            max_loss = max(max(train_losses), max(val_losses))
            ax.set_ylim(0, max_loss * 1.1)  # Add 10% padding at the top

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
    except Exception:
        pass


def evaluate_specific_episodes(model: nn.Module,
                               episodes: List[np.ndarray],
                               episode_indices: List[int],
                               input_stats: Dict[str, np.ndarray],
                               output_stats: Dict[str, np.ndarray],
                               device: str,
                               chunk_size: int = 1,
                               action_dim: int = 7,
                               batch_size: int = 512,
                               logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Evaluate model on specific episodes.

    Args:
        model: The trained model
        episodes: List of all episodes
        episode_indices: Indices of episodes to evaluate
        input_stats: Normalization stats for inputs
        output_stats: Normalization stats for outputs
        device: Device to run evaluation on
        chunk_size: Action chunk size
        action_dim: Action dimension
        batch_size: Batch size for evaluation
        logger: Optional logger

    Returns:
        Dictionary containing loss and metrics for the specified episodes
    """
    from dataset import RobotTrajectoryDataset

    selected_episodes = [episodes[i] for i in episode_indices]

    if logger:
        logger.info(f"Evaluating {len(selected_episodes)} specific episodes: {episode_indices}")

    eval_dataset = RobotTrajectoryDataset(
        selected_episodes,
        input_stats,
        output_stats,
        chunk_size=chunk_size
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device != 'cpu' else False
    )

    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            all_outputs.append(outputs)
            all_targets.append(targets)

    avg_loss = total_loss / len(eval_loader)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_outputs, all_targets, action_dim=action_dim)

    results = {
        'episode_indices': episode_indices,
        'num_episodes': len(selected_episodes),
        'num_timesteps': len(eval_dataset),
        'loss': avg_loss,
        'metrics': metrics
    }

    if logger:
        logger.info(f"Results for episodes {episode_indices}:")
        logger.info(f"  Loss: {avg_loss:.6f}")
        logger.info(f"  Joint Vel MSE: {metrics['joint_vel_mse']:.6f}")
        logger.info(f"  Joint Vel Max Error: {metrics['joint_vel_max_error']:.6f}")
        logger.info(f"  Gripper Accuracy: {metrics['gripper_accuracy']:.4f}")

    return results
