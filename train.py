import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from datetime import datetime

from model import MLP
from dataset import load_dataset, split_episodes, compute_normalization_stats, RobotTrajectoryDataset
from config import TrainingConfig
from utils import setup_logging, save_checkpoint, load_checkpoint, compute_metrics, log_metrics, plot_loss_curves, setup_live_plot, update_live_plot, evaluate_specific_episodes


class CompositeLoss(nn.Module):
    def __init__(self, action_dim: int, chunk_size: int, gripper_weight: float = 1.0):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.gripper_weight = gripper_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss, _, _ = self._compute_losses(predictions, targets)
        return total_loss

    def _compute_losses(self, predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = predictions.shape[0]

        # Reshape to (batch_size, chunk_size, action_dim)
        pred_reshaped = predictions.view(batch_size, self.chunk_size, self.action_dim)
        target_reshaped = targets.view(batch_size, self.chunk_size, self.action_dim)

        # Split into joint velocities and gripper
        pred_joint_vels = pred_reshaped[:, :, :6]  # (batch, chunk_size, 6)
        pred_gripper = pred_reshaped[:, :, 6:]      # (batch, chunk_size, 1)

        target_joint_vels = target_reshaped[:, :, :6]
        target_gripper = target_reshaped[:, :, 6:]

        # Compute MSE loss for joint velocities
        joint_vel_loss = self.mse_loss(pred_joint_vels, target_joint_vels)

        # Compute BCE loss for gripper (binary classification)
        gripper_loss = self.bce_loss(pred_gripper, target_gripper)

        # Combine losses with weighting
        total_loss = joint_vel_loss + self.gripper_weight * gripper_loss

        return total_loss, joint_vel_loss, gripper_loss

    def compute_with_components(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        total_loss, joint_vel_loss, gripper_loss = self._compute_losses(predictions, targets)
        return {
            'total': total_loss,
            'joint_vel_mse': joint_vel_loss,
            'gripper_bce': gripper_loss
        }


def train_one_epoch(model: nn.Module,
                   train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module,
                   device: str) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    joint_vel_loss_sum = 0.0
    gripper_loss_sum = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute loss with components for logging
        if isinstance(loss_fn, CompositeLoss):
            loss_components = loss_fn.compute_with_components(outputs, targets)
            loss = loss_components['total']
            joint_vel_loss_sum += loss_components['joint_vel_mse'].item()
            gripper_loss_sum += loss_components['gripper_bce'].item()
        else:
            loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    num_batches = len(train_loader)
    return {
        'total': total_loss / num_batches,
        'joint_vel_mse': joint_vel_loss_sum / num_batches,
        'gripper_bce': gripper_loss_sum / num_batches
    }


def validate(model: nn.Module,
            val_loader: DataLoader,
            loss_fn: nn.Module,
            device: str,
            action_dim: int = 7) -> tuple[dict[str, float], dict]:
    model.eval()
    total_loss = 0.0
    joint_vel_loss_sum = 0.0
    gripper_loss_sum = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Compute loss with components for logging
            if isinstance(loss_fn, CompositeLoss):
                loss_components = loss_fn.compute_with_components(outputs, targets)
                loss = loss_components['total']
                joint_vel_loss_sum += loss_components['joint_vel_mse'].item()
                gripper_loss_sum += loss_components['gripper_bce'].item()
            else:
                loss = loss_fn(outputs, targets)

            total_loss += loss.item()

            all_outputs.append(outputs)
            all_targets.append(targets)

    num_batches = len(val_loader)
    loss_dict = {
        'total': total_loss / num_batches,
        'joint_vel_mse': joint_vel_loss_sum / num_batches,
        'gripper_bce': gripper_loss_sum / num_batches
    }

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_outputs, all_targets, action_dim=action_dim)

    return loss_dict, metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--chunk_size', type=int, help='Action chunk size')
    parser.add_argument('--lr_scheduler_factor', type=float, help='LR scheduler factor')
    parser.add_argument('--lr_scheduler_patience', type=int, help='LR scheduler patience')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--no_lr_scheduler', action='store_true', help='Disable learning rate scheduler')
    args = parser.parse_args()

    config = TrainingConfig()

    # Override config with command-line arguments
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size
    if args.lr_scheduler_factor is not None:
        config.lr_scheduler_factor = args.lr_scheduler_factor
    if args.lr_scheduler_patience is not None:
        config.lr_scheduler_patience = args.lr_scheduler_patience
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.no_lr_scheduler:
        config.use_lr_scheduler = False

    logger, metrics_path = setup_logging(config.log_dir)

    logger.info("=" * 80)
    logger.info("Robot Control MLP Training")
    logger.info("=" * 80)
    logger.info(f"Device: {config.device}")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Weight decay: {config.weight_decay}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Patience: {config.patience}")
    if config.use_lr_scheduler:
        logger.info(f"LR scheduler: ReduceLROnPlateau (factor={config.lr_scheduler_factor}, patience={config.lr_scheduler_patience}, threshold={config.lr_scheduler_threshold}, cooldown={config.lr_scheduler_cooldown}, min_lr={config.lr_scheduler_min_lr})")
    else:
        logger.info("LR scheduler: Disabled")
    logger.info(f"Gripper loss weight: {config.gripper_loss_weight}")
    logger.info(f"Action chunking: chunk_size={config.chunk_size}, action_dim={config.action_dim}")
    logger.info(f"Model: {config.input_size}D -> {config.hidden_size}H x {config.num_hidden_layers} -> {config.output_size}D")
    logger.info("=" * 80)

    logger.info("\nLoading dataset...")
    episodes = load_dataset(config.data_path)
    train_episodes, val_episodes, test_episodes = split_episodes(
        episodes, [config.train_ratio, config.val_ratio, config.test_ratio]
    )

    logger.info("\nComputing normalization statistics...")
    input_stats, output_stats = compute_normalization_stats(
        train_episodes,
        chunk_size=config.chunk_size,
        action_dim=config.action_dim
    )

    logger.info("\nCreating datasets...")
    train_dataset = RobotTrajectoryDataset(train_episodes, input_stats, output_stats, chunk_size=config.chunk_size)
    val_dataset = RobotTrajectoryDataset(val_episodes, input_stats, output_stats, chunk_size=config.chunk_size)
    test_dataset = RobotTrajectoryDataset(test_episodes, input_stats, output_stats, chunk_size=config.chunk_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    logger.info("\nInitializing model...")
    model = MLP(
        config.input_size,
        config.hidden_size,
        config.num_hidden_layers,
        config.output_size,
        action_dim=config.action_dim,
        max_joint_velocity=config.max_joint_velocity
    )
    model = model.to(config.device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
            threshold=config.lr_scheduler_threshold,
            cooldown=config.lr_scheduler_cooldown,
            min_lr=config.lr_scheduler_min_lr
        )

    loss_fn = CompositeLoss(
        action_dim=config.action_dim,
        chunk_size=config.chunk_size,
        gripper_weight=config.gripper_loss_weight
    ).to(config.device)

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []  # Will store total losses for plotting
    val_losses = []    # Will store total losses for plotting

    loss_plot_path = os.path.join(config.log_dir, f'loss_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')

    fig, ax, train_line, val_line = setup_live_plot()
    if fig is not None:
        logger.info("Live plotting enabled - loss curves will update in real-time")

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()

        logger.info(f"\nEpoch {epoch}/{config.epochs}")
        train_loss_dict = train_one_epoch(
            model, train_loader, optimizer, loss_fn, config.device
        )

        val_loss_dict, metrics = validate(model, val_loader, loss_fn, config.device, config.action_dim)

        epoch_time = time.time() - epoch_start_time

        # Store total losses for plotting
        train_losses.append(train_loss_dict['total'])
        val_losses.append(val_loss_dict['total'])

        log_metrics(logger, metrics_path, epoch, train_loss_dict, val_loss_dict, metrics, epoch_time)

        update_live_plot(fig, ax, train_line, val_line, train_losses, val_losses)

        # Update learning rate based on validation loss
        if scheduler is not None:
            scheduler.step(val_loss_dict['total'])

        # Use total loss for model selection
        if val_loss_dict['total'] < best_val_loss:
            best_val_loss = val_loss_dict['total']
            epochs_without_improvement = 0
            best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
            save_checkpoint(
                model, optimizer, epoch, train_loss_dict['total'], val_loss_dict['total'],
                input_stats, output_stats, best_model_path, metrics,
                chunk_size=config.chunk_size, action_dim=config.action_dim
            )
            logger.info(f"  -> Best model saved (val_loss: {val_loss_dict['total']:.6f})")
        else:
            epochs_without_improvement += 1

        if epoch % config.save_interval == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(
                model, optimizer, epoch, train_loss_dict['total'], val_loss_dict['total'],
                input_stats, output_stats, checkpoint_path, metrics,
                chunk_size=config.chunk_size, action_dim=config.action_dim
            )

        latest_path = os.path.join(config.checkpoint_dir, 'latest_model.pth')
        save_checkpoint(
            model, optimizer, epoch, train_loss_dict['total'], val_loss_dict['total'],
            input_stats, output_stats, latest_path, metrics,
            chunk_size=config.chunk_size, action_dim=config.action_dim
        )

        if epochs_without_improvement >= config.patience:
            logger.info(f"\nEarly stopping triggered after {epoch} epochs")
            logger.info(f"No improvement for {config.patience} epochs")
            break

    logger.info("\n" + "=" * 80)
    logger.info("Training completed. Evaluating on test set...")
    logger.info("=" * 80)

    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    checkpoint_data = load_checkpoint(best_model_path, model)
    logger.info(f"Loaded best model from epoch {checkpoint_data['epoch']}")

    test_loss_dict, test_metrics = validate(model, test_loader, loss_fn, config.device, config.action_dim)

    logger.info("\nTest Set Results:")
    logger.info(f"  Test Loss (Total): {test_loss_dict['total']:.6f}")
    logger.info(f"  Test Loss (Joint Vel MSE): {test_loss_dict['joint_vel_mse']:.6f}")
    logger.info(f"  Test Loss (Gripper BCE): {test_loss_dict['gripper_bce']:.6f}")
    logger.info(f"  Joint Vel MSE: {test_metrics['joint_vel_mse']:.6f}")
    logger.info(f"  Joint Vel Max Error: {test_metrics['joint_vel_max_error']:.6f}")
    logger.info(f"  Gripper Accuracy: {test_metrics['gripper_accuracy']:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("Saving final loss curve...")
    logger.info("=" * 80)
    plot_loss_curves(train_losses, val_losses, loss_plot_path, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Best model saved at: {best_model_path}")
    logger.info(f"Metrics saved at: {metrics_path}")
    logger.info(f"Loss curves saved at: {loss_plot_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
