import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from model import MLP
from dataset import load_dataset, split_episodes, compute_normalization_stats, RobotTrajectoryDataset
from config import TrainingConfig
from utils import setup_logging, save_checkpoint, load_checkpoint, compute_metrics, log_metrics


def train_one_epoch(model: nn.Module,
                   train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module,
                   device: str) -> float:
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model: nn.Module,
            val_loader: DataLoader,
            loss_fn: nn.Module,
            device: str) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            all_outputs.append(outputs)
            all_targets.append(targets)

    val_loss = total_loss / len(val_loader)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_outputs, all_targets)

    return val_loss, metrics


def main(args):
    config = TrainingConfig()

    if args.data_path:
        config.data_path = args.data_path
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.epochs:
        config.epochs = args.epochs

    logger, metrics_path = setup_logging(config.log_dir)

    logger.info("=" * 80)
    logger.info("Robot Control MLP Training")
    logger.info("=" * 80)
    logger.info(f"Device: {config.device}")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Model: {config.input_size}D -> {config.hidden_size}H x {config.num_hidden_layers} -> {config.output_size}D")
    logger.info("=" * 80)

    logger.info("\nLoading dataset...")
    episodes = load_dataset(config.data_path)
    train_episodes, val_episodes, test_episodes = split_episodes(
        episodes, [config.train_ratio, config.val_ratio, config.test_ratio]
    )

    logger.info("\nComputing normalization statistics...")
    input_stats, output_stats = compute_normalization_stats(train_episodes)

    logger.info("\nCreating datasets...")
    train_dataset = RobotTrajectoryDataset(train_episodes, input_stats, output_stats)
    val_dataset = RobotTrajectoryDataset(val_episodes, input_stats, output_stats)
    test_dataset = RobotTrajectoryDataset(test_episodes, input_stats, output_stats)

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
        config.output_size
    )
    model = model.to(config.device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()

        logger.info(f"\nEpoch {epoch}/{config.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, config.device
        )

        val_loss, metrics = validate(model, val_loader, loss_fn, config.device)

        epoch_time = time.time() - epoch_start_time

        log_metrics(logger, metrics_path, epoch, train_loss, val_loss, metrics, epoch_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                input_stats, output_stats, best_model_path, metrics
            )
            logger.info(f"  -> Best model saved (val_loss: {val_loss:.6f})")
        else:
            epochs_without_improvement += 1

        if epoch % config.save_interval == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                input_stats, output_stats, checkpoint_path, metrics
            )

        latest_path = os.path.join(config.checkpoint_dir, 'latest_model.pth')
        save_checkpoint(
            model, optimizer, epoch, train_loss, val_loss,
            input_stats, output_stats, latest_path, metrics
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

    test_loss, test_metrics = validate(model, test_loader, loss_fn, config.device)

    logger.info("\nTest Set Results:")
    logger.info(f"  Test Loss: {test_loss:.6f}")
    logger.info(f"  Joint Vel MSE: {test_metrics['joint_vel_mse']:.6f}")
    logger.info(f"  Joint Vel Max Error: {test_metrics['joint_vel_max_error']:.6f}")
    logger.info(f"  Gripper Accuracy: {test_metrics['gripper_accuracy']:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Best model saved at: {best_model_path}")
    logger.info(f"Metrics saved at: {metrics_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train robot control MLP')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to NPZ dataset file')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train')

    args = parser.parse_args()
    main(args)
