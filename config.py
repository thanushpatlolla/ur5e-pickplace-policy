from dataclasses import dataclass, field
from datetime import datetime
import torch


@dataclass
class DataCollectionConfig:
    target_episodes: int = 2000
    max_attempts: int = 4000  # Safety limit for failed runs
    noise_std: float = 0.002    # Gaussian noise std dev in meters (0.002 = 2mm)


@dataclass
class TrainingConfig:
    data_path: str = "data"  # Will auto-detect latest dataset in this directory

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Model architecture
    input_size: int = 31          # Joint pos(6) + vel(6) + EE pos(3) + EE quat(4) + obj pos(3) + obj quat(4) + obj size(3) + gripper_joint_pos(1) + gripper_joint_vel(1)
    hidden_size: int = 256
    num_hidden_layers: int = 3
    chunk_size: int = 10          # Number of future actions to predict
    action_dim: int = 7           # Joint vel commands(6) + gripper(1)

    # Velocity limits (rad/s) - must match the IK solver limits
    max_joint_velocity: float = 3.14159265359 / 3  # π/3 rad/s for all joints

    @property
    def output_size(self) -> int:
        return self.action_dim * self.chunk_size

    batch_size: int = 256
    learning_rate: float = 3e-4   #AdamW
    weight_decay: float = 1e-4
    epochs: int = 100

    # Learning rate scheduler (ReduceLROnPlateau)
    use_lr_scheduler: bool = False          # Whether to use learning rate scheduler
    lr_scheduler_factor: float = 0.5       # Factor by which LR is reduced
    lr_scheduler_patience: int = 5         # Epochs with no improvement before reducing LR
    lr_scheduler_threshold: float = 2e-3   # Threshold for measuring improvement
    lr_scheduler_cooldown: int = 1         # Epochs to wait before resuming normal operation
    lr_scheduler_min_lr: float = 1e-6      # Minimum learning rate

    # Loss weighting
    gripper_loss_weight: float = 10.0  # Weight for BCE loss on gripper (vs MSE for joint vels)             

    # Early stopping
    patience: int = 10        

    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Run identification - generates timestamp for each training run
    run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    @property
    def checkpoint_dir(self) -> str:
        return f"checkpoints/{self.run_name}"

    @property
    def log_dir(self) -> str:
        return f"logs/{self.run_name}"

    log_interval: int = 10        
    save_interval: int = 5        

    num_workers: int = 4          # Number of workers for data loading
    pin_memory: bool = True       # Pin memory for faster GPU transfer
