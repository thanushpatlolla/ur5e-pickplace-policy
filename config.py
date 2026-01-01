from dataclasses import dataclass
import torch


@dataclass
class TrainingConfig:
    """Configuration for training the robot control MLP."""

    data_path: str = "data/pick_place_dataset_20251230_004112.npz"

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Model architecture
    input_size: int = 30          # Joint pos(6) + vel(6) + EE pos(3) + EE quat(4) + obj pos(3) + obj quat(4) + obj size(3) + gripper(1)
    hidden_size: int = 256
    num_hidden_layers: int = 3
    chunk_size: int = 10          # Number of future actions to predict
    action_dim: int = 7           # Joint vel commands(6) + gripper(1)

    @property
    def output_size(self) -> int:
        return self.action_dim * self.chunk_size

    batch_size: int = 512         
    learning_rate: float = 3e-4   #AdamW
    weight_decay: float = 1e-4    
    epochs: int = 100             

    # Early stopping
    patience: int = 10           

    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    log_interval: int = 10        
    save_interval: int = 5        

    num_workers: int = 4          # Number of workers for data loading
    pin_memory: bool = True       # Pin memory for faster GPU transfer
