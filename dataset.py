import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from pathlib import Path
import glob


def find_latest_dataset(data_dir: str, pattern: str = "pick_place_dataset_*.npz") -> str:
    """Find the latest dataset file in the given directory based on filename timestamp."""
    search_path = Path(data_dir) / pattern
    dataset_files = glob.glob(str(search_path))

    if not dataset_files:
        raise FileNotFoundError(f"No dataset files matching pattern '{pattern}' found in {data_dir}")

    # Sort by filename (which includes timestamp) to get the latest
    latest_file = sorted(dataset_files)[-1]
    return latest_file


def load_dataset(npz_path: str) -> List[np.ndarray]:
    """Load dataset from a .npz file or find the latest dataset in a directory."""
    path = Path(npz_path)

    # If path is a directory, find the latest dataset file
    if path.is_dir():
        npz_path = find_latest_dataset(str(path))
        print(f"Auto-detected latest dataset: {npz_path}")
    elif not path.exists():
        # If the path doesn't exist, try treating it as a directory
        parent_dir = path.parent
        if parent_dir.exists() and parent_dir.is_dir():
            npz_path = find_latest_dataset(str(parent_dir))
            print(f"Auto-detected latest dataset: {npz_path}")

    data = np.load(npz_path)
    episodes = []

    episode_idx = 0
    while f'episode_{episode_idx}' in data:
        episodes.append(data[f'episode_{episode_idx}'])
        episode_idx += 1

    print(f"Loaded {len(episodes)} episodes from {npz_path}")

    return episodes


def split_episodes(episodes: List[np.ndarray],
                   ratios: List[float] = [0.8, 0.1, 0.1]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    num_episodes = len(episodes)
    train_end = int(num_episodes * ratios[0])
    val_end = train_end + int(num_episodes * ratios[1])

    train_episodes = episodes[:train_end]
    val_episodes = episodes[train_end:val_end]
    test_episodes = episodes[val_end:]

    print(f"Split: {len(train_episodes)} train, {len(val_episodes)} val, {len(test_episodes)} test episodes")

    return train_episodes, val_episodes, test_episodes


def extract_inputs(episode_data: np.ndarray, chunk_size: int = 1) -> np.ndarray:
    inputs = np.concatenate([
        episode_data[:, 0:6],    # Joint positions (6)
        episode_data[:, 6:12],   # Joint velocities (6)
        episode_data[:, 18:21],  # End-effector position (3)
        episode_data[:, 21:25],  # End-effector orientation quaternion (4)
        episode_data[:, 25:28],  # Object position (3)
        episode_data[:, 28:32],  # Object quaternion (4)
        episode_data[:, 32:35],  # Object size (3)
        episode_data[:, 35:36],  # Gripper driver joint position (1)
        episode_data[:, 36:37],  # Gripper driver joint velocity (1)
    ], axis=1)

    # If using action chunks, only return inputs for valid timesteps
    if chunk_size > 1:
        num_valid_timesteps = len(inputs) - chunk_size + 1
        inputs = inputs[:num_valid_timesteps]

    return inputs


def extract_outputs(episode_data: np.ndarray, chunk_size: int = 1) -> np.ndarray:
    joint_vels = episode_data[:, 12:18]
    gripper = episode_data[:, 37:38]
    gripper = gripper / 255.0

    single_step_outputs = np.concatenate([joint_vels, gripper], axis=1)  # (T, 7)

    if chunk_size == 1:
        return single_step_outputs

    # Create action chunks: for each timestep, get next chunk_size actions
    num_valid_timesteps = len(single_step_outputs) - chunk_size + 1
    chunked_outputs = []

    for t in range(num_valid_timesteps):
        chunk = single_step_outputs[t:t+chunk_size].flatten()  # (chunk_size*7,)
        chunked_outputs.append(chunk)

    return np.array(chunked_outputs)  # (T-chunk_size+1, 7*chunk_size)


def compute_normalization_stats(train_episodes: List[np.ndarray],
                                chunk_size: int = 1,
                                action_dim: int = 7) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    all_inputs = []
    all_outputs = []

    for episode in train_episodes:
        inputs = extract_inputs(episode, chunk_size)
        outputs = extract_outputs(episode, chunk_size)
        all_inputs.append(inputs)
        all_outputs.append(outputs)

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    #normalize
    input_mean = all_inputs.mean(axis=0)
    input_std = all_inputs.std(axis=0)
    output_mean = all_outputs.mean(axis=0)
    output_std = all_outputs.std(axis=0)

    input_std[input_std < 1e-6] = 1.0
    output_std[output_std < 1e-6] = 1.0

    # For action chunks, set gripper dimension stats (every action_dim-th element) to 0 mean, 1 std
    # Gripper is the last dimension of each action (index 6, 13, 20, ...)
    for i in range(chunk_size):
        gripper_idx = i * action_dim + (action_dim - 1)
        output_mean[gripper_idx] = 0.0
        output_std[gripper_idx] = 1.0

    input_stats = {'mean': input_mean, 'std': input_std}
    output_stats = {'mean': output_mean, 'std': output_std}

    print(f"Computed normalization stats from {len(train_episodes)} training episodes")
    print(f"Total training timesteps: {len(all_inputs)}")
    print(f"Chunk size: {chunk_size}, Action dim: {action_dim}, Output dim: {len(output_mean)}")

    return input_stats, output_stats


class RobotTrajectoryDataset(Dataset):
    def __init__(self, episodes: List[np.ndarray],
                 input_stats: Dict[str, np.ndarray],
                 output_stats: Dict[str, np.ndarray],
                 chunk_size: int = 1):
        self.input_stats = input_stats
        self.output_stats = output_stats
        self.chunk_size = chunk_size

        all_inputs = []
        all_outputs = []

        for episode in episodes:
            inputs = extract_inputs(episode, chunk_size)
            outputs = extract_outputs(episode, chunk_size)
            all_inputs.append(inputs)
            all_outputs.append(outputs)

        # Concatenate all timesteps
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)

        normalized_inputs = (all_inputs - input_stats['mean']) / input_stats['std']
        normalized_outputs = (all_outputs - output_stats['mean']) / output_stats['std']

        self.inputs = torch.tensor(normalized_inputs, dtype=torch.float32)
        self.outputs = torch.tensor(normalized_outputs, dtype=torch.float32)

        print(f"Created dataset with {len(self.inputs)} timesteps (chunk_size={chunk_size})")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx]
