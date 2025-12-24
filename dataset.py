import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict


def load_dataset(npz_path: str) -> List[np.ndarray]:
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


def extract_inputs(episode_data: np.ndarray) -> np.ndarray:
    inputs = np.concatenate([
        episode_data[:, 0:6],    # Joint positions (6)
        episode_data[:, 6:12],   # Joint velocities (6)
        episode_data[:, 18:21],  # End-effector position (3)
        episode_data[:, 21:24],  # Object position (3)
        episode_data[:, 24:28],  # Object quaternion (4)
    ], axis=1)

    return inputs


def extract_outputs(episode_data: np.ndarray) -> np.ndarray:
    joint_vels = episode_data[:, 12:18]
    gripper = episode_data[:, 28:29]

    gripper = gripper / 255.0

    outputs = np.concatenate([joint_vels, gripper], axis=1)

    return outputs


def compute_normalization_stats(train_episodes: List[np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    all_inputs = []
    all_outputs = []

    for episode in train_episodes:
        inputs = extract_inputs(episode)
        outputs = extract_outputs(episode)
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

    output_mean[6] = 0.0
    output_std[6] = 1.0

    input_stats = {'mean': input_mean, 'std': input_std}
    output_stats = {'mean': output_mean, 'std': output_std}

    print(f"Computed normalization stats from {len(train_episodes)} training episodes")
    print(f"Total training timesteps: {len(all_inputs)}")

    return input_stats, output_stats


class RobotTrajectoryDataset(Dataset):
    def __init__(self, episodes: List[np.ndarray],
                 input_stats: Dict[str, np.ndarray],
                 output_stats: Dict[str, np.ndarray]):
        self.input_stats = input_stats
        self.output_stats = output_stats

        all_inputs = []
        all_outputs = []

        for episode in episodes:
            inputs = extract_inputs(episode)
            outputs = extract_outputs(episode)
            all_inputs.append(inputs)
            all_outputs.append(outputs)

        # Concatenate all timesteps
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)

        normalized_inputs = (all_inputs - input_stats['mean']) / input_stats['std']
        normalized_outputs = (all_outputs - output_stats['mean']) / output_stats['std']

        self.inputs = torch.tensor(normalized_inputs, dtype=torch.float32)
        self.outputs = torch.tensor(normalized_outputs, dtype=torch.float32)

        print(f"Created dataset with {len(self.inputs)} timesteps")

    def __len__(self) -> int:
        """Return total number of timesteps."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single (input, output) pair.

        Args:
            idx: Index of the timestep

        Returns:
            (input_tensor, output_tensor): (22D, 7D) tensors
        """
        return self.inputs[idx], self.outputs[idx]
