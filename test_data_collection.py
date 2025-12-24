"""Test script to verify data collection with a small batch of 5 episodes."""
from run_sim import run_sim
import numpy as np
from datetime import datetime
import os

if __name__ == "__main__":
    TARGET_EPISODES = 5
    MAX_ATTEMPTS = 20

    successful_episodes = []
    episode_count = 0
    attempt_count = 0

    print(f"Testing data collection with {TARGET_EPISODES} episodes...")
    print("Running in headless mode...\n")

    while episode_count < TARGET_EPISODES and attempt_count < MAX_ATTEMPTS:
        np.random.seed(attempt_count)

        success, episode_data = run_sim(sleep_time=0.0, headless=True)

        if success:
            successful_episodes.append(episode_data)
            episode_count += 1
            print(f"Episode {episode_count}/{TARGET_EPISODES} - Success! "
                  f"({episode_data.shape[0]} timesteps, attempt {attempt_count + 1})")
            print(f"  Data shape: {episode_data.shape}")
            print(f"  Sample data (first timestep): {episode_data[0][:10]}...")  # Show first 10 values
        else:
            print(f"Attempt {attempt_count + 1} - Failed (timeout)")

        attempt_count += 1

    if episode_count > 0:
        print(f"\n{'='*60}")
        print(f"Test Results:")
        print(f"Successful episodes: {episode_count}/{attempt_count}")
        print(f"Success rate: {episode_count/attempt_count*100:.1f}%")

        for i, ep in enumerate(successful_episodes):
            assert ep.shape[1] == 23, f"Episode {i} has incorrect number of features: {ep.shape[1]}"
            assert ep.shape[0] > 0, f"Episode {i} has no timesteps"
            assert ep.shape[0] <= 10000, f"Episode {i} exceeds max timesteps: {ep.shape[0]}"
            print(f"Episode {i}: {ep.shape[0]} timesteps - ✓")

        print("\nAll episodes have correct shape (timesteps, 23)")

        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_dir}/test_dataset_{timestamp}.npz"

        data_dict = {}
        for i, episode in enumerate(successful_episodes):
            data_dict[f'episode_{i}'] = episode

        metadata = {
            'num_episodes': episode_count,
            'total_attempts': attempt_count,
            'success_rate': episode_count / attempt_count,
            'data_format': 'joint_pos(6)|joint_vel(6)|ee_pos(3)|obj_pos(3)|obj_quat(4)|gripper(1)',
            'total_timesteps': sum(ep.shape[0] for ep in successful_episodes),
            'min_episode_length': min(ep.shape[0] for ep in successful_episodes),
            'max_episode_length': max(ep.shape[0] for ep in successful_episodes),
            'mean_episode_length': np.mean([ep.shape[0] for ep in successful_episodes]),
        }
        data_dict['metadata'] = metadata

        np.savez_compressed(filename, **data_dict)

        print(f"\nTest dataset saved to: {filename}")
        print(f"{'='*60}")

        print("\nTesting data loading...")
        loaded_data = np.load(filename, allow_pickle=True)
        loaded_metadata = loaded_data['metadata'].item()
        print(f"Loaded {loaded_metadata['num_episodes']} episodes successfully")
        print(f"Episode 0 shape: {loaded_data['episode_0'].shape}")

        episode_0 = loaded_data['episode_0']
        print("\nVerifying data columns (from first timestep of episode 0):")
        print(f"  Joint positions (6): {episode_0[0, 0:6]}")
        print(f"  Joint velocities (6): {episode_0[0, 6:12]}")
        print(f"  EE position (3): {episode_0[0, 12:15]}")
        print(f"  Object position (3): {episode_0[0, 15:18]}")
        print(f"  Object quaternion (4): {episode_0[0, 18:22]}")
        print(f"  Gripper command (1): {episode_0[0, 22]}")

        print("\n✓ All tests passed! Data collection is working correctly.")
    else:
        print(f"\n✗ Failed to collect any episodes after {MAX_ATTEMPTS} attempts")
