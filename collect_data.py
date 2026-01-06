from run_sim import run_sim
import numpy as np
from datetime import datetime
import os

if __name__ == "__main__":
    TARGET_EPISODES = 500
    MAX_ATTEMPTS = 1000 #if we have 500 failed runs it doesnt work, this wont happen
                        #because our simulations work well

    successful_episodes = []
    episode_count = 0
    attempt_count = 0


    while episode_count < TARGET_EPISODES and attempt_count < MAX_ATTEMPTS:
        np.random.seed(attempt_count)

        success, episode_data = run_sim(sleep_time=0.0, headless=True, noise_std=0.002)

        if success:
            successful_episodes.append(episode_data)
            episode_count += 1
            print(f"Episode {episode_count}/{TARGET_EPISODES} - Success! "
                  f"({episode_data.shape[0]} timesteps, attempt {attempt_count + 1})")
        else:
            print(f"Attempt {attempt_count + 1} - Failed (timeout)")

        attempt_count += 1

    if episode_count > 0:
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_dir}/pick_place_dataset_{timestamp}.npz"

        data_dict = {}
        for i, episode in enumerate(successful_episodes):
            data_dict[f'episode_{i}'] = episode

        metadata = {
            'num_episodes': episode_count,
            'total_attempts': attempt_count,
            'success_rate': episode_count / attempt_count,
            'data_format': 'joint_pos(6)|joint_vel(6)|cmd_joint_vel(6)|ee_pos(3)|ee_quat(4)|obj_pos(3)|obj_quat(4)|obj_size(3)|gripper_joint_pos(1)|gripper_joint_vel(1)|gripper_cmd(1)',
            'total_timesteps': sum(ep.shape[0] for ep in successful_episodes),
            'min_episode_length': min(ep.shape[0] for ep in successful_episodes),
            'max_episode_length': max(ep.shape[0] for ep in successful_episodes),
            'mean_episode_length': np.mean([ep.shape[0] for ep in successful_episodes]),
        }
        data_dict['metadata'] = metadata

        np.savez_compressed(filename, **data_dict)

        print(f"\n{'='*60}")
        print(f"Dataset saved to: {filename}")
        print(f"Total episodes: {metadata['num_episodes']}")
        print(f"Total timesteps: {metadata['total_timesteps']}")
        print(f"Episode length - Min: {metadata['min_episode_length']}, "
              f"Max: {metadata['max_episode_length']}, "
              f"Mean: {metadata['mean_episode_length']:.1f}")
        print(f"Success rate: {metadata['success_rate']*100:.1f}%")
        print(f"{'='*60}")
    else:
        print(f"\nFailed to collect any episodes after {MAX_ATTEMPTS} attempts")
