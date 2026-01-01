import mujoco
import mujoco.viewer
import numpy as np
import mink
import mink.limits
import time
import torch
import argparse
from pathlib import Path
from initialize_object import initialize_object
from rotation_matrix import get_rotation_matrix
from model import MLP
from utils import load_checkpoint
import imageio

def run_sim(sleep_time=0.0, headless=False):
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)
    initialize_object(model, data)

    table_geom = model.geom("table_top")
    table_body = model.body("table")
    table_body_z = model.body_pos[table_body.id][2]
    table_geom_z = model.geom_pos[table_geom.id][2]
    table_geom_half_height = model.geom_size[table_geom.id][2]
    table_top_z = table_body_z + table_geom_z + table_geom_half_height

    object_geom = model.geom("object_geom")
    object_half_height = model.geom_size[object_geom.id][2]
    target_placement = np.array([0.6, 0.3, table_top_z + object_half_height])
    
    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="grasp_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.1,
            lm_damping=1e-6,
        ),
        posture_task := mink.PostureTask(model, cost=1e-3),
    ]
    dt=model.opt.timestep
    

    limits = [
        mink.ConfigurationLimit(model=model),
    ]

    table_plane_id = model.geom("table_collision_plane").id
    gripper_geoms = [
        model.geom("gripper_base_collision").id,
        model.geom("right_follower_collision").id,
        model.geom("left_follower_collision").id,
        model.geom("right_pad1").id,
        model.geom("right_pad2").id,
        model.geom("left_pad1").id,
        model.geom("left_pad2").id,
        model.geom("wrist_3_link").id,
        model.geom("wrist_2_link").id,
        model.geom("wrist_1_collision").id,
        model.geom("forearm_collision_1").id,
        model.geom("forearm_collision_2").id,
        model.geom("upper_arm_collision_1").id,
        model.geom("upper_arm_collision_2").id,
        model.geom("shoulder_collision").id,
    ]

    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=[([table_plane_id], [gid]) for gid in gripper_geoms],
        minimum_distance_from_collisions=0.01,
        collision_detection_distance=0.05,
    )
    limits.append(collision_avoidance_limit)

    max_velocities = {
        "shoulder_pan": np.pi/3,
        "shoulder_lift": np.pi/3,
        "elbow": np.pi/3,
        "wrist_1": np.pi/3,
        "wrist_2": np.pi/3,
        "wrist_3": np.pi/3,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    solver = "daqp"

    current_step = 0
    step_threshold = 0.02

    def check_gripper_object_contact(data, model):
        object_geom_id = model.geom("object_geom").id
        left_contact = False
        right_contact = False

        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            if geom1 == object_geom_id or geom2 == object_geom_id:
                geom_name1 = model.geom(geom1).name
                geom_name2 = model.geom(geom2).name

                if 'left_pad' in geom_name1 or 'left_pad' in geom_name2:
                    left_contact = True
                if 'right_pad' in geom_name1 or 'right_pad' in geom_name2:
                    right_contact = True

        return left_contact and right_contact


    object_joint_id = model.joint("object").id
    obj_qpos_start = model.jnt_qposadr[object_joint_id]
    obj_quat_start = obj_qpos_start + 3
    initial_object_pos = data.qpos[obj_qpos_start:obj_qpos_start+3].copy()

    if headless:
        viewer = None
    else:
        viewer = mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        )
    try:
        if viewer is not None:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        configuration.update_from_keyframe("home")
        posture_task.set_target(configuration.q)
        mujoco.mj_forward(model, data)
        
        object_geom_id = model.geom("object_geom").id


        target_positions = [
            initial_object_pos + np.array([0, 0, 0.3]),  # Step 0: 30cm above initial object
            initial_object_pos + np.array([0, 0, 0.1]),  # Step 1: 10cm above initial object, helps vertical approach
            initial_object_pos,  # Step 2: At object middle
            initial_object_pos,  # Step 3: Grasp at object middle
            initial_object_pos + np.array([0, 0, 0.3]),  # Step 4: Lift to 30cm above initial pos
            target_placement + np.array([0, 0, 0.3]),  # Step 5: Above target
            target_placement + np.array([0, 0, 0.1]),  # Step 6: to help slow down approach
            target_placement + np.array([0, 0, 0.02]),  # Step 7: Placing block, dont release when it hits the table
            target_placement,  # Step 8: Release block
            target_placement + np.array([0, 0, 0.3]),  # Step 9: Up from target
        ]
      
        down_W = np.array([0., 0., -1.])
        T_WE = configuration.get_transform_frame_to_world("grasp_site", "site")
        R_WE = T_WE.as_matrix()[:3, :3]
        a_E = np.array([0., 0., 1.])
        c_E = np.array([1., 0., 0.])
        
        waiting_for_grasp = False
        waiting_for_release = False

        # Initialize data collection
        timestep_data = []
        MAX_TIMESTEPS = 10000
        episode_success = False

        iterations = 0
        while (viewer is None or viewer.is_running()) and iterations < MAX_TIMESTEPS:
            iterations += 1
            if sleep_time > 0:
                time.sleep(sleep_time) #just so i can watch it
                
            
            if current_step == 1:
                end_effector_task.set_orientation_cost(1.0)
            elif current_step == 4:
                end_effector_task.set_orientation_cost(0.1)
            elif current_step == 6:
                end_effector_task.set_orientation_cost(1.0)

            T_WE = configuration.get_transform_frame_to_world("grasp_site", "site")

            obj_quat = data.qpos[obj_quat_start:obj_quat_start+4]
            obj_rotmat = np.zeros(9)
            mujoco.mju_quat2Mat(obj_rotmat, obj_quat)
            R_WO = obj_rotmat.reshape(3, 3)

            R_target = get_rotation_matrix(T_WE, a_E, c_E, down_W, R_WO, current_step)

            target_pos = target_positions[current_step]

            target_transform = mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(R_target),
                target_pos
            )
            end_effector_task.set_target(target_transform)

            qdot = mink.solve_ik(
                configuration,
                tasks,
                dt=dt,
                solver=solver,
                limits=limits,
                damping=1e-3
            )
            
            if current_step == 2:
                ee_pos = data.site("grasp_site").xpos
                r = np.linalg.norm(target_pos - ee_pos)
                qdot = qdot * np.clip(r/0.1, 0.01, 1.0)

            # Store commanded joint velocity before integration
            commanded_qdot = qdot[:6].copy()

            configuration.integrate_inplace(qdot, dt)

            data.ctrl[:6] = configuration.q[:6]

            if current_step == 3:
                data.ctrl[6] = 255
                if check_gripper_object_contact(data, model):
                    waiting_for_grasp = False
                else:
                    waiting_for_grasp = True
            elif current_step >= 4 and current_step <= 7:
                data.ctrl[6] = 255
            elif current_step == 8:
                data.ctrl[6] = 0
                if not check_gripper_object_contact(data, model):
                    waiting_for_release = False
                else:
                    waiting_for_release = True
            else:
                data.ctrl[6] = 0



            mujoco.mj_step(model, data)

            # Collect timestep data
            # Get EE orientation as quaternion
            ee_mat = data.site("grasp_site").xmat.reshape(3, 3)
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, ee_mat.flatten())

            current_timestep = np.concatenate([
                configuration.q[:6],                              # Joint positions (6)
                data.qvel[0:6],                                   # Joint velocities (6)
                commanded_qdot,                                   # Commanded joint velocities (6)
                data.site("grasp_site").xpos,                     # EE position (3)
                ee_quat,                                          # EE orientation quaternion (4)
                data.qpos[obj_qpos_start:obj_qpos_start+3],       # Object position (3)
                data.qpos[obj_qpos_start+3:obj_qpos_start+7],     # Object orientation (4)
                model.geom_size[object_geom_id],                  # Object size (3)
                np.array([data.ctrl[6]])                          # Gripper command (1)
            ])
            timestep_data.append(current_timestep)

            ee_pos = data.site("grasp_site").xpos
            distance = np.linalg.norm(ee_pos - target_pos)

            if current_step == 2 and distance < 0.01:
                current_step += 1
            elif distance < step_threshold:
                if current_step == 3:
                    if not waiting_for_grasp:
                        current_step += 1
                elif current_step == 8:
                    if not waiting_for_release:
                        current_step += 1
                else:
                    current_step += 1

            if current_step >= len(target_positions):
                print(f"Iterations: {iterations}")
                episode_success = True
                break

            if viewer is not None:
                viewer.sync()
            
    finally:
        if viewer is not None:
            viewer.close()
            time.sleep(0.3)

        # Return success status and data
        if episode_success and iterations < MAX_TIMESTEPS:
            return True, np.array(timestep_data)
        else:
            return False, None


def run_sim_with_model(checkpoint_path, sleep_time=0.0, headless=False, save_video=False, video_path="rollout.mp4", max_steps=10000, actions_per_query=None):
    # np.random.seed(38478)

    checkpoint_temp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    chunk_size = checkpoint_temp.get('chunk_size', 1)  # Default to 1 for backward compatibility
    action_dim = checkpoint_temp.get('action_dim', 7)  # Default to 7

    if actions_per_query is None:
        actions_per_query = 1

    output_size = action_dim * chunk_size

    policy_model = MLP(
        input_size=30,
        hidden_size=256,
        num_hidden_layers=3,
        output_size=output_size
    )

    checkpoint_data = load_checkpoint(checkpoint_path, policy_model)
    input_stats = checkpoint_data['input_stats']
    output_stats = checkpoint_data['output_stats']

    policy_model.eval()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    policy_model = policy_model.to(device)

    mj_model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(mj_model)
    configuration = mink.Configuration(mj_model)
    initialize_object(mj_model, data)

    object_joint_id = mj_model.joint("object").id
    obj_qpos_start = mj_model.jnt_qposadr[object_joint_id]
    obj_quat_start = obj_qpos_start + 3
    object_geom_id = mj_model.geom("object_geom").id

    # Setup video recording if requested
    frames = []
    renderer = None
    if save_video:
        renderer = mujoco.Renderer(mj_model, height=480, width=640)
        print(f"Saving video to: {video_path}")

    if headless:
        viewer = None
    else:
        viewer = mujoco.viewer.launch_passive(
            model=mj_model, data=data, show_left_ui=False, show_right_ui=False
        )

    try:
        if viewer is not None:
            mujoco.mjv_defaultFreeCamera(mj_model, viewer.cam)

        configuration.update_from_keyframe("home")
        data.qpos[:6] = configuration.q[:6]
        mujoco.mj_forward(mj_model, data)

        iterations = 0
        action_chunk = None
        chunk_idx = 0

        with torch.no_grad():
            while (viewer is None or viewer.is_running()) and iterations < max_steps:
                iterations += 1

                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Query model if we've used up our actions or it's the first iteration
                if action_chunk is None or chunk_idx >= actions_per_query:
                    # Get EE orientation as quaternion
                    ee_mat = data.site("grasp_site").xmat.reshape(3, 3)
                    ee_quat = np.zeros(4)
                    mujoco.mju_mat2Quat(ee_quat, ee_mat.flatten())

                    # Construct model input (30D):
                    # Joint pos(6) + vel(6) + EE pos(3) + EE quat(4) + obj pos(3) + obj quat(4) + obj size(3) + gripper(1)
                    model_input = np.concatenate([
                        data.qpos[:6],                                    # Joint positions (6)
                        data.qvel[:6],                                    # Joint velocities (6)
                        data.site("grasp_site").xpos,                     # EE position (3)
                        ee_quat,                                          # EE orientation quaternion (4)
                        data.qpos[obj_qpos_start:obj_qpos_start+3],       # Object position (3)
                        data.qpos[obj_quat_start:obj_quat_start+4],       # Object quaternion (4)
                        mj_model.geom_size[object_geom_id],               # Object size (3)
                        np.array([data.ctrl[6] / 255.0]),                 # Gripper state (1), normalized
                    ])

                    normalized_input = (model_input - input_stats['mean']) / input_stats['std']

                    input_tensor = torch.tensor(normalized_input, dtype=torch.float32, device=device).unsqueeze(0)

                    normalized_output = policy_model(input_tensor).squeeze(0)

                    output = normalized_output.cpu().numpy() * output_stats['std'] + output_stats['mean']

                    # Reshape output to (chunk_size, action_dim)
                    action_chunk = output.reshape(chunk_size, action_dim)
                    chunk_idx = 0

                # Extract the current action from the chunk
                current_action = action_chunk[chunk_idx]
                chunk_idx += 1

                joint_velocities = current_action[:6]
                gripper_logit = current_action[6]

                gripper_command = 255.0 if gripper_logit > 0.5 else 0.0

                dt = mj_model.opt.timestep
                configuration.q[:6] = data.qpos[:6]

                qdot = np.zeros(mj_model.nv)
                qdot[:6] = joint_velocities 

                configuration.integrate_inplace(qdot, dt)

                data.ctrl[:6] = configuration.q[:6]
                data.ctrl[6] = gripper_command

                mujoco.mj_step(mj_model, data)

                # Capture frame for video if recording
                if save_video and renderer is not None:
                    renderer.update_scene(data)
                    pixels = renderer.render()
                    frames.append(pixels)

                if viewer is not None:
                    viewer.sync()

                if iterations % 100 == 0 and not headless:
                    ee_pos = data.site("grasp_site").xpos
                    obj_pos = data.qpos[obj_qpos_start:obj_qpos_start+3]
                    print(f"Step {iterations}: EE={ee_pos}, Obj={obj_pos}, Gripper={gripper_command:.0f}")

    finally:
        if viewer is not None:
            viewer.close()
            time.sleep(0.3)

    print(f"\nSimulation completed after {iterations} steps")

    # Save video if recording was enabled
    if save_video and len(frames) > 0:
        print(f"Saving video with {len(frames)} frames to {video_path}...")
        imageio.mimsave(video_path, frames, fps=int(1.0 / mj_model.opt.timestep))
        print(f"Video saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run robot pick-and-place simulation')
    parser.add_argument('--mode', type=str, default='ik', choices=['ik', 'model'],
                       help='Simulation mode: "ik" for inverse kinematics (default), "model" for trained model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint (only used in model mode)')
    parser.add_argument('--sleep', type=float, default=0.01,
                       help='Sleep time between steps (default: 0.01s)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no visualization)')
    parser.add_argument('--save-video', action='store_true',
                       help='Save video of the rollout to an MP4 file')
    parser.add_argument('--video-path', type=str, default='rollout.mp4',
                       help='Path to save the video (default: rollout.mp4)')
    parser.add_argument('--max-steps', type=int, default=10000,
                       help='Maximum number of simulation steps (default: 10000)')
    parser.add_argument('--actions-per-query', type=int, default=None,
                       help='Number of actions to execute before re-querying the model (default: chunk_size/2)')

    args = parser.parse_args()

    if args.mode == 'ik':
        print("Running simulation with inverse kinematics and motion planning...")
        success, data = run_sim(sleep_time=args.sleep, headless=args.headless)
        if success:
            print("Task completed successfully!")
        else:
            print("Task failed or timed out")

    elif args.mode == 'model':
        if not Path(args.checkpoint).exists():
            print(f"Error: Checkpoint file not found at {args.checkpoint}")
            print("Please train a model first or specify a valid checkpoint path with --checkpoint")
            exit(1)

        print("Running simulation with trained model...")
        run_sim_with_model(
            checkpoint_path=args.checkpoint,
            sleep_time=args.sleep,
            headless=args.headless,
            save_video=args.save_video,
            video_path=args.video_path,
            max_steps=args.max_steps,
            actions_per_query=args.actions_per_query
        )