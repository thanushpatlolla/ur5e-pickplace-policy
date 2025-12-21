import mujoco
import mujoco.viewer
import numpy as np
import mink
import mink.limits
import torch.nn as nn
import torch.optim as optim
import time
from initialize_object import initialize_object

def run_sim():
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)
    initialize_object(model, data)

    table_geom = model.geom("table_top")
    table_body = model.body("table")
    table_top_z = (model.body_pos[table_body.id] + model.geom_pos[table_geom.id])[2] + model.geom_size[table_geom.id][2]
    target_placement = np.array([0.6, 0.3, table_top_z])
    
    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="grasp_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-6,
        ),
        posture_task := mink.PostureTask(model, cost=1e-3),
    ]
    

    limits = [
        mink.ConfigurationLimit(model=model),
    ]

    max_velocities = {
        "shoulder_pan": np.pi,
        "shoulder_lift": np.pi,
        "elbow": np.pi,
        "wrist_1": np.pi,
        "wrist_2": np.pi,
        "wrist_3": np.pi,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    # Keep arm links from scraping the table or object by activating velocity limits
    # whenever those geoms approach each other.
    arm_collision_geoms = [
        "shoulder_collision",
        "upper_arm_collision_1",
        "upper_arm_collision_2",
        "forearm_collision_1",
        "forearm_collision_2",
        "wrist_1_collision",
    ]
    hand_collision_geoms = [
        "wrist_2_link",
        "wrist_3_link",
        "right_pad1",
        "right_pad2",
        "left_pad1",
        "left_pad2",
    ]
    table_geoms = ["table_top"]
    grasp_target_geoms = ["object_geom"]
    collision_pairs = [
        (arm_collision_geoms, table_geoms),
        (arm_collision_geoms, grasp_target_geoms),
        (hand_collision_geoms, table_geoms),
        (hand_collision_geoms, grasp_target_geoms),
    ]
    collision_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,
        gain=0.7,
        minimum_distance_from_collisions=0.02,
        collision_detection_distance=0.05,
    )
    limits.append(collision_limit)

    solver = "daqp"

    current_step = 0
    step_threshold = 0.025  # 25mm threshold helps advance even with minor IK residuals
    
    # Get initial object position and geometry (computed once, not every frame)
    object_joint_id = model.joint("object").id
    obj_qpos_start = model.jnt_qposadr[object_joint_id]
    initial_object_pos = data.qpos[obj_qpos_start:obj_qpos_start+3].copy()

    # Get object half-height to target the top surface (not center)
    object_geom = model.geom("object_geom")
    object_half_height = model.geom_size[object_geom.id][2]  # 0.025m

    # Define all target positions using INITIAL object position
    # Include a safety waypoint above the table center before moving over the object.
    pregrasp_height = table_top_z + 0.4
    pregrasp_position = np.array([0.35, 0.0, pregrasp_height])
    # Use HIGHER approach height (25cm) to give robot room to orient vertically
    target_positions = [
        pregrasp_position,  # Step 0: Raise above table center to clear the table edge
        initial_object_pos + np.array([0, 0, 0.25]),  # Step 1: 25cm above initial object
        initial_object_pos + np.array([0, 0, 0.10]),  # Step 2: 10cm above object (intermediate)
        initial_object_pos + np.array([0, 0, object_half_height]),  # Step 3: At object top surface
        initial_object_pos + np.array([0, 0, object_half_height]),  # Step 4: Grasp at top
        initial_object_pos + np.array([0, 0, 0.25]),  # Step 5: Lift to 25cm above initial pos
        target_placement + np.array([0, 0, 0.25]),  # Step 6: Above target
        target_placement + np.array([0, 0, 0.10]),  # Step 7: 10cm above target (intermediate)
        target_placement + np.array([0, 0, object_half_height]),  # Step 8: At target top surface
        target_placement + np.array([0, 0, object_half_height]),  # Step 9: Release at surface
        target_placement + np.array([0, 0, 0.25]),  # Step 10: Up from target
    ]
    final_step_index = len(target_positions) - 1

    
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        configuration.update_from_keyframe("home")
        posture_task.set_target(configuration.q)

        vertical_orientation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])
        vertical_so3 = mink.SO3.from_matrix(vertical_orientation)
        orientation_start_step = 3  # allow the first two waypoints before enforcing verticality
        orientation_cost_vector = np.array([1.0, 1.0, 1.0])
        zero_orientation_cost = np.zeros(3)

        iteration = 0  # Counter for debug prints
        while viewer.is_running():
            # Keep IK configuration synchronized with the simulated robot state
            configuration.update(data.qpos.copy())

            target_pos = target_positions[current_step]

            ee_site = data.site("grasp_site")
            current_rotation = mink.SO3.from_matrix(ee_site.xmat.reshape(3, 3))

            target_orientation_matrix = None
            if current_step <= 8 and current_step >= orientation_start_step:
                rotation_so3 = vertical_so3
                target_orientation_matrix = vertical_orientation
                end_effector_task.set_orientation_cost(orientation_cost_vector)
            elif current_step <= 8:
                rotation_so3 = current_rotation
                target_orientation_matrix = rotation_so3.as_matrix()
                end_effector_task.set_orientation_cost(zero_orientation_cost)
            else:
                rotation_so3 = current_rotation
                target_orientation_matrix = rotation_so3.as_matrix()
                end_effector_task.set_orientation_cost(zero_orientation_cost)

            target_transform = mink.SE3.from_rotation_and_translation(
                rotation_so3,
                target_pos
            )
            end_effector_task.set_target(target_transform)

            qdot = mink.solve_ik(
                configuration,
                tasks,
                dt=1/200,
                solver=solver,
                limits=limits,
                damping=1e-2  # Increased from 1e-3 for stability near singularities
            )
            configuration.integrate_inplace(qdot, 1/200)

            data.ctrl[:6] = configuration.q[:6]
            # Gripper control: 0 = open, 255 = closed
            # Keep gripper open during approach, close during grasp
            if current_step >= 3 and current_step <= 7:
                data.ctrl[6] = 255  # Close gripper during pick and transport
            else:
                data.ctrl[6] = 0  # Open gripper during approach and after release

            mujoco.mj_step(model, data)

            # Update EE position after physics step
            ee_pos = data.site("grasp_site").xpos
            distance = np.linalg.norm(ee_pos - target_pos)

            # Debug prints every 50 iterations
            if iteration % 50 == 0:
                # Get current object position
                current_obj_pos = data.qpos[obj_qpos_start:obj_qpos_start+3]

                # Add orientation error tracking
                current_ee_rot = ee_site.xmat.reshape(3, 3)
                if (
                    current_step >= orientation_start_step
                    and current_step <= 8
                    and target_orientation_matrix is not None
                ):
                    # Compute orientation error as angle between current and target rotation
                    rot_error = current_ee_rot.T @ target_orientation_matrix
                    angle_error = np.arccos(np.clip((np.trace(rot_error) - 1) / 2, -1, 1))

                    print(f"\n=== Step {current_step} ===")
                    print(f"Current EE pos:  {ee_pos}")
                    print(f"Current obj pos: {current_obj_pos}")
                    print(f"Target pos:      {target_pos}")
                    print(f"Distance:        {distance:.4f}m (threshold: {step_threshold}m)")
                    print(f"Orientation error: {np.degrees(angle_error):.2f} degrees")
                    print(f"Current EE rotation matrix:\n{current_ee_rot}")
                else:
                    print(f"\n=== Step {current_step} ===")
                    print(f"Current EE pos:  {ee_pos}")
                    print(f"Current obj pos: {current_obj_pos}")
                    print(f"Target pos:      {target_pos}")
                    print(f"Distance:        {distance:.4f}m (threshold: {step_threshold}m)")

            iteration += 1

            if distance < step_threshold and current_step < final_step_index:
                current_step += 1

            viewer.sync()
            # time.sleep(0.04)  # 20ms delay = 50 FPS max (20x slower than real-time for observation)

if __name__ == "__main__":
    np.random.seed(54321)
    run_sim()
