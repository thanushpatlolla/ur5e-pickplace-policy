import mujoco
import mujoco.viewer
import numpy as np
import mink
import mink.limits
import time
from initialize_object import initialize_object
from rotation_matrix import get_rotation_matrix

def run_sim(sleep_time=0.0):
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

    viewer = mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    )
    try:
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

        iterations = 0
        while viewer.is_running():
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
                break
            

            viewer.sync()
            
    finally:
        viewer.close()
        time.sleep(0.3)