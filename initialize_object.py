import mujoco
import numpy as np

def initialize_object(model, data):
    table_geom = model.geom("table_top")
    table_geom_id = table_geom.id
    table_body = model.body("table")
    table_body_id = table_body.id

    table_body_pos = model.body_pos[table_body_id]
    table_geom_pos = model.geom_pos[table_geom_id]
    table_size = model.geom_size[table_geom_id]

    table_center = table_body_pos + table_geom_pos
    table_top_z = table_center[2] + table_size[2]

    table_x_min = table_center[0] - table_size[0]
    table_x_max = table_center[0] + table_size[0]
    table_y_min = table_center[1] - table_size[1]
    table_y_max = table_center[1] + table_size[1]

    object_geom = model.geom("object_geom")
    object_geom_id = object_geom.id
    
    # Randomize cuboid dimensions (half-extents)
    width = np.random.uniform(0.02, 0.05)
    depth = np.random.uniform(0.02, 0.05)
    height = np.random.uniform(0.015, 0.04)
    model.geom_size[object_geom_id] = np.array([width, depth, height])
    
    object_size = model.geom_size[object_geom_id]
    object_half_height = object_size[2]

    object_joint = model.joint("object")
    object_joint_id = object_joint.id
    qpos_start = model.jnt_qposadr[object_joint_id]

    margin = 0.05
    robot_base_xy = np.array([0.0, 0.0])  # Robot base is at (0, 0, 0.4)
    min_distance_from_robot = 0.35  # Minimum 35cm from robot base

    # Rejection sampling: keep generating positions until we find one far enough from robot
    max_attempts = 100
    for _ in range(max_attempts):
        pos_xy = np.array([
            np.random.uniform(table_x_min + margin, table_x_max - margin),
            np.random.uniform(table_y_min + margin, table_y_max - margin)
        ])
        distance_from_robot = np.linalg.norm(pos_xy - robot_base_xy)
        if distance_from_robot >= min_distance_from_robot:
            break

    pos = np.array([pos_xy[0], pos_xy[1], table_top_z + object_half_height])

    angle = np.random.uniform(0, 2*np.pi)
    quat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
    data.qpos[qpos_start:qpos_start+7] = np.concatenate([pos, quat])

    qvel_start = model.jnt_dofadr[object_joint_id]
    data.qvel[qvel_start:qvel_start+6] = 0

    mujoco.mj_forward(model, data)
