import numpy as np

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / (n + eps)

def rodrigues(axis, angle):
    """axis must be unit."""
    wx, wy, wz = axis
    K = np.array([[0, -wz,  wy],
                  [wz,  0, -wx],
                  [-wy, wx,  0]], dtype=float)
    I = np.eye(3)
    return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def align_vectors(v_from, v_to, eps=1e-9):
    v_from = normalize(v_from)
    v_to   = normalize(v_to)

    k = np.cross(v_from, v_to)
    s = np.linalg.norm(k)
    c = float(np.dot(v_from, v_to))

    if s < eps:
        # Already aligned (or opposite). If opposite, choose an arbitrary orthogonal axis.
        if c > 0:
            return np.eye(3)
        # 180°: pick an axis orthogonal to v_from
        ortho = np.array([1., 0., 0.]) if abs(v_from[0]) < 0.9 else np.array([0., 1., 0.])
        axis = normalize(np.cross(v_from, ortho))
        return rodrigues(axis, np.pi)

    axis = k / s
    angle = np.arctan2(s, c)
    return rodrigues(axis, angle)

def make_basis(z_world, x_world_hint):
    z = normalize(z_world)
    x = x_world_hint - z * (x_world_hint @ z)   # remove any component along z
    x = normalize(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])

def rotation_vertical_and_align_to_cuboid(R_WE, R_WO, down_W, a_E, c_E):
    v_W = R_WE @ a_E
    R_align = align_vectors(v_W, down_W)

    R_vert = R_align @ R_WE

    ox_W = R_WO[:, 0]
    oy_W = R_WO[:, 1]

    def horiz(u):
        u = u - down_W * (u @ down_W)
        return normalize(u)

    ox_h = horiz(ox_W)
    oy_h = horiz(oy_W)

    c_W_current = R_vert @ c_E
    c_h_current = horiz(c_W_current)

    score_x = abs(c_h_current @ ox_h)
    score_y = abs(c_h_current @ oy_h)
    chosen = ox_h if score_x >= score_y else oy_h

    if (c_h_current @ chosen) < 0:
        chosen = -chosen

    c_W_desired = chosen  # desired world closing direction, horizontal + best-aligned

    B_W = make_basis(down_W, c_W_desired)

    B_E = make_basis(a_E, c_E)

    R_target = B_W @ B_E.T
    return R_target

def get_rotation_matrix(T_WE, a_E, c_E, down_W, R_WO, current_step):
    R_WE = T_WE.as_matrix()[:3, :3]
    if current_step in [1,2,3]:
        # R_WO is already a rotation matrix, passed directly
        R_target = rotation_vertical_and_align_to_cuboid(
            R_WE=R_WE,
            R_WO=R_WO,
            down_W=down_W,
            a_E=a_E,
            c_E=c_E,
        )
    else:        
        v_W = R_WE @ a_E

        k = np.cross(v_W, down_W)
        s = np.linalg.norm(k)
        c = float(np.dot(v_W, down_W))

        if s < 1e-9:
            R_align = np.eye(3)
        else:
            axis = k / s
            angle = np.arctan2(s, c)
            wx, wy, wz = axis
            K=np.array([[0, -wz,  wy],
                    [wz,  0, -wx],
                    [-wy, wx,  0]], dtype=float)
            R_align=np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


        R_target = R_align @ R_WE
        
    return R_target

