import torch
import numpy as np
import torch.nn.functional as F
from pdb import set_trace as bp
from scipy.spatial.transform import Rotation as R


def bgs(d6s):
    # print(d6s.shape)
    b_copy = d6s.clone()
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1),
                                    a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1)

def process_action(raw_action, translation):
    # [batch, 9] -> [batch, 4, 4]
    # given a 9D vector of action(3translation + 6rotation), convert it to a 4x4 matrix of SE3
    # assert raw_action.shape[-1] == 9
    batch_size = raw_action.shape[0]
    action = torch.zeros(batch_size, 4, 4, device=raw_action.device)
    action[:,3,3] = 1
    action[:,:3,3] += translation[:,:] # translation
    R = bgs(raw_action[:,3:].reshape(-1, 2, 3).permute(0, 2, 1))
    action[:,:3,:3] += R # rotation
    return action


def orthogonalization(raw_action):
    # [batch, 9] -> [batch, 4, 4]
    batch_size = raw_action.shape[0]
    R = bgs(raw_action[:,3:].reshape(-1, 2, 3).permute(0, 2, 1))
    return R

def bgdR(Rgts, Rps):
    Rgts = Rgts.float()
    Rps = Rps.float()
    Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
    Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1) #batch trace
    # necessary or it might lead to nans and the likes
    theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
    return torch.acos(theta)

def SE3dist(Rgts, Rps, connection_cfg=None):
    """
    Compute the distance between two SE3 transformations.
    Args:
        Rgts: Ground truth SE3 transformation (b, 4, 4)
        Rps: Predicted SE3 transformation (b, 4, 4)
    Returns:
        R_dist: Rotation distance in radians
        t_dist: Translation distance
    """
    axis_r = connection_cfg.axis_r if connection_cfg is not None else None
    axis_t = connection_cfg.axis_t if connection_cfg is not None else None

    if isinstance(Rgts, np.ndarray):
        Rgts = torch.tensor(Rgts, dtype=torch.float32)
    if isinstance(Rps, np.ndarray):
        Rps = torch.tensor(Rps, dtype=torch.float32)

    if Rgts.shape == (4, 4):
        Rgts = Rgts[None, :, :]
    if Rps.shape == (4, 4):
        Rps = Rps[None, :, :]

    # Extract rotation matrices and translation vectors
    Rgt = Rgts[:, :3, :3]
    Rp = Rps[:, :3, :3]
    t_gt = Rgts[:, :3, 3]
    t_p = Rps[:, :3, 3]
    
    # Compute rotation distance
    if axis_r is None:
        R_dist = bgdR(Rgt, Rp)
    else:
        R_rel = np.array(torch.bmm(Rgt.permute(0, 2, 1), Rp))
        r = R.from_matrix(R_rel).as_rotvec()
        axis_r = axis_r / np.linalg.norm(axis_r)
        r_proj = r - np.dot(r, axis_r) * axis_r
        R_dist = np.linalg.norm(r_proj)

    if axis_t is None:
        # when axis is None, we consider the full translation distance, where the tangential is equal to normal distance
        t_tangential = torch.norm(t_gt - t_p, dim=1)
        t_normal = t_tangential
    else:
        t_diff = t_gt - t_p
        t_tangential = np.linalg.norm(t_diff * axis_t)
        t_normal = np.linalg.norm(t_diff - t_tangential * axis_t)

    return R_dist, t_tangential, t_normal