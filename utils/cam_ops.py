import torch
import open3d as o3d
from utils.typing import *


def compose_extrinsic_R_T(R: torch.Tensor, T: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from R and T.
    Batched I/O.
    """
    RT = torch.cat((R, T.unsqueeze(-1)), dim=-1)
    return compose_extrinsic_RT(RT)


def compose_extrinsic_RT(RT: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from RT.
    Batched I/O.
    """
    return torch.cat([
        RT,
        torch.tensor([[[0, 0, 0, 1]]], dtype=RT.dtype, device=RT.device).repeat(RT.shape[0], 1, 1)
        ], dim=1)


def decompose_extrinsic_R_T(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into R and T.
    Batched I/O.
    """
    RT = decompose_extrinsic_RT(E)
    return RT[:, :, :3], RT[:, :, 3]


def decompose_extrinsic_RT(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into RT.
    Batched I/O.
    """
    return E[:, :3, :]

def get_normalized_camera_intrinsics(intrinsics: torch.Tensor):
    """
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    Return batched fx, fy, cx, cy
    """
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 0, 1]
    cx, cy = intrinsics[:, 1, 0], intrinsics[:, 1, 1]
    width, height = intrinsics[:, 2, 0], intrinsics[:, 2, 1]
    fx, fy = fx / width, fy / height
    cx, cy = cx / width, cy / height
    return fx, fy, cx, cy

def camera_normalization_objaverse(normed_dist_to_center, poses: torch.Tensor, ret_transform: bool = False):
    assert normed_dist_to_center is not None
    pivotal_pose = compose_extrinsic_RT(poses[:1])
    dist_to_center = pivotal_pose[:, :3, 3].norm(dim=-1, keepdim=True).item() \
        if normed_dist_to_center == 'auto' else normed_dist_to_center

    # compute camera norm (new version)
    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 0, -1, -dist_to_center],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]], dtype=torch.float32)
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm_matrix = torch.bmm(canonical_camera_extrinsics, pivotal_pose_inv)

    # normalize all views
    poses = compose_extrinsic_RT(poses)
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)
    poses = decompose_extrinsic_RT(poses)

    if ret_transform:
        return poses, camera_norm_matrix.squeeze(dim=0)
    return poses

def camera_relative_pose_koolai(poses: Float[Tensor, "N 4 4"], ret_transform: bool = False):
    pivotal_pose = poses[:1]
    pivotal_pose_inv = torch.inverse(pivotal_pose)

    # normalize all views
    poses = torch.bmm(pivotal_pose_inv.repeat(poses.shape[0], 1, 1), poses)

    if ret_transform:
        return poses, pivotal_pose_inv.squeeze(dim=0)
    return poses


def build_camera_principle(RT: torch.Tensor, intrinsics: torch.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    return torch.cat([
        RT.reshape(-1, 12),
        fx.unsqueeze(-1), fy.unsqueeze(-1), cx.unsqueeze(-1), cy.unsqueeze(-1),
    ], dim=-1)


def build_camera_standard(RT: torch.Tensor, intrinsics: torch.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    E = compose_extrinsic_RT(RT)
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    I = torch.stack([
        torch.stack([fx, torch.zeros_like(fx), cx], dim=-1),
        torch.stack([torch.zeros_like(fy), fy, cy], dim=-1),
        torch.tensor([[0, 0, 1]], dtype=torch.float32, device=RT.device).repeat(RT.shape[0], 1),
    ], dim=1)
    return torch.cat([
        E.reshape(-1, 16),
        I.reshape(-1, 9),
    ], dim=-1)
    
def build_spherical_camera_standard(extrinsic: torch.Tensor):
    """
    extrinsic: (N, 4, 4)
    """
    E = extrinsic
    return E.reshape(-1, 16)

def project_points(points: torch.Tensor, extrinsic: torch.Tensor):
    """ project 3D points from world to camera space, or vice versa
    points: (B, N, 3)
    extrinsic: (B, 4, 4)
    """
    points = torch.cat([points, torch.ones_like(points[:, :, :1])], dim=-1)
    points = torch.bmm(extrinsic, points.permute(0, 2, 1)).permute(0, 2, 1)
    points = points[:,:,:3] / points[:, :, 3:]
    return points

def cvt_to_perspective_pointcloud(rgb_image: torch.Tensor,
                                  depth_image: torch.Tensor, 
                                  depth_scale: float = 1.0,
                                  hfov_rad: float = 0.5*np.pi,
                                  wfov_rad: float = 0.5*np.pi,):
    """
    rgb: (3, H, W)
    depth: (1, H, W)
    
    """
    C, H, W = depth_image.shape
    K = torch.tensor([
                [1 / np.tan(wfov_rad / 2.), 0., 0., 0.],
                [0., 1 / np.tan(hfov_rad / 2.), 0., 0.],
                [0., 0.,  1, 0],
                [0., 0., 0, 1]]).float().to(depth_image.device)

    # Now get an approximation for the true world coordinates -- see if they make sense
    # [-1, 1] for x and [-1, -1] for y as array indexing is y-down
    ys, xs = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing='ij')
    xs = xs.reshape(1,H,W).to(depth_image.device)
    ys = ys.reshape(1,H,W).to(depth_image.device)

    # Unproject
    # positive depth as the camera looks along Z
    depth = depth_image / depth_scale
    xys = torch.cat([xs * depth , ys * depth, depth, torch.ones_like(depth)], dim=0)
    xys = xys.reshape(4, -1)
    points = torch.inverse(K) @ xys
    
    o3d_pointcloud = o3d.geometry.PointCloud()
    colors = rgb_image.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    colors = np.clip(colors, 0, 1.0)
    points = points.permute(1, 0).cpu().numpy()[:, :3]
    o3d_pointcloud.points = o3d.utility.Vector3dVector(points)
    o3d_pointcloud.colors = o3d.utility.Vector3dVector(colors)
    return o3d_pointcloud