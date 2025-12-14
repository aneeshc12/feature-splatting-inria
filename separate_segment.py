import numpy as np
import torch
import pickle
from scene import Scene
import os
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import skip_feat_decoder
from scipy.spatial.transform import Rotation as R
import featsplat_editor
from einops import einsum
from typing import List

import open3d as o3d
import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry

def get_arrow(end, origin=np.array([0, 0, 0]), scale=1):
    import open3d as o3d
    assert(not np.all(end == origin))
    vec = end - origin
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/17.5 * scale,
        cone_height=size*0.2 * scale,
        cylinder_radius=size/30 * scale,
        cylinder_height=size*(1 - 0.2*scale))
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return(mesh)

def select_gaussians(gaussians : GaussianModel, idx : np.array):
    gaussians._xyz = gaussians._xyz[idx]
    gaussians._features_dc = gaussians._features_dc[idx]
    gaussians._features_rest = gaussians._features_rest[idx]
    gaussians._scaling = gaussians._scaling[idx]
    gaussians._rotation = gaussians._rotation[idx]
    gaussians._opacity = gaussians._opacity[idx]
    gaussians._distill_features = gaussians._distill_features[idx]

def transform_positions_and_orientations(P, Q, R, t):
    """
    Apply a rigid transform to positions and quaternions.

    Parameters
    ----------
    P : (N, 3) ndarray
        XYZ positions
    Q : (N, 4) ndarray
        Quaternions in (x, y, z, w) order
    R : (3, 3) ndarray
        Rotation matrix
    t : (3,) ndarray
        Translation vector

    Returns
    -------
    P_out : (N, 3) ndarray
        Transformed positions
    Q_out : (N, 4) ndarray
        Transformed quaternions (x, y, z, w)
    """

    # Transform positions
    P_out = P @ R.T + t

    # Global rotation
    rot_R = R.from_matrix(R)

    # Original orientations
    rot_Q = R.from_quat(Q)

    # Apply rotation to orientations (world-frame)
    Q_out = (rot_R * rot_Q).as_quat()

    return P_out, Q_out

@torch.no_grad()
def select_gs_for_phys(dataset : ModelParams,
                  modifier_name : str,
                  iteration : int,
                  interactive_viz : bool,
                  output_name : str,):

    # set paths and declare models    
    gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    ply_editing_dir = os.path.join(scene.model_path,
                                    "point_cloud",
                                    "iteration_" + str(scene.loaded_iter),
                                    )
    
    ply_path = os.path.join(ply_editing_dir, "point_cloud.ply")
    editing_modifier_save_path = os.path.join(ply_editing_dir, modifier_name)
    output_path = os.path.join(ply_editing_dir, output_name + ".ply")

    # Load modifier and gaussian
    with open(editing_modifier_save_path, "rb") as f:
        editing_modifier_dict = pickle.load(f)

    gaussians.load_ply(ply_path)
    object_gaussian_idx = editing_modifier_dict["objects"][0]["affected_gaussian_idx"]

    select_gaussians(gaussians, object_gaussian_idx)

    # NOT ROTATIONS OR PRESERVING EMBEDDINGS!!!!
    # export the splat into another ply to be loaded elsewhere
    print("Saving selected gaussians to ply file")
    if interactive_viz:
        o3d_point_cloud = o3d.geometry.PointCloud()

        o3d_point_cloud.points = o3d.utility.Vector3dVector(gaussians.get_xyz.cpu().numpy())
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([o3d_point_cloud, mesh_frame])
    gaussians.save_ply(os.path.join(output_path))

    # editing_modifier_dict = {
    #     "scene": {
    #         "ground_R": ground_R,
    #         "ground_T": ground_T,
    #     },
    #     "objects": [
    #         # {
    #         #     "name": ','.join(fg_obj_list),
    #         #     "affected_gaussian_idx": final_obj_flag,
    #         #     "actions": [
    #         #         {
    #         #             "action": "rotate",
    #         #             "rotation": rot_mat,
    #         #         },
    #         #         {
    #         #             "action": "translate",
    #         #             "translation": translate_vec
    #         #         }
    #         #     ]
    #         # }
    #         {
    #             "name": ','.join(fg_obj_list),
    #             "affected_gaussian_idx": final_obj_flag,
    #             "actions": [
    #                 {
    #                     "action": "physics",
    #                     "particle_type": "elastic",
    #                     "infilling_surface_pts": pts_on_disk_n3,
    #                     "static_idx": rigid_obj_similarity if rigid_object_name else None
    #                 }
    #             ]
    #         },
    #         # {
    #         #     "name": BG_OBJ_NAME,
    #         #     "affected_gaussian_idx": bg_obj_idx,
    #         #     "actions": [
    #         #         {
    #         #             "action": "remove",
    #         #         }
    #         #     ]
    #         # }
    #     ]
    # }

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--modifier_name", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    
    ## need this for model.extract()
    parser.add_argument("--fg_obj_list", default="vase,flowers,plants", type=str)
    parser.add_argument("--bg_obj_list", default="tabletop,wooden table", type=str)
    parser.add_argument("--ground_plane_name", default="tabletop", type=str)
    parser.add_argument("--rigid_object_name", default="", type=str)
    parser.add_argument("--threshold", default=0.6, type=float)
    parser.add_argument("--object_select_eps", default=0.1, type=float)
    parser.add_argument("--inward_bbox_offset", default=99, type=float, help="Offset for selecting particles inward. Recommended value: 99 (no selection) or 0.1 (select some particles)")
    parser.add_argument("--final_noise_filtering", action="store_true")
    parser.add_argument("--interactive_viz", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    fg_obj_list = args.fg_obj_list.split(",")
    bg_obj_list = args.bg_obj_list.split(",")

    select_gs_for_phys(model.extract(args), args.modifier_name, args.iteration, args.interactive_viz, args.output_name)
