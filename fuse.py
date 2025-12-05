import torch

import numpy as np
import open3d as o3d

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from src.entities.arguments import OptimizationParams
from src.entities.gaussian_model import GaussianModel
from src.entities.datasets import get_dataset
from src.evaluation.evaluator import filter_depth_outliers
from src.utils.io_utils import load_config
from src.utils.utils import setup_seed, torch2np, render_gaussian_model, get_render_settings
from src.utils.validation import compare_depth_scale

if __name__ == "__main__":
    parser = ArgumentParser(description="Fuse depth maps into a 3D mesh")
    parser.add_argument("-m", "--model_dir", type=str, required=True)
    parser.add_argument("-s", "--use_label", action="store_true", help="Whether to use semantic labels during fusion")
    parser.add_argument("-v", "--val_depth", action="store_true", help="Whether to validate rendered depth with GT depth")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config_file = model_dir / "config.yaml"

    mesh_dir = model_dir / "mesh"
    mesh_dir.mkdir(exist_ok=True)

    depth_dir = model_dir / "depth"
    depth_dir.mkdir(exist_ok=True)

    config = load_config(config_file.as_posix())
    setup_seed(config["seed"])
    print(f"Loaded config: {config_file.as_posix()}")

    dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"]})
    gt_poses = np.array(dataset.poses)
    fx, fy = dataset.intrinsics[0, 0], dataset.intrinsics[1, 1]
    cx, cy = dataset.intrinsics[0, 2], dataset.intrinsics[1, 2]

    opt_settings = OptimizationParams(ArgumentParser(description="Training script parameters"))
    intrinsics = o3d.camera.PinholeCameraIntrinsic(dataset.width, dataset.height, fx, fy, cx, cy)
    scale = 1.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    camera_poses = torch2np(torch.load(model_dir / "estimated_c2w.ckpt", map_location="cuda"))
    submap_paths = sorted(list((model_dir / "submaps").glob('*')))

    for submap_path in tqdm(submap_paths, desc="Fusing submaps", ncols=120):
        submap = torch.load(submap_path, map_location="cuda")
        gaussian_model = GaussianModel()
        gaussian_model.training_setup(opt_settings)
        gaussian_model.restore_from_params(submap["gaussian_params"], opt_settings)

        for keyframe_id in submap["submap_keyframes"]:
            estimate_c2w = camera_poses[keyframe_id]
            estimate_w2c = np.linalg.inv(estimate_c2w)

            render_dict = render_gaussian_model(
                gaussian_model, get_render_settings(dataset.width, dataset.height, dataset.intrinsics, estimate_w2c))
            
            rendered_depth = render_dict["depth"][0].detach().cpu().numpy() # (H, W)
            rendered_depth = filter_depth_outliers(rendered_depth, kernel_size=20, threshold=0.1)

            if args.val_depth:
                gt_depth = dataset.load_depth_map(keyframe_id)
                stats = compare_depth_scale(gt_depth, rendered_depth)
                tqdm.write(f"Keyframe {keyframe_id:>4d}: Scale={stats['scale_factor']:.4f}, RMSE={stats['rmse']:.4f} over {stats['num_valid_pixels']} pixels")

            depth_image_o3d = o3d.geometry.Image(rendered_depth)
            color_image_o3d = None

            if args.use_label:
                label_map = dataset.load_label_map(keyframe_id)
                color_image_o3d = o3d.geometry.Image(np.ascontiguousarray(label_map.astype(np.uint8)))
            else:
                rendered_color = render_dict["color"].detach()
                rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)
                rendered_color = (torch2np(rendered_color.permute(1, 2, 0)) * 255).astype(np.uint8)
                color_image_o3d = o3d.geometry.Image(np.ascontiguousarray(rendered_color))

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image_o3d, depth_image_o3d, depth_scale=scale,
                depth_trunc=30, convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsics, estimate_w2c)

    o3d_mesh = volume.extract_triangle_mesh()
    compensate_vector = (-0.0 * scale / 512.0, 2.5 * scale / 512.0, -2.5 * scale / 512.0)
    o3d_mesh = o3d_mesh.translate(compensate_vector)

    mesh_file = mesh_dir / "fused_mesh.ply" if not args.use_label else mesh_dir / "fused_mesh_semantic.ply"
    o3d.io.write_triangle_mesh(str(mesh_file), o3d_mesh)
    print(f"Saved fused mesh to {mesh_file.as_posix()}")