import torch
import viser
import time
import torch.nn.functional as F

from argparse import ArgumentParser
from pathlib import Path

from gsplat.rendering import rasterization
from nerfview import CameraState, RenderTabState, apply_float_colormap
from splat_viewer import GaussianViewer, GaussianRenderTabState

RENDER_MODE_MAP = {
    "rgb": "RGB",
    "depth(accumulated)": "D",
    "depth(expected)": "ED",
    "alpha": "RGB",
}

def main(args, data):
    means = data['xyz'].to("cuda")
    quats = F.normalize(data['rotation'].to("cuda"), p=2, dim=-1)
    scales = torch.exp(data['scaling'].to("cuda"))
    opacities = torch.sigmoid(data['opacity'].squeeze(-1).to("cuda"))
    colors = data['features_dc'].to("cuda")
    sh_degree = 0

    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GaussianRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to("cuda")
        K = torch.from_numpy(K).float().to("cuda")
        viewmat = c2w.inverse()

        render_colors, render_alphas, info = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, sh_degree)
                if sh_degree is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device="cuda") / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
            with_ut=False,
            with_eval3d=False,
        )
        render_tab_state.total_gaussian_count = len(means)
        render_tab_state.rendered_gaussian_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap).cpu().numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders
    
    server = viser.ViserServer(port=args.port)
    _ = GaussianViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )

    print("Viewer running... Ctrl+C to exit.")
    while True: time.sleep(16.0)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    print("Loading checkpoint")
    ckpt = torch.load(f"{args.output_dir}/submaps/000005.ckpt", map_location="cuda")
    data = ckpt['gaussian_params']

    main(args, data)