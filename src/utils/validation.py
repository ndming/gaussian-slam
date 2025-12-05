import numpy as np

def compare_depth_scale(gt_depth, render_depth):
    """
    gt_depth: (H,W) float numpy array, metric depth. 0 = invalid.
    render_depth: (H,W) float numpy array, rendered depth (arbitrary scale).
    """

    # 1. Build mask of valid pixels
    mask = gt_depth > 0
    if mask.sum() == 0:
        raise ValueError("No valid GT depth pixels found.")

    gt = gt_depth[mask].astype(np.float64)
    rd = render_depth[mask].astype(np.float64)

    # 2. Solve optimal scale using least squares:
    #     minimize || s*rd - gt ||^2
    #     s = (rd·gt) / (rd·rd)
    numerator = np.dot(rd, gt)
    denominator = np.dot(rd, rd)
    if denominator < 1e-12:
        raise ValueError("Rendered depth is zero everywhere.")
    scale = numerator / denominator

    # 3. Compute RMSE after scaling
    scaled_rd = scale * rd
    rmse = np.sqrt(np.mean((scaled_rd - gt)**2))

    # 4. Report
    return {
        "scale_factor": scale,
        "rmse": rmse,
        "num_valid_pixels": mask.sum()
    }