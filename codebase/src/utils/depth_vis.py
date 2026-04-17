import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_depth_png(path, to_meters=True):
    """
    Load a 16-bit depth PNG (e.g. NYU Depth v2).

    Returns:
        depth: float32 numpy array
    """
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise ValueError(f"Failed to load depth image: {path}")

    if depth.dtype != np.uint16:
        print(f"Warning: Expected uint16, got {depth.dtype}")

    depth = depth.astype(np.float32)

    if to_meters:
        depth = depth / 1000.0  # NYU is stored in millimeters

    return depth


def normalize_depth(depth,
                    min_depth=None,
                    max_depth=None,
                    clip=True,
                    ignore_invalid=True):
    """
    Normalize depth map to [0,1] for visualization.

    Args:
        depth: float32 depth map in meters
        min_depth: lower bound for normalization (optional)
        max_depth: upper bound for normalization (optional)
        clip: whether to clamp values to [min_depth, max_depth]
        ignore_invalid: ignore 0 values when computing min/max

    Returns:
        depth_norm: float32 in [0,1]
    """

    depth = depth.copy()

    # Mask invalid pixels (NYU invalid depth = 0)
    if ignore_invalid:
        valid_mask = depth > 0
    else:
        valid_mask = np.ones_like(depth, dtype=bool)

    if min_depth is None:
        min_depth = depth[valid_mask].min()

    if max_depth is None:
        max_depth = depth[valid_mask].max()

    if clip:
        depth = np.clip(depth, min_depth, max_depth)

    depth_norm = (depth - min_depth) / (max_depth - min_depth + 1e-8)

    # Set invalid pixels to 0
    depth_norm[~valid_mask] = 0.0

    return depth_norm


# def torch_normalize_depth(depth_tensor, min_depth=0.1, max_depth=10.0):
#     depth = depth_tensor.clone()
#     depth = torch.clamp(depth, min_depth, max_depth)
#     depth = (depth - min_depth) / (max_depth - min_depth)
#     return depth


def visualize_depth(depth_norm, cmap="plasma", colorbar=True):
    """
    Display normalized depth map.
    """
    plt.imshow(depth_norm, cmap=cmap)
    if colorbar:
        plt.colorbar()
    plt.axis("off")
    plt.show()


def depth_to_colormap(depth_norm, cmap="plasma"):
    """
    Convert normalized depth to colored RGB image.
    Useful for saving to disk.
    """
    colormap = plt.get_cmap(cmap)
    colored = colormap(depth_norm)[:, :, :3]  # drop alpha
    return (colored * 255).astype(np.uint8)


