from src.utils.depth_vis import (
    load_depth_png,
    normalize_depth,
    visualize_depth
)

import numpy as np

depth = load_depth_png("/Users/andrewpan/Documents/datasets/nyu_depth_v2/official_splits/test/living_room/sync_depth_00152.png")

print("min/max:", depth.min(), depth.max())

valid = depth[depth > 0]
p2, p98 = np.percentile(valid, [2, 98])

depth_norm = normalize_depth(depth,
                             min_depth=p2,
                             max_depth=p98)

visualize_depth(depth_norm)