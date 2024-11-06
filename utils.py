import torch


def transform_from_angles(angles):
    angles = torch.deg2rad(torch.tensor(angles))
    t = torch.eye(4)
    c1, s1 = torch.cos(angles[0]), torch.sin(angles[0])
    c2, s2 = torch.cos(angles[1]), torch.sin(angles[1])
    c3, s3 = torch.cos(angles[2]), torch.sin(angles[2])
    t[:3, :3] = torch.tensor(
        [
            [c2 * c3, s1 * s2 * c3 - c1 * s3, c1 * s2 * c3 + s1 * s3],
            [c2 * s3, s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3],
            [-s2, s1 * c2, c1 * c2],
        ]
    )
    return t

def disp_to_depth(disp, min_depth, max_depth, inverse=False):
    """Convert network's sigmoid output into depth prediction. The formula for this conversion is given in the 'additional considerations' section of the paper.
    """
    if inverse:
        scaled_disp, depth = disp, disp * 80.0 / 5.4
    else:
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
    return scaled_disp, depth
