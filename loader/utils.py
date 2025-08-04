import torch

def compute_angle(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    r1 = -edges.src["offset"]
    r2 = edges.dst["offset"]
    bond_cosine = torch.sum((r1 * r2), dim=1) / (torch.norm(r1, dim=1) * torch.norm(r2, dim=1) + 1e-06)
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return {"angle": (bond_cosine.float())}


def compute_cross(edges):
    r1 = edges.src["offset"]
    r2 = edges.dst["offset"]

    cross = torch.cross(r1, r2, dim=1)

    return {"cross": (cross.float())}


def compute_dihedral_angle(edges):
    n1 = edges.src["cross"]
    n2 = edges.dst["cross"]

    n1_norm = torch.norm(n1, dim=1)
    n2_norm = torch.norm(n2, dim=1)

    n1 = n1 / (n1_norm.unsqueeze(1) + 1e-06)
    n2 = n2 / (n2_norm.unsqueeze(1) + 1e-06)

    cos_theta = torch.sum((n1 * n2), dim=1)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    return {"dihedral_angle": (torch.rad2deg(theta).float())}

