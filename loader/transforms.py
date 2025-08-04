import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from .utils import compute_angle, compute_cross, compute_dihedral_angle
import math
from scipy.spatial.distance import cdist


class AddNodeFeature(object):

    def __init__(self, features=["atomic_numbers"]):
        self.features = features

    def __call__(self, crystal):
        structure = crystal["structure"]
        for feat in self.features:
            assert hasattr(structure, feat)
            crystal["graph"].ndata[feat] = torch.tensor((getattr(structure, feat)), dtype=(torch.long))
        else:
            return crystal


class AddEdgeFeature(object):

    def __init__(self, features=["distance"]):
        self.features = features

    def __call__(self, crystal):
        for feat in self.features:
            if feat == "offset":
                pass
            else:
                if feat == "distance":
                    crystal["graph"].edata[feat] = torch.norm((crystal["graph"].edata.pop("offset")), dim=1).float()
                    crystal["line_graph"].ndata["distance"] = crystal["graph"].edata["distance"]
                return crystal


class AddAngleFeature(object):

    def __init__(self, features=["angle", "cross"]):
        self.features = features

    def __call__(self, crystal):
        crystal["line_graph"].apply_edges(compute_angle)
        crystal["line_graph"].apply_edges(compute_cross)

        for feat in self.features:
            if feat == "angle":
                crystal["dihedral_graph"].ndata[feat] = crystal["line_graph"].edata[feat]
            elif feat == "cross":
                crystal["dihedral_graph"].ndata[feat] = crystal["line_graph"].edata[feat]
                crystal["dihedral_graph"].apply_edges(compute_dihedral_angle)
                crystal["line_graph"].edata.pop(feat)
                crystal["dihedral_graph"].ndata.pop(feat)

        crystal["line_graph"].ndata.pop("offset")
        return crystal


class GraphCollecting(object):

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __call__(self, crystal):
        inputs = (
         crystal["graph"], crystal["line_graph"], crystal["dihedral_graph"])
        targets = []
        for i in self.targets:
            i = "gap pbe" if i == "gap_pbe" else i
            if i in crystal:
                targets.append(torch.tensor([crystal[i]], dtype=(torch.float32)))
            elif not i in crystal["info"]:
                raise AssertionError
            targets.append(torch.tensor([crystal["info"][i]], dtype=(torch.float32)))
        else:
            return (
             inputs, torch.cat(targets))


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        else:
            return x

