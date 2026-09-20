"""One observation contract for offline training and live inference.
Only observed geometry/gain and path-verified connectivity are model inputs.
Expert routes, costs influenced by LKH, and outcomes are never input features.
"""

from dataclasses import dataclass
import numpy as np
import torch

FEATURE_VERSION = 2
FEATURE_CONFIG = {
    "position_scale_m": 40.0,
    "gain_scale_m3": 1000.0,
    "gain_definition": "observed_voxel_raycast_v1",
    "version": FEATURE_VERSION,
}


@dataclass
class Observation:
    nodes: np.ndarray
    edges: np.ndarray
    current: int
    candidates: np.ndarray
    reachable: np.ndarray


def observation(record, config=FEATURE_CONFIG):
    if record.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Learning requires feature_version=2; recollect old episodes")
    if record.get("gain_definition") != config["gain_definition"]:
        raise ValueError("Gain definition mismatch")
    nodes = record["nodes"]
    candidates = record["candidates"]
    n = len(nodes)
    v = len(candidates)
    if [x["id"] for x in nodes] != list(range(n)) or [
        x["id"] for x in candidates
    ] != list(range(v)):
        raise ValueError("Noncontiguous IDs")
    current = record["current_node_id"]
    if not 0 <= current < n or not nodes[current]["is_current"]:
        raise ValueError("Invalid current node")
    xyz = np.asarray([x["position"] for x in nodes], dtype=np.float32)
    gain = np.asarray(record["information_gain_m3"], dtype=np.float32)
    if len(gain) != v or not np.isfinite(gain).all() or (gain < 0).any():
        raise ValueError("Invalid gains")
    indices = np.asarray([x["node_id"] for x in candidates], dtype=np.int64)
    if len(set(indices)) != v or (indices < 0).any() or (indices >= n).any():
        raise ValueError("Invalid candidates")
    # xyz + unknown volume + agent/skeleton/frontier one-hot = 7 features.
    x = np.zeros((n, 7), dtype=np.float32)
    x[:, :3] = (xyz - np.asarray(record["position"])) / config["position_scale_m"]
    x[:, 5] = 1.0
    x[current, 4:] = [1, 0, 0]
    for i, c in enumerate(candidates):
        j = c["node_id"]
        x[j, 3] = gain[i] / config["gain_scale_m3"]
        x[j, 4:] = [0, 0, 1]
        if not np.allclose(xyz[j], c["position"], atol=1e-5):
            raise ValueError("Candidate mapping mismatch")
    edges = np.asarray([[e[0], e[1]] for e in record["edges"]], dtype=np.int64).reshape(
        -1, 2
    )
    if edges.size and ((edges < 0).any() or (edges >= n).any()):
        raise ValueError("Invalid graph edge")
    d = record["reachability_dimension"]
    if v and d != v + 1:
        raise ValueError("Invalid reachability dimensions")
    reach = np.asarray(record["reachable"], dtype=bool).reshape(d, d)
    if not v:
        reach = np.zeros((1, 1), dtype=bool)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features")
    return Observation(x, edges, current, indices, reach)


def collate(observations, device="cpu"):
    b = len(observations)
    n = max(len(o.nodes) for o in observations)
    v = max(1, max(len(o.candidates) for o in observations))
    nodes = torch.zeros(b, n, 7)
    valid = torch.zeros(b, n, dtype=torch.bool)
    candidates = torch.zeros(b, v, dtype=torch.long)
    candidate_valid = torch.zeros(b, v, dtype=torch.bool)
    reachable = torch.zeros(b, v + 1, v + 1, dtype=torch.bool)
    current = torch.empty(b, dtype=torch.long)
    edges = []
    for i, o in enumerate(observations):
        nn = len(o.nodes)
        vv = len(o.candidates)
        nodes[i, :nn] = torch.from_numpy(o.nodes)
        valid[i, :nn] = True
        current[i] = o.current
        candidates[i, :vv] = torch.from_numpy(o.candidates)
        candidate_valid[i, :vv] = True
        reachable[i, : vv + 1, : vv + 1] = torch.from_numpy(o.reachable)
        if o.edges.size:
            edges.append(torch.from_numpy(o.edges.T.copy()) + i * n)
    return {
        k: t.to(device)
        for k, t in dict(
            nodes=nodes,
            valid=valid,
            current=current,
            candidates=candidates,
            candidate_valid=candidate_valid,
            reachable=reachable,
            edges=torch.cat(edges, dim=1)
            if edges
            else torch.empty(2, 0, dtype=torch.long),
        ).items()
    }
