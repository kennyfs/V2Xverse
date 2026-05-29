# -*- coding: utf-8 -*-
"""
robosac.py

ROBOSAC: ROBust cOllaborative perception via SAmple Consensus.

Implements Algorithm 1 from:
    "Among Us: Adversarially Robust Collaborative Perception by Consensus"
    ICCV 2023.

The key insight being tested here is the FALSE POSITIVE vulnerability:
    A benign collaborator with unique, non-overlapping observations (e.g.,
    detecting a pedestrian in the ego's blind spot) will cause the
    collaborative output Y_s to diverge from the ego-only baseline Y_0,
    since d(Y_s, Y_0) >> eps.  ROBOSAC never reaches consensus with that
    CAV in the sample, exhausts its budget N, and falls back to ego-only,
    discarding the critical information entirely.

Usage (from inference_utils.py):
    from opencood.defense.robosac import compute_s, run_robosac

The model must expose  encode()  and  fuse_decode_subset()  (added to
center_point_codriving.py).

Difference measure d
--------------------
Box-level IoU + Hungarian matching, exactly as in §4 of the paper.
Given two box sets A and B (each [N,7] in ego-lidar frame, order hwl):

  * Both empty             → d = 0.0  (trivial consensus)
  * One empty, one not     → d = 1.0  (critical case: new detection added)
  * Both non-empty         → Hungarian match on BEV IoU;
                             d = 1 - sum_matched_IoU / max(|A|, |B|)

Consensus threshold eps is applied as  d <= eps → consensus.
Paper uses eps = 0.3 by default.
"""

import math
import random
from collections import OrderedDict

import numpy as np
import torch

from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from opencood.utils import box_utils
from opencood.utils import common_utils


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_s(eta: float, N_budget: int, p: float = 0.99) -> int:
    """
    Eq. 1 from the paper: guaranteed attacker-free collaborators per sample.

        s = floor( ln[1 - (1-p)^(1/N)] / ln(1 - eta) )

    Parameters
    ----------
    eta       : attacker ratio in (0, 1).
    N_budget  : sampling budget N.
    p         : desired probability of at least one successful sampling.

    Returns
    -------
    int >= 1
    """
    if eta <= 0.0 or eta >= 1.0:
        raise ValueError(f"eta must be in (0,1), got {eta}")
    if N_budget < 1:
        raise ValueError(f"N_budget must be >= 1, got {N_budget}")

    numerator   = math.log(1.0 - (1.0 - p) ** (1.0 / N_budget))
    denominator = math.log(1.0 - eta)
    s = int(math.floor(numerator / denominator))
    return max(s, 1)


def run_robosac(
    collab_indices,
    model,
    raw_features,
    psm_single,
    pairwise_t_matrix,
    record_len,
    N_budget: int = 7,
    eta: float = 0.2,
    eps: float = 0.3,
    p: float = 0.99,
    score_thresh: float = 0.2,
    nms_thresh: float = 0.15,
    box_order: str = 'hwl',
):
    """
    Algorithm 1 from the AmongUs paper.

    Parameters
    ----------
    collab_indices : list[int]
        Indices of collaborator CAVs (> 0).  Index 0 is always ego.
    model : centerpointcodriving
        The loaded CoDriving model (must have fuse_decode_subset()).
    raw_features, psm_single, pairwise_t_matrix, record_len :
        Outputs of model.encode().
    N_budget : int
        Maximum sampling attempts (N in the paper).
    eta : float
        Assumed attacker ratio (used to compute s).
    eps : float
        Consensus threshold; d <= eps triggers early stop.
    p : float
        Desired success probability for computing s.
    score_thresh : float
        Score threshold for box decoding (applied before NMS).
    nms_thresh : float
        NMS IoU threshold.
    box_order : str
        Box encoding order passed to boxes_to_corners_3d ('hwl' for CoDriving).

    Returns
    -------
    accepted_indices : list[int]
        Collaborator indices in the accepted sample, or [] on failure.
    n_trials : int
        How many sampling attempts were made before stopping.
    reached_consensus : bool
        True if a consensus sample was found within the budget.
    log : dict
        Per-trial log for FPR analysis:
            trials : list of {sample, d, consensus}
            Y0_n_boxes : int  (boxes in ego-only prediction)
    """
    device = raw_features.device

    # --- Step 1: compute s (subset size) and ego-only Y_0 boxes -----------
    s = compute_s(eta, N_budget, p)
    s = min(s, len(collab_indices))  # cannot sample more than available

    with torch.no_grad():
        cls0, bbox0, _ = model.fuse_decode_subset(
            raw_features, psm_single, pairwise_t_matrix, record_len, [0]
        )
    Y0_boxes = _decode_to_boxes(model, cls0, bbox0, score_thresh, nms_thresh,
                                box_order)

    log = {
        'trials': [],
        'Y0_n_boxes': 0 if Y0_boxes is None else Y0_boxes.shape[0],
        's': s,
        'N_budget': N_budget,
        'eta': eta,
        'eps': eps,
    }

    # --- Step 2: ROBOSAC loop (Algorithm 1) --------------------------------
    n = 0
    while n < N_budget:
        n += 1

        sample = random.sample(collab_indices, s)

        with torch.no_grad():
            cls_s, bbox_s, _ = model.fuse_decode_subset(
                raw_features, psm_single, pairwise_t_matrix, record_len,
                [0] + sample
            )
        Ys_boxes = _decode_to_boxes(model, cls_s, bbox_s, score_thresh,
                                    nms_thresh, box_order)

        d = _box_iou_distance(Y0_boxes, Ys_boxes)

        log['trials'].append({
            'sample': list(sample),
            'd': float(d),
            'consensus': d <= eps,
            'Ys_n_boxes': 0 if Ys_boxes is None else Ys_boxes.shape[0],
        })

        if d <= eps:
            return sample, n, True, log

    # Budget exhausted — ego-only fallback
    return [], N_budget, False, log


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_to_boxes(model, cls_logits, bbox_logits, score_thresh, nms_thresh,
                     box_order):
    """
    Decode (cls, bbox) tensors from fuse_decode_subset to a set of 3D boxes
    after score thresholding and per-class NMS.

    Parameters
    ----------
    model       : centerpointcodriving  (needs generate_predicted_boxes)
    cls_logits  : [1, num_class, H', W']  raw logits
    bbox_logits : [1, num_class*8, H', W']  raw regression output
    score_thresh : float
    nms_thresh   : float
    box_order    : str  ('hwl' for CoDriving CenterPoint output)

    Returns
    -------
    torch.Tensor [N_kept, 7] in ego-lidar frame, or None if no detections.
    """
    device   = cls_logits.device
    B, num_class, H, W = cls_logits.shape

    # bbox_logits is [1, num_class*8, H, W].
    # Rearrange to [B, H, W, num_class, 8] then extract per-class slices.
    bbox_hwc = bbox_logits.permute(0, 2, 3, 1).contiguous()          # [1, H, W, C*8]
    bbox_hwc = bbox_hwc.view(B, H, W, num_class, 8)                   # [1, H, W, nc, 8]

    all_boxes  = []
    all_scores = []

    for c in range(num_class):
        # Per-class box predictions
        bbox_c = bbox_hwc[:, :, :, c, :].permute(0, 3, 1, 2).contiguous()  # [1,8,H,W]
        _, batch_box_preds = model.generate_predicted_boxes(
            cls_logits[:, c, :, :], bbox_c
        )
        # batch_box_preds: [1, H*W, 7]  in [x,y,z,h,w,l,yaw] (hwl order)

        scores_c = torch.sigmoid(cls_logits[:, c, :, :]).reshape(-1)  # [H*W]
        boxes_c  = batch_box_preds[0]                                   # [H*W, 7]

        mask = scores_c > score_thresh
        boxes_c  = boxes_c[mask]    # [M, 7]
        scores_c = scores_c[mask]   # [M]

        if boxes_c.shape[0] == 0:
            continue

        # NMS on BEV footprint
        corners3d  = box_utils.boxes_to_corners_3d(boxes_c, order=box_order)  # [M,8,3]
        bev_corners = corners3d[:, :4, :2]                                      # [M,4,2]
        keep = box_utils.nms_rotated(bev_corners, scores_c, nms_thresh)

        if len(keep) == 0:
            continue
        all_boxes.append(boxes_c[keep])
        all_scores.append(scores_c[keep])

    if len(all_boxes) == 0:
        return None

    return torch.cat(all_boxes, dim=0)   # [N_kept, 7]


def _box_iou_distance(boxes_a, boxes_b, box_order: str = 'hwl') -> float:
    """
    Dissimilarity measure d(A, B) in [0, 1] based on IoU + Hungarian matching.

    d = 0   →  identical box sets (perfect consensus)
    d = 1   →  completely disjoint box sets (one sees a new object)

    Special cases:
      both None / empty  →  0.0
      one empty          →  1.0  (critical FDR case)
    """
    a_empty = (boxes_a is None) or (boxes_a.shape[0] == 0)
    b_empty = (boxes_b is None) or (boxes_b.shape[0] == 0)

    if a_empty and b_empty:
        return 0.0
    if a_empty or b_empty:
        return 1.0   # <-- this is the FDR-triggering case

    boxes_a_np = boxes_a.detach().cpu().numpy()   # [M, 7]
    boxes_b_np = boxes_b.detach().cpu().numpy()   # [N, 7]

    M = boxes_a_np.shape[0]
    N = boxes_b_np.shape[0]

    # BEV IoU matrix [M, N]
    iou_mat = _bev_iou_matrix(boxes_a_np, boxes_b_np, box_order)

    # Hungarian matching (maximise total IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_mat)
    matched_iou = iou_mat[row_ind, col_ind]

    # d = 1 - (sum matched IoU) / max(M, N)
    d = 1.0 - float(matched_iou.sum()) / max(M, N)
    return float(np.clip(d, 0.0, 1.0))


def _bev_iou_matrix(boxes_a_np: np.ndarray, boxes_b_np: np.ndarray,
                    box_order: str) -> np.ndarray:
    """
    Compute [M, N] BEV IoU matrix between two box arrays.
    Uses 2D polygon intersection via shapely (same as nms_rotated).
    """
    import torch as _torch

    def _to_bev_polygons(boxes_np):
        boxes_t = _torch.from_numpy(boxes_np)
        corners3d = box_utils.boxes_to_corners_3d(boxes_t, order=box_order)  # [K,8,3]
        bev = corners3d[:, :4, :2].cpu().numpy()                              # [K,4,2]
        return [Polygon([(bev[i, j, 0], bev[i, j, 1]) for j in range(4)])
                for i in range(bev.shape[0])]

    polys_a = _to_bev_polygons(boxes_a_np)
    polys_b = _to_bev_polygons(boxes_b_np)

    M, N = len(polys_a), len(polys_b)
    iou_mat = np.zeros((M, N), dtype=np.float32)
    for i, pa in enumerate(polys_a):
        iou_mat[i] = common_utils.compute_iou(pa, polys_b)

    return iou_mat
