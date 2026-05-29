# -*- coding: utf-8 -*-
"""
cp_guard.py

CP-Guard defense for intermediate-fusion collaborative perception.

Reference:
    "CP-Guard: Malicious Agent Detection and Defense in Collaborative
     Bird's Eye View Perception", AAAI 2025.

This module implements two components:
  1. compute_ccloss  — Collaborative Consistency Loss (CCLoss), Eq. (4).
  2. pasac           — Probability-Agnostic Sample Consensus (PASAC),
                       Algorithm 1.

Adaptation notes
----------------
The original paper targets BEV segmentation (dense per-pixel class
probabilities).  In V2Xverse the perception model (V2X-ViT /
PointPillarMulticlass) performs 3-class object detection and outputs
raw logit heatmaps via cls_head.  We apply sigmoid to those logits to
obtain per-pixel probabilities, which play the same role as segmentation
probabilities in Eq. (4).

Algorithm 1 pseudocode note
----------------------------
Algorithm 1 in the paper contains a sign inversion in the single-CAV
base case (line 8 uses <= instead of >).  The paper's own text states
"if CCLoss exceeds the threshold the collaborator is benign", and the
ablation table (Table 2) confirms:

  * Small eps  -> almost nothing accepted   -> ego-solo performance
  * Large eps  -> everything accepted       -> defence degraded

Both observations are consistent only with "CCLoss > eps -> benign".
This implementation therefore uses the semantically correct condition:
    CCLoss > eps  ->  accept as benign
    CCLoss <= eps ->  suspect / recurse

for *both* single-CAV and group levels.  The group-level condition
matches the pseudocode (>= eps to add-all, < eps to recurse) so only
the single-CAV sign is corrected.
"""

import torch


# ---------------------------------------------------------------------------
# CCLoss
# ---------------------------------------------------------------------------

def compute_ccloss(cls_preds_0: torch.Tensor,
                   cls_preds_fuse: torch.Tensor) -> float:
    """
    Collaborative Consistency Loss between two BEV detection heatmaps.

    Implements Eq. (4) of CP-Guard.  A higher value means the two maps
    are more similar (better consensus), a lower value means they diverge.

    Parameters
    ----------
    cls_preds_0 : torch.Tensor
        Shape [1, C, H, W].  Ego-solo detection heatmap, **after sigmoid**.
    cls_preds_fuse : torch.Tensor
        Shape [1, C, H, W].  Collaborative detection heatmap, **after sigmoid**.

    Returns
    -------
    float
        CCLoss value in [0, ~0.5].  Higher -> more similar -> more benign.
    """
    _EPS = 1e-8

    p0 = cls_preds_0.squeeze(0)    # [C, H, W]
    pf = cls_preds_fuse.squeeze(0) # [C, H, W]
    C = p0.shape[0]

    # Per-class total probability mass  (sum over all spatial positions)
    P0 = p0.reshape(C, -1).sum(dim=1)   # [C]
    Pf = pf.reshape(C, -1).sum(dim=1)   # [C]

    # Per-class inverse-square-frequency weight  w_j = 1 / (P0_j + Pf_j)^2
    weights = 1.0 / ((P0 + Pf) ** 2 + _EPS)  # [C]

    # Per-class dot product  (numerator of per-class Dice)
    dot = (p0 * pf).reshape(C, -1).sum(dim=1)  # [C]

    # Eq. (4)
    numerator   = (weights * dot).sum()
    denominator = (weights * (P0 + Pf)).sum() + _EPS

    return (numerator / denominator).item()


# ---------------------------------------------------------------------------
# PASAC
# ---------------------------------------------------------------------------

def pasac(collab_indices: list,
          Y_0: torch.Tensor,
          fuse_decode_fn,
          n_upper: int = 5,
          eps: float = 0.08) -> list:
    """
    Probability-Agnostic Sample Consensus (PASAC), Algorithm 1 of CP-Guard.

    Recursively splits the collaborator list into two halves and verifies
    consensus with the ego's solo prediction Y_0 via CCLoss.  Groups whose
    CCLoss with Y_0 exceeds `eps` are accepted as benign; groups below the
    threshold are recursed into to locate the malicious CAV(s).

    Parameters
    ----------
    collab_indices : list[int]
        Indices of collaborator CAVs in the encoded feature tensor.
        Index 0 is always the ego; collaborators start at index 1.
        E.g. for 4 collaborators: [1, 2, 3, 4].
    Y_0 : torch.Tensor
        Shape [1, C, H, W].  Ego-solo detection heatmap (sigmoid of cls_preds),
        computed by calling cls_head directly on ego's features (no fusion).
    fuse_decode_fn : callable
        Signature:  fuse_decode_fn(indices: list[int]) -> torch.Tensor [1,C,H,W]
        Fuses pre-encoded features for the given CAV indices (must include 0
        for ego) and returns the sigmoid cls_preds.
    n_upper : int
        Maximum number of benign collaborators to return.  PASAC stops early
        once this count is reached.  Default: 5.
    eps : float
        CCLoss threshold.  CCLoss > eps -> benign; CCLoss <= eps -> suspect.
        Default: 0.08  (optimal per CP-Guard Table 2).

    Returns
    -------
    list[int]
        Collaborator indices (not including ego index 0) identified as benign.
        May be empty if all collaborators are malicious or none are present.
    """
    benign: list = []
    if not collab_indices:
        return benign
    _pasac_recursive(collab_indices, Y_0, fuse_decode_fn, benign, n_upper, eps)
    return benign


def _pasac_recursive(collab_indices: list,
                     Y_0: torch.Tensor,
                     fuse_decode_fn,
                     benign: list,
                     n_upper: int,
                     eps: float) -> None:
    """Internal recursive worker for PASAC (modifies `benign` in-place)."""

    # Early-exit: enough benign collaborators collected
    if len(benign) >= n_upper:
        return

    if len(collab_indices) == 0:
        return

    # ------------------------------------------------------------------
    # Base case: single remaining collaborator
    # ------------------------------------------------------------------
    if len(collab_indices) == 1:
        k = collab_indices[0]
        Y_k = fuse_decode_fn([0, k])
        loss = compute_ccloss(Y_0, Y_k)
        print(f'  [CCLoss] single CAV {k}: {loss:.6f}  (threshold eps={eps}  accepted={loss > eps})')
        # NOTE: Algorithm 1 line 8 has "<= eps" but the paper text and ablation
        # table both confirm the correct condition is "> eps" for benign.
        if loss > eps:
            benign.append(k)
        return

    # ------------------------------------------------------------------
    # Recursive case: split into two halves, verify each group
    # ------------------------------------------------------------------
    mid = len(collab_indices) // 2
    G1 = collab_indices[:mid]
    G2 = collab_indices[mid:]

    # --- Group 1 ---
    Y_G1 = fuse_decode_fn([0] + G1)
    loss_G1 = compute_ccloss(Y_0, Y_G1)

    if loss_G1 <= eps:
        # Group 1 is suspect: recurse to isolate malicious agent(s)
        _pasac_recursive(G1, Y_0, fuse_decode_fn, benign, n_upper, eps)
    else:
        # Group 1 is clearly benign: accept all members
        remaining = n_upper - len(benign)
        benign.extend(G1[:remaining])

    if len(benign) >= n_upper:
        return

    # --- Group 2 ---
    Y_G2 = fuse_decode_fn([0] + G2)
    loss_G2 = compute_ccloss(Y_0, Y_G2)

    if loss_G2 <= eps:
        _pasac_recursive(G2, Y_0, fuse_decode_fn, benign, n_upper, eps)
    else:
        remaining = n_upper - len(benign)
        benign.extend(G2[:remaining])
