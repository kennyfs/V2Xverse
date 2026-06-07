# -*- coding: utf-8 -*-
"""
feature_attack.py

Adversarial feature-space attacks on collaborative perception for the
ROBOSAC false-positive-rate study.

Attack model
------------
In intermediate fusion each CAV transmits its backbone feature map.
Here we model this by perturbing the attacker's scatter features
(raw_features[attacker_idx]) — the pre-backbone BEV tensor that flows
through CoDriving's backbone + AttenFusion + heads.

The attacker is white-box: it has full knowledge of the model weights.

Attack types
------------
  random  : bounded Gaussian noise (baseline)
  fgsm    : single-step gradient sign (Goodfellow et al. 2014)
  pgd     : K-step projected gradient descent (Madry et al. 2018)
             as used in the AmongUs paper (Xie et al. ICCV 2023).

Attack goals
------------
  fp  : False Positive injection — maximise detection confidence across
        the entire BEV map to create phantom objects.
  fn  : False Negative suppression — minimise detection scores at
        locations where the ego already detects objects.

Usage
-----
    from opencood.attack.feature_attack import perturb_attacker_features
    perturbed = perturb_attacker_features(
        model, raw_features, psm_single, t_matrix, rl,
        attacker_idx=1, attack_type='pgd', attack_goal='fp',
        eps=0.2, pgd_steps=10)
"""

import torch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def perturb_attacker_features(
    model,
    raw_features,
    psm_single,
    pairwise_t_matrix,
    record_len,
    attacker_idx: int,
    attack_type: str = 'pgd',
    attack_goal: str = 'fp',
    eps: float = 0.2,
    pgd_steps: int = 10,
    pgd_alpha: float = None,
):
    """
    Return a copy of raw_features with the attacker's row adversarially
    perturbed.

    Parameters
    ----------
    model               : CenterPointCoDriving — must have fuse_decode_subset()
    raw_features        : [sum_N, 64, H, W]  output of model.encode()
    psm_single          : [sum_N, anchor, H', W']
    pairwise_t_matrix   : [1, max_cav, max_cav, 4, 4]
    record_len          : [1]
    attacker_idx        : int in [1, sum_N-1]  — which CAV is the attacker
    attack_type         : 'random' | 'fgsm' | 'pgd'
    attack_goal         : 'fp' | 'fn'
    eps                 : L-inf perturbation budget relative to feature mean-abs
                          (eps=0.2 means delta bounded by 0.2 * mean|f|)
    pgd_steps           : number of PGD iterations
    pgd_alpha           : PGD step size; default = 2.5 * eps_abs / pgd_steps
                          (standard setting from Madry et al.)

    Returns
    -------
    torch.Tensor : same shape as raw_features, dtype/device preserved,
                   only row [attacker_idx] is modified.
    """
    device = raw_features.device

    # Scale eps to absolute feature units using max of non-zero activations.
    # mean_abs is dominated by sparse zeros (~0.03), giving negligible eps_abs.
    feat = raw_features[attacker_idx]
    flat = feat.abs().flatten()
    nonzero = flat[flat > 0]
    if nonzero.numel() > 0:
        eps_abs = eps * torch.quantile(nonzero, 0.95).item()
    else:
        eps_abs = eps * flat.max().item()
    if pgd_alpha is None:
        pgd_alpha = 2.5 * eps_abs / max(pgd_steps, 1)

    if attack_type == 'random':
        return _random_attack(raw_features, attacker_idx, eps_abs, device)

    # Always compute ego heatmap: FP attack needs it to mask real detections,
    # FN attack needs it to know where to suppress.
    ego_heatmap = _get_ego_heatmap(model, raw_features, psm_single,
                                   pairwise_t_matrix, record_len)
    # Temporarily freeze model parameters so backward() only touches delta
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        if attack_type == 'fgsm':
            result = _fgsm(model, raw_features, psm_single,
                           pairwise_t_matrix, record_len,
                           attacker_idx, attack_goal, eps_abs, ego_heatmap)
        elif attack_type == 'pgd':
            result = _pgd(model, raw_features, psm_single,
                          pairwise_t_matrix, record_len,
                          attacker_idx, attack_goal, eps_abs,
                          pgd_steps, pgd_alpha, ego_heatmap)
        else:
            raise ValueError(f"Unknown attack_type '{attack_type}'. "
                             "Choose 'random', 'fgsm', or 'pgd'.")
    finally:
        # Always restore model gradients
        for p in model.parameters():
            p.requires_grad_(True)

    return result


# ---------------------------------------------------------------------------
# Individual attack implementations
# ---------------------------------------------------------------------------

def _random_attack(raw_features, attacker_idx, eps_abs, device):
    """Gaussian noise clipped to L-inf ball."""
    perturbed = raw_features.clone().detach()
    noise = torch.randn_like(perturbed[attacker_idx]) * eps_abs
    noise = noise.clamp(-eps_abs, eps_abs)
    perturbed[attacker_idx] = perturbed[attacker_idx] + noise
    return perturbed


def _fgsm(model, raw_features, psm_single, t_matrix, rl,
          attacker_idx, goal, eps_abs, ego_heatmap):
    """Single-step gradient sign attack."""
    with torch.enable_grad():
        delta = torch.zeros_like(raw_features[attacker_idx]).requires_grad_(True)
        perturbed_all = _build_perturbed(raw_features, attacker_idx, delta)
        psm_d = None if psm_single is None else psm_single.detach()
        cls_out, _, _ = model.fuse_decode_subset(
            perturbed_all, psm_d,
            t_matrix.detach(), rl.detach(),
            [0, attacker_idx])
        loss = _attack_loss(cls_out, goal, ego_heatmap)
        loss.backward()
        grad_sign = delta.grad.sign().detach()

    perturbed = raw_features.clone().detach()
    perturbed[attacker_idx] = raw_features[attacker_idx].detach() + eps_abs * grad_sign
    return perturbed


def _pgd(model, raw_features, psm_single, t_matrix, rl,
         attacker_idx, goal, eps_abs, pgd_steps, pgd_alpha, ego_heatmap):
    """K-step PGD (Madry et al. 2018)."""
    # Initialise delta at zero (paper default; could also start random)
    delta_data = torch.zeros_like(raw_features[attacker_idx]).detach()
    psm_d = None if psm_single is None else psm_single.detach()

    with torch.enable_grad():
        for _ in range(pgd_steps):
            delta = delta_data.clone().requires_grad_(True)
            perturbed_all = _build_perturbed(raw_features, attacker_idx, delta)
            cls_out, _, _ = model.fuse_decode_subset(
                perturbed_all, psm_d,
                t_matrix.detach(), rl.detach(),
                [0, attacker_idx])
            loss = _attack_loss(cls_out, goal, ego_heatmap)
            loss.backward()
            grad_sign = delta.grad.sign().detach()

            # Gradient ascent step + project onto L-inf ball
            delta_data = (delta_data + pgd_alpha * grad_sign).clamp(-eps_abs, eps_abs)

    perturbed = raw_features.clone().detach()
    perturbed[attacker_idx] = raw_features[attacker_idx].detach() + delta_data
    return perturbed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_perturbed(raw_features, attacker_idx, delta):
    """
    Construct a perturbed feature tensor with gradient only through delta.
    Uses vectorized broadcasting to avoid slow loops and in-place autograd issues.
    """
    N = raw_features.shape[0]
    mask = torch.zeros(N, 1, 1, 1, device=delta.device)
    mask[attacker_idx] = 1.0
    return raw_features.detach() + delta.unsqueeze(0) * mask


def _attack_loss(cls_out, goal, ego_heatmap):
    """
    Scalar loss to MAXIMISE (gradient ascent).

    fp  : maximise mean sigmoid score  → create phantom detections everywhere
    fn  : minimise mean(sigmoid * ego_heatmap)  → suppress ego-visible objects
    """
    scores = torch.sigmoid(cls_out)   # [1, num_class, H, W]
    if goal == 'fp':
        # Inject phantom detections only in locations ego sees as empty.
        # Targeting all locations would boost real detections and improve AP.
        empty_mask = (ego_heatmap < 0.15).float()
        return (scores * empty_mask).mean()
    elif goal == 'fn':
        # Suppress detections at ego-visible object locations.
        return -(scores * ego_heatmap).mean()
    else:
        raise ValueError(f"Unknown attack_goal '{goal}'. Choose 'fp' or 'fn'.")


def _get_ego_heatmap(model, raw_features, psm_single, t_matrix, rl):
    """
    Ego-only class heatmap — used as pseudo-GT for the FN attack.
    Returns [1, num_class, H, W] probability map (sigmoid-normalised).
    """
    with torch.no_grad():
        cls0, _, _ = model.fuse_decode_subset(
            raw_features, psm_single, t_matrix, rl, [0])
    return torch.sigmoid(cls0).detach()
