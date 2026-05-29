# -*- coding: utf-8 -*-
"""
inference_multiclass_attack_robosac.py

Evaluate CoDriving under adversarial feature-space attack, with and without
the ROBOSAC consensus defence.

Attack model (AmongUs ICCV 2023, §4)
--------------------------------------
One non-ego CAV is designated as the attacker.  It perturbs its own scatter
features (pre-backbone BEV representation) to maximise a detection loss before
transmitting them.  The ego and the remaining collaborators are always benign.

Attack types:
  random  — bounded Gaussian noise (sanity baseline)
  fgsm    — single-step gradient sign
  pgd     — K-step projected gradient descent  ← paper-aligned default

Attack goals:
  fp  — False Positive injection: maximise detection confidence everywhere
  fn  — False Negative suppression: suppress scores where ego already detects

Pipeline per frame
------------------
  1. Encode all CAVs (once, shared across both conditions).
  2. Select the attacker CAV.
  3. Perturb attacker's features.
  4. Condition A — No defence:   fuse ego + ALL CAVs (with perturbed attacker).
  5. Condition B — ROBOSAC def.: run ROBOSAC with perturbed attacker present.
  6. Evaluate AP and record per-frame defence outcome.

Output files (in model_dir)
---------------------------
  AP_attack_{tag}.txt              — AP summary for both conditions
  frame_log_attack_{tag}.json      — per-frame log for post-hoc analysis

Usage
-----
python opencood/tools/inference_multiclass_attack_robosac.py \\
    --model_dir v2xverse/codriving/perception \\
    --attack_type pgd --attack_goal fp \\
    --attack_eps 0.2 --pgd_steps 10 \\
    --attacker_idx -1 \\
    --eta 0.2 --N_budget 7 --eps 0.3

Extra flags (beyond inference_multiclass_robosac.py)
----------------------------------------------------
  --attack_type   random | fgsm | pgd  (default: pgd)
  --attack_goal   fp | fn              (default: fp)
  --attack_eps    L-inf budget as fraction of mean|feature| (default: 0.2)
  --pgd_steps     number of PGD steps  (default: 10)
  --attacker_idx  which CAV to corrupt; -1 = random per frame (default: -1)
"""

import argparse
import json
import os
import random as py_random

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.attack.feature_attack import perturb_attacker_features
from opencood.defense.robosac import run_robosac, compute_s, _decode_to_boxes, _box_iou_distance

torch.multiprocessing.set_sharing_strategy('file_system')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_parser():
    parser = argparse.ArgumentParser(
        description='CoDriving eval: adversarial feature attack + ROBOSAC defence'
    )
    # --- standard args ---
    parser.add_argument('--model_dir', type=str, required=False,
                        default='v2xverse/codriving/perception')
    parser.add_argument('--save_vis_interval', type=int, default=9999,
                        help='BEV visualisation interval (large = skip)')
    parser.add_argument('--note', default='', type=str)

    # --- attack args ---
    parser.add_argument('--attack_type', default='pgd',
                        choices=['random', 'fgsm', 'pgd'])
    parser.add_argument('--attack_goal', default='fp',
                        choices=['fp', 'fn'])
    parser.add_argument('--attack_eps', type=float, default=0.2,
                        help='Perturbation budget as fraction of mean|feature|')
    parser.add_argument('--pgd_steps', type=int, default=10)
    parser.add_argument('--attacker_idx', type=int, default=-1,
                        help='CAV index to attack; -1 = random non-ego CAV')

    # --- ROBOSAC defence args ---
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--N_budget', type=int, default=7)
    parser.add_argument('--eps', type=float, default=0.3,
                        help='ROBOSAC consensus threshold')
    parser.add_argument('--p', type=float, default=0.99)
    parser.add_argument('--score_thresh', type=float, default=0.2)
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='Stop after this many frames (-1 = all)')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_attacker(collab_indices: list, fixed_idx: int) -> int:
    """Return the attacker CAV index (into the full frame, not collab list)."""
    if fixed_idx >= 0:
        if fixed_idx in collab_indices:
            return fixed_idx
        # Fall back gracefully
        return collab_indices[0]
    return py_random.choice(collab_indices)


def _fuse_decode_all(model, perturbed_raw, psm_single, t_matrix, rl, all_indices):
    """Fuse ego + all collaborators (including attacker) — the no-defence path."""
    with torch.no_grad():
        cls, bbox, fused = model.fuse_decode_subset(
            perturbed_raw, psm_single, t_matrix, rl, all_indices)
    return cls, bbox, fused


def _build_output_dict(model, cls, bbox, fused):
    """Build the output_dict expected by dataset.post_process_multiclass."""
    import torch.nn.functional as F
    from collections import OrderedDict

    if fused.size(2) == 48:
        scaled = F.interpolate(fused, scale_factor=2, mode='nearest')
    else:
        scaled = fused

    box_preds = bbox.permute(0, 2, 3, 1).contiguous()
    B, H, W, _ = box_preds.shape
    num_class = box_preds.shape[3] // 8
    box_preds = box_preds.view(B, H, W, num_class, 8)

    bbox_temp = []
    for c in range(num_class):
        bpc = box_preds[:, :, :, c, :].permute(0, 3, 1, 2)
        _, bt = model.generate_predicted_boxes(cls[:, c, :, :], bpc)
        bbox_temp.append(bt)
    reg_preds = torch.stack(bbox_temp, dim=1)

    od = OrderedDict()
    od['ego'] = {
        'cls_preds':            cls,
        'bbox_preds':           bbox,
        'reg_preds_multiclass': reg_preds,
        'fused_feature':        scaled,
    }
    return od


def _init_result_stat():
    stat = {}
    for c in [0, 1, 3]:
        stat[c] = {iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
                   for iou in [0.3, 0.5, 0.7]}
    return stat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    opt = test_parser()

    hypes = yaml_utils.load_yaml(None, opt)
    hypes['validate_dir'] = hypes['test_dir']

    print('Creating model …')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resume_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
    model.to(device).eval()
    print(f'Resumed epoch {resume_epoch}')

    tag = (f'_epoch{resume_epoch}'
           f'_{opt.attack_type}_{opt.attack_goal}'
           f'_eps{opt.attack_eps}'
           f'_eta{opt.eta}_N{opt.N_budget}_robosac{opt.eps}')
    tag += opt.note

    np.random.seed(30330)
    torch.manual_seed(10000)
    py_random.seed(42)

    print('Building dataset …')
    ds = build_dataset(hypes, visualize=True, train=False)

    if opt.max_frames > 0:
        import random as _random
        _rng = _random.Random(42)
        _indices = list(range(len(ds)))
        _rng.shuffle(_indices)
        _indices = _indices[:opt.max_frames]
        from torch.utils.data import Subset
        _ds = Subset(ds, _indices)
        _collate = ds.collate_batch_test
    else:
        _ds = ds
        _collate = ds.collate_batch_test

    loader = DataLoader(
        _ds, batch_size=1, num_workers=4,
        collate_fn=_collate,
        shuffle=False, pin_memory=False, drop_last=False)

    # Two AP accumulators: no-defence (attacked) and ROBOSAC-defended
    stat_nodef   = _init_result_stat()
    stat_robosac = _init_result_stat()

    frame_logs = {}  # per-frame log for post-hoc analysis

    log_path = os.path.join(opt.model_dir, f'frame_log_attack{tag}.json')

    for i, batch_data in enumerate(loader):
        print(f'frame {i}', flush=True)
        if batch_data is None:
            continue

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

        cav = batch_data['ego']

        # ---------------------------------------------------------------
        # Step 1: encode all CAVs
        # ---------------------------------------------------------------
        with torch.no_grad():
            raw_features, psm_single, t_matrix, record_len = \
                model.encode(cav)

        N_cavs = int(record_len[0].item())
        all_collab = list(range(1, N_cavs))

        if not all_collab:
            # Ego-only frame — nothing to attack, skip logging
            continue

        # ---------------------------------------------------------------
        # Step 2: select attacker and apply perturbation
        # ---------------------------------------------------------------
        atk_idx = _select_attacker(all_collab, opt.attacker_idx)

        perturbed_raw = perturb_attacker_features(
            model,
            raw_features,
            psm_single,
            t_matrix,
            record_len,
            attacker_idx=atk_idx,
            attack_type=opt.attack_type,
            attack_goal=opt.attack_goal,
            eps=opt.attack_eps,
            pgd_steps=opt.pgd_steps,
        )

        # ---------------------------------------------------------------
        # Step 3a: No-defence — accept all collaborators (attacker included)
        # ---------------------------------------------------------------
        with torch.no_grad():
            cls_nd, bbox_nd, fused_nd = _fuse_decode_all(
                model, perturbed_raw, psm_single,
                t_matrix, record_len, [0] + all_collab)
        od_nd = _build_output_dict(model, cls_nd, bbox_nd, fused_nd)
        pred_nd, score_nd, gt_nd = ds.post_process_multiclass(
            {'ego': batch_data['ego']}, od_nd, False)

        # ---------------------------------------------------------------
        # Step 3b: ROBOSAC defence
        # ---------------------------------------------------------------
        with torch.no_grad():
            # Y_0 — ego only (clean, not perturbed)
            cls0, bbox0, _ = model.fuse_decode_subset(
                raw_features, psm_single, t_matrix, record_len, [0])

        Y0_boxes = _decode_to_boxes(model, cls0, bbox0,
                                    opt.score_thresh, 0.15, 'hwl')

        s = compute_s(opt.eta, opt.N_budget, opt.p)
        s = min(s, len(all_collab))

        accepted_set = []
        reached_consensus = False
        n_trials = 0
        best_d = 1.0
        trials_log = []

        for trial in range(opt.N_budget):
            n_trials += 1
            sample = py_random.sample(all_collab, s)

            with torch.no_grad():
                cls_s, bbox_s, _ = model.fuse_decode_subset(
                    perturbed_raw, psm_single, t_matrix, record_len,
                    [0] + sample)
            Ys_boxes = _decode_to_boxes(model, cls_s, bbox_s,
                                        opt.score_thresh, 0.15, 'hwl')
            d = _box_iou_distance(Y0_boxes, Ys_boxes)

            trials_log.append({'sample': list(sample),
                                'd': float(d),
                                'consensus': d <= opt.eps})
            if d < best_d:
                best_d = d

            if d <= opt.eps:
                accepted_set = list(sample)
                reached_consensus = True
                break

        # Final ROBOSAC prediction
        final_indices = [0] + accepted_set  # ego-only if no consensus
        with torch.no_grad():
            cls_r, bbox_r, fused_r = model.fuse_decode_subset(
                perturbed_raw, psm_single, t_matrix, record_len, final_indices)
        od_r = _build_output_dict(model, cls_r, bbox_r, fused_r)
        pred_r, score_r, gt_r = ds.post_process_multiclass(
            {'ego': batch_data['ego']}, od_r, False)

        # ---------------------------------------------------------------
        # Step 4: Accumulate AP
        # ---------------------------------------------------------------
        for iou_thresh in [0.3, 0.5, 0.7]:
            eval_utils.caluclate_tp_fp_multiclass(
                pred_nd, score_nd, gt_nd, stat_nodef, iou_thresh)
            eval_utils.caluclate_tp_fp_multiclass(
                pred_r, score_r, gt_r, stat_robosac, iou_thresh)

        # ---------------------------------------------------------------
        # Step 5: Per-frame log
        # ---------------------------------------------------------------
        attacker_rejected = (atk_idx not in accepted_set)
        frame_logs[i] = {
            'n_cavs':              N_cavs,
            'attacker_idx':        atk_idx,
            'collab_indices':      all_collab,
            'accepted_set':        list(accepted_set),
            'attacker_rejected':   attacker_rejected,
            'reached_consensus':   reached_consensus,
            'n_trials':            n_trials,
            'best_d':              float(best_d),
            'Y0_n_boxes':          0 if Y0_boxes is None else Y0_boxes.shape[0],
            'trials':              trials_log,
        }

        print(f'  [ATK] CAVs={N_cavs}  attacker={atk_idx}  '
              f'accepted={accepted_set}  rejected={attacker_rejected}  '
              f'best_d={best_d:.3f}  consensus={reached_consensus}',
              flush=True)

        # Incremental save every 200 frames so partial results survive interruption
        if i % 200 == 0 and frame_logs:
            with open(log_path, 'w') as _f:
                json.dump(frame_logs, _f)

        torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Final AP
    # -------------------------------------------------------------------
    print('\n=== No-Defence (attacked) ===')
    res_nd, _, _, _ = eval_utils.eval_final_results_multiclass(
        stat_nodef, opt.model_dir,
        'attack_nodef' + tag)

    print('\n=== ROBOSAC Defence (attacked) ===')
    res_r, _, _, _ = eval_utils.eval_final_results_multiclass(
        stat_robosac, opt.model_dir,
        'attack_robosac' + tag)

    # -------------------------------------------------------------------
    # Aggregate defence statistics
    # -------------------------------------------------------------------
    n_total    = len(frame_logs)
    n_rejected = sum(1 for v in frame_logs.values() if v['attacker_rejected'])
    n_accepted = n_total - n_rejected   # ROBOSAC accepted (failed to filter)
    n_ego_fallback = sum(1 for v in frame_logs.values()
                         if not v['reached_consensus'])

    trr = n_rejected / max(n_total, 1)
    far = n_accepted / max(n_total, 1)

    # AP comparison table
    ap_nd_v = res_nd[0]['ap50']
    ap_nd_p = res_nd[1]['ap50']
    ap_nd_b = res_nd[3]['ap50']
    ap_r_v  = res_r[0]['ap50']
    ap_r_p  = res_r[1]['ap50']
    ap_r_b  = res_r[3]['ap50']

    summary = (
        f'\n========================================\n'
        f' Attack: {opt.attack_type.upper()}-{opt.attack_goal.upper()}  '
        f'eps={opt.attack_eps}  steps={opt.pgd_steps}\n'
        f' ROBOSAC: eta={opt.eta}  N={opt.N_budget}  eps_cons={opt.eps}\n'
        f'========================================\n'
        f' Frames evaluated : {n_total}\n'
        f' Attacker rejected (TRR) : {n_rejected}/{n_total} = {100*trr:.1f}%\n'
        f' Attacker accepted (FAR) : {n_accepted}/{n_total} = {100*far:.1f}%\n'
        f' Ego-only fallback       : {n_ego_fallback}/{n_total} '
        f'= {100*n_ego_fallback/max(n_total,1):.1f}%\n'
        f'\n AP@0.5 comparison:\n'
        f'   Class       No-def(atk)  ROBOSAC(def)  Recovery\n'
        f'   Vehicle(0)  {ap_nd_v:.3f}        {ap_r_v:.3f}        '
        f'{ap_r_v - ap_nd_v:+.3f}\n'
        f'   Pedestr(1)  {ap_nd_p:.3f}        {ap_r_p:.3f}        '
        f'{ap_r_p - ap_nd_p:+.3f}\n'
        f'   Bicycle(3)  {ap_nd_b:.3f}        {ap_r_b:.3f}        '
        f'{ap_r_b - ap_nd_b:+.3f}\n'
        f'========================================\n'
    )
    print(summary)

    ap_txt = os.path.join(opt.model_dir, f'AP_attack{tag}.txt')
    with open(ap_txt, 'w') as f:
        f.write(summary)
    print(f'AP summary written to {ap_txt}')

    with open(log_path, 'w') as f:
        json.dump(frame_logs, f, indent=2)
    print(f'Per-frame log saved to {log_path}')


if __name__ == '__main__':
    main()
