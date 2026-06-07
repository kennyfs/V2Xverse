# -*- coding: utf-8 -*-
"""
inference_multiclass_attack_cpguard.py

Evaluate CoDriving under adversarial feature-space attack, with and without
the CP-Guard consensus defence.

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
  5. Condition B — CP-Guard def.: run CP-Guard with perturbed attacker present.
  6. Evaluate AP and record per-frame defence outcome.

Output files (in model_dir)
---------------------------
  AP_attack_{tag}.txt              — AP summary for both conditions
  frame_log_attack_{tag}.json      — per-frame log for post-hoc analysis

Usage
-----
python opencood/tools/inference_multiclass_attack_cpguard.py \\
    --model_dir v2xverse/codriving/perception \\
    --attack_type pgd --attack_goal fp \\
    --attack_eps 0.2 --pgd_steps 10 \\
    --attacker_idx -1 \\
    --eps 0.08 --n_upper 5

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
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import cv2

# Restrict thread spawning to prevent OOM/CPU kills
torch.set_num_threads(2)
cv2.setNumThreads(0)

from torch.utils.data import DataLoader, Subset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.attack.feature_attack import perturb_attacker_features
from opencood.defense.cp_guard import pasac

torch.multiprocessing.set_sharing_strategy('file_system')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_parser():
    parser = argparse.ArgumentParser(
        description='CoDriving eval: adversarial feature attack + CP-Guard defence'
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

    # --- CP-Guard defence args ---
    parser.add_argument('--eps', type=float, nargs='+', default=[0.08],
                        help='CP-Guard CCLoss threshold(s)')
    parser.add_argument('--n_upper', type=int, default=5,
                        help='CP-Guard maximum benign collaborators')
    parser.add_argument('--no_pasac', action='store_true',
                        help='Bypass PASAC filtering')
    parser.add_argument('--log_frames', action='store_true',
                        help='Save per-frame CP-Guard log to JSON (for FDR analysis)')
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
    return random.choice(collab_indices)


def _fuse_decode_all(model, perturbed_raw, psm_single, t_matrix, rl, all_indices):
    """Fuse ego + all collaborators (including attacker) — the no-defence path."""
    with torch.no_grad():
        cls, bbox, fused = model.fuse_decode_subset(
            perturbed_raw, psm_single, t_matrix, rl, all_indices)
    return cls, bbox, fused


def _build_output_dict(model, cls, bbox, fused):
    """Build the output_dict expected by dataset.post_process_multiclass."""
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
           f'_atkeps{opt.attack_eps}'
           f'_N{opt.n_upper}')
    tag += opt.note

    np.random.seed(30330)
    torch.manual_seed(10000)
    random.seed(42)
    torch.backends.cudnn.benchmark = True

    print('Building dataset …')
    ds = build_dataset(hypes, visualize=True, train=False)

    if opt.max_frames > 0:
        _rng = random.Random(42)
        _indices = list(range(len(ds)))
        _rng.shuffle(_indices)
        _indices = _indices[:opt.max_frames]
        _ds = Subset(ds, _indices)
        _collate = ds.collate_batch_test
    else:
        _ds = ds
        _collate = ds.collate_batch_test

    loader = DataLoader(
        _ds, batch_size=1, num_workers=4,
        collate_fn=_collate,
        shuffle=False, pin_memory=True, drop_last=False)

    # Two AP accumulators: no-defence (attacked) and CP-Guard-defended
    stat_nodef   = _init_result_stat()
    stat_cpguard = {e: _init_result_stat() for e in opt.eps}

    frame_logs = {e: {} for e in opt.eps}  # per-frame log for post-hoc analysis
    n_evaluated = 0  # actual frames processed (for TRR/FAR when log_frames=False)

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
            n_evaluated += 1
            continue

        n_evaluated += 1

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
        # Step 3b: CP-Guard defence
        # ---------------------------------------------------------------
        with torch.no_grad():
            cls0, bbox0, _ = model.fuse_decode_subset(
                raw_features, psm_single, t_matrix, record_len, [0])
            Y_0 = torch.sigmoid(cls0)

        subset_cache = {}
        def fuse_decode_fn(indices):
            key = tuple(indices)
            if key not in subset_cache:
                cls_s, _, _ = model.fuse_decode_subset(
                    perturbed_raw, psm_single, t_matrix, record_len, indices)
                subset_cache[key] = torch.sigmoid(cls_s)
            return subset_cache[key]

        for e in opt.eps:
            if opt.no_pasac:
                accepted_set = all_collab[:]
            else:
                accepted_set = pasac(all_collab, Y_0, fuse_decode_fn, opt.n_upper, e)

            # Final CP-Guard prediction
            final_indices = [0] + accepted_set  # ego-only if no consensus
            with torch.no_grad():
                cls_r, bbox_r, fused_r = model.fuse_decode_subset(
                    perturbed_raw, psm_single, t_matrix, record_len, final_indices)
            od_r = _build_output_dict(model, cls_r, bbox_r, fused_r)
            pred_r, score_r, gt_r = ds.post_process_multiclass(
                {'ego': batch_data['ego']}, od_r, False)

            for iou_thresh in [0.3, 0.5, 0.7]:
                eval_utils.caluclate_tp_fp_multiclass(
                    pred_r, score_r, gt_r, stat_cpguard[e], iou_thresh)

            attacker_rejected = (atk_idx not in accepted_set)
            if opt.log_frames:
                frame_logs[e][i] = {
                    'n_cavs':              N_cavs,
                    'attacker_idx':        atk_idx,
                    'collab_indices':      all_collab,
                    'accepted_set':        list(accepted_set),
                    'attacker_rejected':   attacker_rejected,
                }

            print(f'  [EPS={e}] CAVs={N_cavs}  attacker={atk_idx}  '
                  f'accepted={accepted_set}  rejected={attacker_rejected}',
                  flush=True)

        # ---------------------------------------------------------------
        # Accumulate No-defence AP
        # ---------------------------------------------------------------
        for iou_thresh in [0.3, 0.5, 0.7]:
            eval_utils.caluclate_tp_fp_multiclass(
                pred_nd, score_nd, gt_nd, stat_nodef, iou_thresh)

        # Incremental save every 200 frames so partial results survive interruption
        if i % 200 == 0 and opt.log_frames:
            for e in opt.eps:
                tag_e_tmp = tag + f'_eps{e}'
                lp = os.path.join(opt.model_dir, f'frame_log_attack{tag_e_tmp}.json')
                with open(lp, 'w') as _f:
                    json.dump(frame_logs[e], _f)

        torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Final AP
    # -------------------------------------------------------------------
    print('\n=== No-Defence (attacked) ===')
    res_nd, _, _, _ = eval_utils.eval_final_results_multiclass(
        stat_nodef, opt.model_dir,
        'attack_nodef' + tag)

    for e in opt.eps:
        print(f'\n=== CP-Guard Defence (attacked) EPS={e} ===')
        tag_e = tag + f'_eps{e}'
        res_r, _, _, _ = eval_utils.eval_final_results_multiclass(
            stat_cpguard[e], opt.model_dir,
            'attack_cpguard' + tag_e)

        # -------------------------------------------------------------------
        # Aggregate defence statistics
        # -------------------------------------------------------------------
        f_logs = frame_logs[e]
        if f_logs:
            # Precise counts from per-frame log
            n_total        = len(f_logs)
            n_rejected     = sum(1 for v in f_logs.values() if v['attacker_rejected'])
            n_accepted     = n_total - n_rejected
            n_ego_fallback = sum(1 for v in f_logs.values() if len(v['accepted_set']) == 0)
        else:
            # --log_frames not set: fall back to frame counter (TRR/FAR unavailable)
            n_total        = max(n_evaluated, 1)
            n_rejected     = 0
            n_accepted     = 0
            n_ego_fallback = 0

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
            f' CP-Guard: eps_ccloss={e}  N={opt.n_upper}\n'
            f'========================================\n'
            f' Frames evaluated : {n_total}\n'
            f' Attacker rejected (TRR) : {n_rejected}/{n_total} = {100*trr:.1f}%\n'
            f' Attacker accepted (FAR) : {n_accepted}/{n_total} = {100*far:.1f}%\n'
            f' Ego-only fallback       : {n_ego_fallback}/{n_total} '
            f'= {100*n_ego_fallback/max(n_total,1):.1f}%\n'
            f'\n AP@0.5 comparison:\n'
            f'   Class       No-def(atk)  CP-Guard(def) Recovery\n'
            f'   Vehicle(0)  {ap_nd_v:.3f}        {ap_r_v:.3f}        '
            f'{ap_r_v - ap_nd_v:+.3f}\n'
            f'   Pedestr(1)  {ap_nd_p:.3f}        {ap_r_p:.3f}        '
            f'{ap_r_p - ap_nd_p:+.3f}\n'
            f'   Bicycle(3)  {ap_nd_b:.3f}        {ap_r_b:.3f}        '
            f'{ap_r_b - ap_nd_b:+.3f}\n'
            f'========================================\n'
        )
        print(summary)

        ap_txt = os.path.join(opt.model_dir, f'AP_attack{tag_e}.txt')
        with open(ap_txt, 'w') as f:
            f.write(summary)
        print(f'AP summary written to {ap_txt}')

        if opt.log_frames:
            lp = os.path.join(opt.model_dir, f'frame_log_attack{tag_e}.json')
            with open(lp, 'w') as f:
                json.dump(f_logs, f, indent=2)
            print(f'Per-frame log saved to {lp}')


if __name__ == '__main__':
    main()
