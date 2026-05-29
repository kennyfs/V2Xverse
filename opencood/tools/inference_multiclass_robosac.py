# -*- coding: utf-8 -*-
"""
inference_multiclass_robosac.py

Offline evaluation of the CoDriving perception model with ROBOSAC defense.

Implements Algorithm 1 from:
    "Among Us: Adversarially Robust Collaborative Perception by Consensus"
    ICCV 2023.

Usage
-----
python opencood/tools/inference_multiclass_robosac.py \\
    --model_dir v2xverse/codriving/perception \\
    --fusion_method intermediate \\
    --eta 0.2 --N_budget 7 --eps 0.3

Extra flags vs inference_multiclass.py:

  --eta        Assumed attacker ratio eta in (0,1).  Used to compute the
               guaranteed subset size s via Eq. 1 (default 0.2).
  --N_budget   Sampling budget N: max ROBOSAC iterations per frame (default 7).
  --eps        Consensus threshold: d(Y_s, Y_0) <= eps triggers early stop.
               Uses box-level IoU + Hungarian matching (default 0.3).
  --p          Desired success probability for computing s (default 0.99).
  --score_thresh  Score threshold for box decoding before NMS (default 0.2).
  --no_robosac    Bypass filtering: accept all collaborators.  Useful for
               establishing a no-attack upper-bound baseline.
  --log_frames    Save per-frame ROBOSAC log to JSON (for FPR analysis).
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import simple_vis_multiclass

torch.multiprocessing.set_sharing_strategy('file_system')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def test_parser():
    parser = argparse.ArgumentParser(
        description='ROBOSAC-protected offline evaluation for CoDriving'
    )
    parser.add_argument('--model_dir', type=str, required=False, default='',
                        help='Path to checkpoint folder')
    parser.add_argument('--fusion_method', type=str, default='intermediate',
                        help='Fusion method (only intermediate is ROBOSAC-protected)')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='Interval for saving BEV visualisations')
    parser.add_argument('--save_npy', action='store_true',
                        help='Save prediction / GT tensors as .npy files')
    parser.add_argument('--range', type=str, default='140.8,40',
                        help='Detection range [x_max, y_max] in metres')
    parser.add_argument('--modal', type=int, default=0,
                        help='Modality flag (heterogeneous setting only)')
    parser.add_argument('--no_score', action='store_true',
                        help='Omit score tensor from visualisation dict')
    parser.add_argument('--note', default='', type=str,
                        help='Optional tag appended to result filenames')
    # ROBOSAC parameters
    parser.add_argument('--eta', type=float, default=0.2,
                        help='Assumed attacker ratio (default 0.2)')
    parser.add_argument('--N_budget', type=int, default=7,
                        help='Sampling budget N (default 7)')
    parser.add_argument('--eps', type=float, default=0.3,
                        help='IoU-based consensus threshold d<=eps (default 0.3)')
    parser.add_argument('--p', type=float, default=0.99,
                        help='Success probability for computing s (default 0.99)')
    parser.add_argument('--score_thresh', type=float, default=0.2,
                        help='Box decode score threshold (default 0.2)')
    parser.add_argument('--no_robosac', action='store_true',
                        help='Bypass ROBOSAC: accept all collaborators '
                             '(no-attack upper-bound baseline)')
    parser.add_argument('--log_frames', action='store_true',
                        help='Save per-frame ROBOSAC log JSON for FPR analysis')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='Stop after this many frames (-1 = all)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    opt = test_parser()
    assert opt.fusion_method in [
        'late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'
    ]

    hypes = yaml_utils.load_yaml(None, opt)
    hypes['validate_dir'] = hypes['test_dir']

    if 'OPV2V' in hypes['test_dir'] or 'v2xsim' in hypes['test_dir']:
        assert 'test' in hypes['validate_dir']

    left_hand = ('OPV2V' in hypes['test_dir'] or 'V2XSET' in hypes['test_dir'])
    print(f'Left-hand visualising: {left_hand}')

    if 'box_align' in hypes:
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f'Resumed from epoch {resume_epoch}')

    tag = f'_epoch{resume_epoch}_robosac_eta{opt.eta}_N{opt.N_budget}_eps{opt.eps}'
    if opt.no_robosac:
        tag += '_nofilter'
    opt.note += tag

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    np.random.seed(30330)
    torch.manual_seed(10000)

    print('Building dataset')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)

    if opt.max_frames > 0:
        import random as _random
        _rng = _random.Random(42)
        _indices = list(range(len(opencood_dataset)))
        _rng.shuffle(_indices)
        _indices = _indices[:opt.max_frames]
        from torch.utils.data import Subset
        _dataset = Subset(opencood_dataset, _indices)
        _collate = opencood_dataset.collate_batch_test
    else:
        _dataset = opencood_dataset
        _collate = opencood_dataset.collate_batch_test

    data_loader = DataLoader(
        _dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=_collate,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    # AP accumulator (multi-class: keys 0=vehicle, 1=pedestrian, 3=bicycle)
    result_stat = {}
    if hypes['model']['args'].get('multi_class', False):
        for c in [0, 1, 3]:
            result_stat[c] = {
                iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
                for iou in [0.3, 0.5, 0.7]
            }
    else:
        result_stat = {
            iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
            for iou in [0.3, 0.5, 0.7]
        }

    infer_info = opt.fusion_method + opt.note

    _ap_suffix = '_nofilter' if opt.no_robosac else '_defense'
    ap_path = os.path.join(opt.model_dir, f'AP_robosac{_ap_suffix}.txt')
    ap_file = open(ap_path, 'w')
    print(f'AP results will be written to {ap_path}')

    # Per-frame log accumulator (used for FPR analysis)
    frame_logs = {}

    for i, batch_data in enumerate(data_loader):
        print(f'{infer_info}_{i}')
        if batch_data is None:
            continue

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == 'intermediate':
                infer_result = \
                    inference_utils.inference_intermediate_fusion_multiclass_robosac(
                        batch_data, model, opencood_dataset,
                        N_budget=opt.N_budget,
                        eta=opt.eta,
                        eps=opt.eps,
                        p=opt.p,
                        score_thresh=opt.score_thresh,
                        no_robosac=opt.no_robosac,
                    )
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion_multiclass(
                    batch_data, model, opencood_dataset, single_gt=True)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion_multiclass(
                    batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion_multiclass(
                    batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion_multiclass(
                    batch_data, model, opencood_dataset, single_gt=True)
            else:
                raise NotImplementedError(opt.fusion_method)

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor   = infer_result['gt_box_tensor']
            pred_score      = infer_result['pred_score']

            eval_utils.caluclate_tp_fp_multiclass(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
            eval_utils.caluclate_tp_fp_multiclass(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
            eval_utils.caluclate_tp_fp_multiclass(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)

            # Collect per-frame log
            if opt.log_frames and 'robosac_log' in infer_result:
                rlog = infer_result['robosac_log']
                frame_logs[i] = {
                    'n_cavs':            rlog.get('n_cavs', -1),
                    'collab_indices':    rlog.get('collab_indices', []),
                    'accepted_set':      rlog.get('accepted_set', []),
                    'n_trials':          rlog.get('n_trials', 0),
                    'reached_consensus': rlog.get('reached_consensus', False),
                    'Y0_n_boxes':        rlog.get('Y0_n_boxes', 0),
                    's':                 rlog.get('s', -1),
                    'trials':            rlog.get('trials', []),
                }

            if opt.save_npy:
                npy_root = os.path.join(opt.model_dir, 'npy_robosac')
                os.makedirs(npy_root, exist_ok=True)
                inference_utils.save_prediction_gt(
                    pred_box_tensor, gt_box_tensor,
                    batch_data['ego']['origin_lidar'][0],
                    i, npy_root
                )

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                vis_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                os.makedirs(vis_root, exist_ok=True)
                simple_vis_multiclass.visualize(
                    infer_result,
                    batch_data['ego']['origin_lidar'][0],
                    hypes['postprocess']['gt_range'],
                    os.path.join(vis_root, '3d_%05d.png' % i),
                    method='3d', left_hand=left_hand)
                simple_vis_multiclass.visualize(
                    infer_result,
                    batch_data['ego']['origin_lidar'][0],
                    hypes['postprocess']['gt_range'],
                    os.path.join(vis_root, 'bev_%05d.png' % i),
                    method='bev', left_hand=left_hand)

        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Final AP
    # -----------------------------------------------------------------------
    all_class_results, _, _, _ = eval_utils.eval_final_results_multiclass(
        result_stat, opt.model_dir, infer_info
    )
    AP_all = {}
    for tpe in all_class_results:
        AP_all[tpe] = {
            'ap30': all_class_results[tpe]['ap30'],
            'ap50': all_class_results[tpe]['ap50'],
            'ap70': all_class_results[tpe]['ap70'],
        }

    yaml_utils.save_yaml(AP_all,
                         os.path.join(opt.model_dir, 'AP_robosac030507.yaml'))

    ap_file.write(
        'veh_ap30: {:.4f} veh_ap50: {:.4f} veh_ap70: {:.4f} '
        'ped_ap30: {:.4f} ped_ap50: {:.4f} ped_ap70: {:.4f} '
        'bicy_ap30: {:.4f} bicy_ap50: {:.4f} bicy_ap70: {:.4f}'.format(
            all_class_results[0]['ap30'], all_class_results[0]['ap50'],
            all_class_results[0]['ap70'],
            all_class_results[1]['ap30'], all_class_results[1]['ap50'],
            all_class_results[1]['ap70'],
            all_class_results[3]['ap30'], all_class_results[3]['ap50'],
            all_class_results[3]['ap70'],
        )
    )
    ap_file.close()
    print(f'Done. AP results saved to {ap_path}')

    # -----------------------------------------------------------------------
    # Save per-frame log for FPR analysis
    # -----------------------------------------------------------------------
    if opt.log_frames and frame_logs:
        log_fname = ('frame_log_nofilter.json' if opt.no_robosac
                     else 'frame_log_robosac.json')
        log_path = os.path.join(opt.model_dir, log_fname)
        with open(log_path, 'w') as f:
            json.dump(frame_logs, f, indent=2)
        print(f'Per-frame log saved to {log_path}')


if __name__ == '__main__':
    main()
