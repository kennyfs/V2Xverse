# -*- coding: utf-8 -*-
"""
inference_multiclass_cpguard.py

Offline evaluation of the V2X-ViT perception model with CP-Guard defense.

Mirrors opencood/tools/inference_multiclass.py but routes the 'intermediate'
fusion branch through the CP-Guard PASAC+CCLoss pipeline instead of plain
intermediate fusion.

Usage
-----
python opencood/tools/inference_multiclass_cpguard.py \
    --model_dir v2xverse/v2xvit/perception \
    --fusion_method intermediate \
    --n_upper 5 \
    --eps 0.08

All flags from inference_multiclass.py are preserved; two extra flags are
added:

  --n_upper   Maximum number of benign collaborators PASAC may accept
              (default 5, matching max_cav in the V2X-ViT config).
  --eps       CCLoss threshold; collaborators with CCLoss > eps are accepted
              as benign (default 0.08, optimal per CP-Guard Table 2).
"""

import argparse
import os
import importlib

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
        description='CP-Guard protected offline evaluation for V2X-ViT'
    )
    parser.add_argument('--model_dir', type=str, required=False, default='',
                        help='Path to checkpoint folder')
    parser.add_argument('--fusion_method', type=str, default='intermediate',
                        help='Fusion method: intermediate (others fall back to '
                             'the unprotected implementation)')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='Interval for saving BEV visualisations')
    parser.add_argument('--save_npy', action='store_true',
                        help='Save prediction / GT tensors as .npy files')
    parser.add_argument('--range', type=str, default='140.8,40',
                        help='Detection range [x_max, y_max] in metres')
    parser.add_argument('--modal', type=int, default=0,
                        help='Modality (heterogeneous setting only)')
    parser.add_argument('--no_score', action='store_true',
                        help='Do not include score tensor in visualisation dict')
    parser.add_argument('--note', default='', type=str,
                        help='Optional tag appended to result filenames')
    # CP-Guard parameters
    parser.add_argument('--n_upper', type=int, default=5,
                        help='PASAC: max benign collaborators (default 5)')
    parser.add_argument('--eps', type=float, default=0.08,
                        help='PASAC: CCLoss acceptance threshold (default 0.08)')
    parser.add_argument('--no_pasac', action='store_true',
                        help='Bypass PASAC filtering: accept all collaborators. '
                             'Use to verify fusion pipeline and establish a no-attack baseline.')
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

    # Heterogeneous modality setup (unchanged from inference_multiclass.py)
    if 'heter' in hypes:
        if opt.modal == 0:
            hypes['heter']['lidar_ratio'] = 1
            hypes['heter']['ego_modality'] = 'lidar'
            opt.note += '_lidaronly'
        elif opt.modal == 1:
            hypes['heter']['lidar_ratio'] = 0
            hypes['heter']['ego_modality'] = 'camera'
            opt.note += '_camonly'
        elif opt.modal == 2:
            hypes['heter']['lidar_ratio'] = 0
            hypes['heter']['ego_modality'] = 'lidar'
            opt.note += 'ego_lidar_other_cam'
        elif opt.modal == 3:
            hypes['heter']['lidar_ratio'] = 1
            hypes['heter']['ego_modality'] = 'camera'
            opt.note += '_ego_cam_other_lidar'

        x_min, x_max = -140.8, 140.8
        y_min, y_max = -40, 40
        opt.note += f'_{x_max}_{y_max}'
        hypes['fusion']['args']['grid_conf']['xbound'] = [
            x_min, x_max, hypes['fusion']['args']['grid_conf']['xbound'][2]
        ]
        hypes['fusion']['args']['grid_conf']['ybound'] = [
            y_min, y_max, hypes['fusion']['args']['grid_conf']['ybound'][2]
        ]
        hypes['model']['args']['grid_conf'] = hypes['fusion']['args']['grid_conf']

        new_cav_range = [
            x_min, y_min,
            hypes['postprocess']['anchor_args']['cav_lidar_range'][2],
            x_max, y_max,
            hypes['postprocess']['anchor_args']['cav_lidar_range'][5],
        ]
        hypes['preprocess']['cav_lidar_range'] = new_cav_range
        hypes['postprocess']['anchor_args']['cav_lidar_range'] = new_cav_range
        hypes['postprocess']['gt_range'] = new_cav_range
        hypes['model']['args']['lidar_args']['lidar_range'] = new_cav_range
        if 'camera_mask_args' in hypes['model']['args']:
            hypes['model']['args']['camera_mask_args']['cav_lidar_range'] = \
                new_cav_range

        yaml_utils_lib = importlib.import_module('opencood.hypes_yaml.yaml_utils')
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes['yaml_parser']:
                hypes = func(hypes)

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
    opt.note += f'_epoch{resume_epoch}_cpguard_eps{opt.eps}_N{opt.n_upper}'

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    np.random.seed(30330)
    torch.manual_seed(10000)

    print('Building dataset')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    # Result statistics
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
    AP_all = {}

    file_path = os.path.join(opt.model_dir, 'AP_cpguard.txt')
    log_file = open(file_path, 'w')
    print(f'Results will be written to {file_path}')

    for i, batch_data in enumerate(data_loader):
        print(f'{infer_info}_{i}')
        if batch_data is None:
            continue

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == 'intermediate':
                # ---- CP-Guard protected path ----
                infer_result = inference_utils.inference_intermediate_fusion_multiclass_cpguard(
                    batch_data, model, opencood_dataset,
                    n_upper=opt.n_upper, eps=opt.eps,
                    no_pasac=opt.no_pasac
                )
            elif opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion_multiclass(
                    batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion_multiclass(
                    batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion_multiclass(
                    batch_data, model, opencood_dataset, single_gt=True)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(
                    batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(
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

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy_cpguard')
                os.makedirs(npy_save_path, exist_ok=True)
                inference_utils.save_prediction_gt(
                    pred_box_tensor, gt_box_tensor,
                    batch_data['ego']['origin_lidar'][0],
                    i, npy_save_path
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

    # Final AP computation
    all_class_results, _, _, _ = eval_utils.eval_final_results_multiclass(
        result_stat, opt.model_dir, infer_info
    )
    for tpe in all_class_results:
        if tpe not in AP_all:
            AP_all[tpe] = {'ap30': [], 'ap50': [], 'ap70': []}
        AP_all[tpe]['ap30'].append(all_class_results[tpe]['ap30'])
        AP_all[tpe]['ap50'].append(all_class_results[tpe]['ap50'])
        AP_all[tpe]['ap70'].append(all_class_results[tpe]['ap70'])

    yaml_utils.save_yaml(AP_all, os.path.join(opt.model_dir, 'AP_cpguard030507.yaml'))

    log_file.write(
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
    log_file.close()
    print(f'Done.  Results saved to {file_path}')


if __name__ == '__main__':
    main()
