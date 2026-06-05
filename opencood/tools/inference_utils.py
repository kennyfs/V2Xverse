# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.utils.transformation_utils import get_relative_transformation
from opencood.utils.box_utils import create_bbx, project_box3d, nms_rotated
from opencood.utils.camera_utils import indices_to_depth
from sklearn.metrics import mean_squared_error

def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict



def inference_no_fusion(batch_data, model, dataset, single_gt=False):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    single_gt : bool
        if True, only use ego agent's label.
        else, use all agent's merged labels.
    """
    output_dict_ego = OrderedDict()
    if single_gt:
        batch_data = {'ego': batch_data['ego']}
        
    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_no_fusion(batch_data,  # only for late fusion dataset
                             output_dict_ego)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict

def inference_no_fusion_w_uncertainty(batch_data, model, dataset):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict_ego = OrderedDict()

    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor, uncertainty_tensor = \
        dataset.post_process_no_fusion_uncertainty(batch_data, # only for late fusion dataset
                             output_dict_ego)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor, \
                    "uncertainty_tensor" : uncertainty_tensor}

    return return_dict


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    output_dict['ego'] = model(cav_content)
    
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)
    
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    if "depth_items" in output_dict['ego']:
        return_dict.update({"depth_items" : output_dict['ego']['depth_items']})
    if "comm_rate" in output_dict['ego']:
        return_dict.update({"comm_rate" : output_dict['ego']['comm_rate']})
    return return_dict


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return_dict = inference_early_fusion(batch_data, model, dataset)
    return return_dict

def inference_intermediate_fusion_multiclass(batch_data, model, dataset, online_eval_only=False):
    """
    Model multi-class detection inference for intermediate fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.IntermediatemulticlassFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_multiclass(batch_data,
                             output_dict, online_eval_only)
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    if "depth_items" in output_dict['ego']:
        return_dict.update({"depth_items" : output_dict['ego']['depth_items']})
    if "comm_rate" in output_dict['ego']:
        return_dict.update({"comm_rate" : output_dict['ego']['comm_rate']})
    return return_dict

def inference_single_from_intermediate_fusion_multiclass(batch_data, model, dataset, online_eval_only=False):
    """
    Model inference for none fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    # cav_content = batch_data['ego']
    # output_dict['ego'] = model(cav_content)

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)
        output_dict[cav_id]['cls_preds'] = output_dict[cav_id]['cls_preds_single_ego']
        output_dict[cav_id]['reg_preds_multiclass'] = output_dict[cav_id]['reg_preds_multiclass_single_ego']
        output_dict[cav_id]['bbox_preds'] = output_dict[cav_id]['bbox_preds_single_ego']

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_multiclass(batch_data,
                             output_dict, online_eval_only)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    if "depth_items" in output_dict['ego']:
        return_dict.update({"depth_items" : output_dict['ego']['depth_items']})
    return return_dict

def inference_intermediate_fusion_multiclass_driving_request(batch_data, model, dataset, online_eval_only=False, waypoints=None):
    """
    Model inference multi-class detection for intermediate fusion based on driving request.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset
    waypoints : points to rend request map

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    # cav_content = batch_data['ego']
    # output_dict['ego'] = model(cav_content)

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content, waypoints)
    
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_multiclass(batch_data,
                             output_dict, online_eval_only)
    
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    if "depth_items" in output_dict['ego']:
        return_dict.update({"depth_items" : output_dict['ego']['depth_items']})
    if "comm_rate" in output_dict['ego']:
        return_dict.update({"comm_rate" : output_dict['ego']['comm_rate']})
    return return_dict

def inference_late_fusion_multiclass(batch_data, model, dataset):
    """
    Model inference multi-class detection for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LatemulticlassFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """

    return_dict = inference_intermediate_fusion_multiclass(batch_data, model, dataset)
    return return_dict

def inference_early_fusion_multiclass(batch_data, model, dataset):
    """
    Model inference multi-class detection for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlymulticlassFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return_dict = inference_intermediate_fusion_multiclass(batch_data, model, dataset)
    return return_dict

def inference_no_fusion_multiclass(batch_data, model, dataset, single_gt=False):
    """
    Model inference multi-class detection for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LatemulticlassFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    single_gt : bool
        if True, only use ego agent's label.
        else, use all agent's merged labels.
    """
    output_dict_ego = OrderedDict()
    
    if single_gt:
        batch_data = {'ego': batch_data['ego']}

    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_multiclass_no_fusion(batch_data,  # only for late fusion dataset
                             output_dict_ego)
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict

def inference_intermediate_fusion_multiclass_cpguard(
        batch_data,
        model,
        dataset,
        n_upper: int = 5,
        eps: float = 0.08,
        online_eval_only: bool = False,
        no_pasac: bool = False):
    """
    CP-Guard-protected inference for intermediate-fusion multiclass detection.

    Replaces a plain model.forward() call with the two-phase CP-Guard
    pipeline:

      Phase 1 — encode once
          Run VFE + scatter + backbone + shrink for every CAV in the scene
          and cache the resulting BEV feature maps.

      Phase 2 — PASAC verification
          Split collaborators recursively, fusing ego + each candidate
          subset and measuring CCLoss against the ego-solo prediction.
          Collaborators whose fusion output agrees with the ego (CCLoss > eps)
          are accepted as benign.

      Phase 3 — final decode
          Re-fuse only the benign collaborators and run the full decoder
          (cls_head + reg_head + generate_predicted_boxes) to produce the
          output dict expected by dataset.post_process_multiclass.

    Works for batch_size = 1 (standard offline evaluation).

    Parameters
    ----------
    batch_data : dict
        As produced by the dataloader; must contain key 'ego'.
    model : PointPillarMulticlass
        The loaded V2X-ViT model (must have encode() method added by CP-Guard).
    dataset : IntermediatemulticlassFusionDataset
    n_upper : int
        Maximum benign collaborators to use (default 5).
    eps : float
        CCLoss threshold (default 0.08, optimal per CP-Guard Table 2).
    online_eval_only : bool
        Passed through to dataset.post_process_multiclass.

    Returns
    -------
    dict with keys pred_box_tensor, pred_score, gt_box_tensor.
    """
    import torch.nn.functional as F
    from collections import OrderedDict
    from opencood.defense.cp_guard import compute_ccloss, pasac

    cav_content = batch_data['ego']
    device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Phase 1: encode all CAVs (once)
    # ------------------------------------------------------------------
    spatial_features_2d, t_matrix_norm, record_len = model.encode(cav_content)
    # spatial_features_2d : [N_cavs, C, H, W]  index 0 = ego
    # t_matrix_norm       : [1, max_cav, max_cav, 2, 3]

    N_cavs = int(record_len[0].item())           # total CAVs in this scene
    collab_indices = list(range(1, N_cavs))      # [1, 2, ..., N-1]

    # ------------------------------------------------------------------
    # Helper: fuse pre-encoded subset + decode to sigmoid cls_preds
    # ------------------------------------------------------------------
    def fuse_decode_fn(indices):
        """
        indices : list[int] including 0 (ego) plus selected collaborators.
        Returns [1, C, H, W] sigmoid cls_preds.
        """
        feats = spatial_features_2d[indices]          # [k, C, H, W]
        k = len(indices)
        rl = torch.tensor([k], dtype=torch.long, device=device)

        idx_t = torch.tensor(indices, device=device)
        # Sub-select rows then columns of the normalised t_matrix
        sub_t = t_matrix_norm[:, idx_t][:, :, idx_t, :, :]  # [1, k, k, 2, 3]

        fused = model.fusion_net(feats, rl, sub_t)
        return torch.sigmoid(model.cls_head(fused))

    # ------------------------------------------------------------------
    # Phase 2: reference prediction Y_0
    #
    # CP-Guard originally uses ego-solo as Y_0, but for detection this
    # fails: collaborative fusion discovers new objects the ego can't see,
    # so ego-solo and collaborative outputs legitimately diverge even with
    # all-benign collaborators.  Low CCLoss ≤ eps causes PASAC to reject
    # every collaborator and degrade to ego-only.
    #
    # Fix: use full-group fusion as Y_0.  PASAC then detects collaborators
    # that deviate from the group consensus (outlier = malicious), rather
    # than from the ego alone.  With no attacks (benign-only baseline),
    # every subset agrees with the full-group → all accepted.
    # ------------------------------------------------------------------
    with torch.no_grad():
        if collab_indices:
            Y_0 = fuse_decode_fn([0] + collab_indices)   # full-group reference
        else:
            Y_0 = torch.sigmoid(model.cls_head(spatial_features_2d[0:1]))

    # ------------------------------------------------------------------
    # Phase 2 (cont.): PASAC — find benign collaborators
    # ------------------------------------------------------------------
    if no_pasac:
        # Bypass filtering: accept all collaborators (no-attack baseline mode)
        benign_collab = collab_indices[:]
    elif collab_indices:
        benign_collab = pasac(
            collab_indices, Y_0, fuse_decode_fn,
            n_upper=n_upper, eps=eps
        )
    else:
        benign_collab = []

    print(f'[CPGuard] CAVs={N_cavs}  collab={collab_indices}  '
          f'benign={benign_collab}  ego_only={len(benign_collab)==0}')

    # ------------------------------------------------------------------
    # Phase 3: final decode with ego + benign collaborators
    # V2XViTFusion's STTF slices x[:, 1:, ...] internally, so it cannot
    # handle L=1 (ego-only). Fall back to direct ego decode when no
    # benign collaborators are available.
    # ------------------------------------------------------------------
    if not benign_collab:
        # Ego-only: skip fusion entirely
        ego_feat = spatial_features_2d[0:1]   # [1, C, H, W]
        cls  = model.cls_head(ego_feat)
        bbox = model.reg_head(ego_feat)
        fused_final = ego_feat
    else:
        final_indices = [0] + benign_collab
        feats_final = spatial_features_2d[final_indices]
        k_final = len(final_indices)
        rl_final = torch.tensor([k_final], dtype=torch.long, device=device)
        idx_final = torch.tensor(final_indices, device=device)
        sub_t_final = t_matrix_norm[:, idx_final][:, :, idx_final, :, :]
        fused_final = model.fusion_net(feats_final, rl_final, sub_t_final)
        cls  = model.cls_head(fused_final)
        bbox = model.reg_head(fused_final)

    # Scale fused feature for the downstream planning module
    if fused_final.size(2) == 48:
        scaled_feature = F.interpolate(fused_final, scale_factor=2, mode='nearest')
    else:
        scaled_feature = fused_final

    # Replicate the generate_predicted_boxes loop from PointPillarMulticlass.forward()
    box_preds = bbox.permute(0, 2, 3, 1).contiguous()
    num_class = int(box_preds.shape[3] / 8)
    box_preds = box_preds.view(
        box_preds.shape[0], box_preds.shape[1], box_preds.shape[2], num_class, 8
    )
    bbox_temp_list = []
    for i in range(num_class):
        bpi = box_preds[:, :, :, i, :].permute(0, 3, 1, 2)
        _, bt = model.generate_predicted_boxes(cls[:, i, :, :], bpi)
        bbox_temp_list.append(bt)
    reg_preds_multiclass = torch.stack(bbox_temp_list, dim=1)

    output_dict = OrderedDict()
    output_dict['ego'] = {
        'cls_preds':           cls,
        'bbox_preds':          bbox,
        'reg_preds_multiclass': reg_preds_multiclass,
        'fused_feature':       scaled_feature,
    }

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process_multiclass(
        batch_data, output_dict, online_eval_only
    )

    return {
        'pred_box_tensor': pred_box_tensor,
        'pred_score':      pred_score,
        'gt_box_tensor':   gt_box_tensor,
    }


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)


def depth_metric(depth_items, grid_conf):
    # depth logdit: [N, D, H, W]
    # depth gt indices: [N, H, W]
    depth_logit, depth_gt_indices = depth_items
    depth_pred_indices = torch.argmax(depth_logit, 1)
    depth_pred = indices_to_depth(depth_pred_indices, *grid_conf['ddiscr'], mode=grid_conf['mode']).flatten()
    depth_gt = indices_to_depth(depth_gt_indices, *grid_conf['ddiscr'], mode=grid_conf['mode']).flatten()
    rmse = mean_squared_error(depth_gt.cpu(), depth_pred.cpu(), squared=False)
    return rmse


def fix_cavs_box(pred_box_tensor, gt_box_tensor, pred_score, batch_data):
    """
    Fix the missing pred_box and gt_box for ego and cav(s).
    Args:
        pred_box_tensor : tensor
            shape (N1, 8, 3), may or may not include ego agent prediction, but it should include
        gt_box_tensor : tensor
            shape (N2, 8, 3), not include ego agent in camera cases, but it should include
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    Returns:
        pred_box_tensor : tensor
            shape (N1+?, 8, 3)
        gt_box_tensor : tensor
            shape (N2+1, 8, 3)
    """
    if pred_box_tensor is None or gt_box_tensor is None:
        return pred_box_tensor, gt_box_tensor, pred_score, 0
    # prepare cav's boxes

    # if key only contains "ego", like intermediate fusion
    if 'record_len' in batch_data['ego']:
        lidar_pose =  batch_data['ego']['lidar_pose'].cpu().numpy()
        N = batch_data['ego']['record_len']
        relative_t = get_relative_transformation(lidar_pose) # [N, 4, 4], cav_to_ego, T_ego_cav
    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        relative_t = []
        for cavid, cav_data in batch_data.items():
            relative_t.append(cav_data['transformation_matrix'])
        N = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()
        
    extent = [2.45, 1.06, 0.75]
    ego_box = create_bbx(extent).reshape(1, 8, 3) # [8, 3]
    ego_box[..., 2] -= 1.2 # hard coded now

    box_list = [ego_box]
    
    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_tensor = torch.tensor(np.concatenate(box_list, axis=0), device=pred_box_tensor.device)
    
    pred_box_tensor_ = torch.cat((cav_box_tensor, pred_box_tensor), dim=0)
    gt_box_tensor_ = torch.cat((cav_box_tensor, gt_box_tensor), dim=0)

    pred_score_ = torch.cat((torch.ones(N, device=pred_score.device), pred_score))

    gt_score_ = torch.ones(gt_box_tensor_.shape[0], device=pred_box_tensor.device)
    gt_score_[N:] = 0.5

    keep_index = nms_rotated(pred_box_tensor_,
                            pred_score_,
                            0.01)
    pred_box_tensor = pred_box_tensor_[keep_index]
    pred_score = pred_score_[keep_index]

    keep_index = nms_rotated(gt_box_tensor_,
                            gt_score_,
                            0.01)
    gt_box_tensor = gt_box_tensor_[keep_index]

    return pred_box_tensor, gt_box_tensor, pred_score, N


def get_cav_box(batch_data):
    """
    Args:
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    """

    # if key only contains "ego", like intermediate fusion
    if 'record_len' in batch_data['ego']:
        lidar_pose =  batch_data['ego']['lidar_pose'].cpu().numpy()
        N = batch_data['ego']['record_len']
        relative_t = get_relative_transformation(lidar_pose) # [N, 4, 4], cav_to_ego, T_ego_cav
        agent_modality_list = batch_data['ego']['agent_modality_list']

    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        relative_t = []
        agent_modality_list = []
        for cavid, cav_data in batch_data.items():
            relative_t.append(cav_data['transformation_matrix'])
            agent_modality_list.append(cav_data['modality_name'])
        N = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()

        

    extent = [0.2, 0.2, 0.2]
    ego_box = create_bbx(extent).reshape(1, 8, 3) # [8, 3]
    ego_box[..., 2] -= 1.2 # hard coded now

    box_list = [ego_box]
    
    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_np = np.concatenate(box_list, axis=0)


# ---------------------------------------------------------------------------
# ROBOSAC inference  (CoDriving / centerpointcodriving)
# ---------------------------------------------------------------------------

def inference_intermediate_fusion_multiclass_robosac(
        batch_data,
        model,
        dataset,
        N_budget: int = 7,
        eta: float = 0.2,
        eps: float = 0.3,
        p: float = 0.99,
        score_thresh: float = 0.2,
        no_robosac: bool = False,
        online_eval_only: bool = False,
):
    """
    ROBOSAC-protected inference for CoDriving intermediate-fusion multiclass
    detection.

    Pipeline
    --------
    1. Encode all CAVs once via model.encode().
    2. Compute Y_0 (ego-only boxes) as the ROBOSAC reference.
    3. Run ROBOSAC (Algorithm 1): iteratively sample s collaborators, fuse,
       decode to boxes, measure d(Y_s, Y_0).  Stop when d <= eps (consensus).
    4. Final decode using ego + accepted collaborators.  Fall back to ego-only
       if no consensus was found within the budget.

    Parameters
    ----------
    batch_data : dict
        Standard dataloader batch (contains 'ego' key).
    model : centerpointcodriving
        Must expose encode() and fuse_decode_subset().
    dataset : IntermediatemulticlassFusionDataset
    N_budget : int   sampling budget N (default 7).
    eta      : float assumed attacker ratio (default 0.2).
    eps      : float consensus threshold d <= eps (default 0.3).
    p        : float desired success probability for computing s (default 0.99).
    score_thresh : float  score threshold for box decoding (default 0.2).
    no_robosac : bool  if True, bypass filtering and accept all collaborators.
    online_eval_only : bool  passed to dataset.post_process_multiclass.

    Returns
    -------
    dict with keys:
        pred_box_tensor, pred_score, gt_box_tensor  — standard eval keys.
        robosac_log — per-frame ROBOSAC metadata for FPR analysis:
            n_cavs, collab_indices, accepted_set, n_trials,
            reached_consensus, Y0_n_boxes, trials (list of per-trial dicts).
    """
    from collections import OrderedDict
    import torch.nn.functional as F
    from opencood.defense.robosac import run_robosac, compute_s

    cav_content = batch_data['ego']
    device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Phase 1: encode all CAVs (once)
    # ------------------------------------------------------------------
    raw_features, psm_single, pairwise_t_matrix, record_len = \
        model.encode(cav_content)

    N_cavs        = int(record_len[0].item())
    collab_indices = list(range(1, N_cavs))

    # ------------------------------------------------------------------
    # Phase 2: ROBOSAC attacker filtering
    # ------------------------------------------------------------------
    if no_robosac or not collab_indices:
        accepted_set      = collab_indices[:]
        n_trials          = 0
        reached_consensus = True
        robosac_log_extra = {}
    else:
        accepted_set, n_trials, reached_consensus, robosac_log_extra = \
            run_robosac(
                collab_indices,
                model,
                raw_features,
                psm_single,
                pairwise_t_matrix,
                record_len,
                N_budget=N_budget,
                eta=eta,
                eps=eps,
                p=p,
                score_thresh=score_thresh,
            )

    print(f'[ROBOSAC] CAVs={N_cavs}  collab={collab_indices}  '
          f'accepted={accepted_set}  trials={n_trials}  '
          f'consensus={reached_consensus}')

    # ------------------------------------------------------------------
    # Phase 3: final decode with ego + accepted collaborators
    # ------------------------------------------------------------------
    final_indices = [0] + accepted_set

    with torch.no_grad():
        cls, bbox, fused = model.fuse_decode_subset(
            raw_features, psm_single, pairwise_t_matrix, record_len,
            final_indices
        )

    # Scale fused feature for downstream planning if needed
    if fused.size(2) == 48:
        scaled_feature = F.interpolate(fused, scale_factor=2, mode='nearest')
    else:
        scaled_feature = fused

    # Build the per-class reg_preds_multiclass tensor expected by post_process
    box_preds_hwc = bbox.permute(0, 2, 3, 1).contiguous()
    B, H, W, _ = box_preds_hwc.shape
    num_class = int(box_preds_hwc.shape[3] / 8)
    box_preds_hwc = box_preds_hwc.view(B, H, W, num_class, 8)
    bbox_temp_list = []
    for i in range(num_class):
        bpi = box_preds_hwc[:, :, :, i, :].permute(0, 3, 1, 2)
        _, bt = model.generate_predicted_boxes(cls[:, i, :, :], bpi)
        bbox_temp_list.append(bt)
    reg_preds_multiclass = torch.stack(bbox_temp_list, dim=1)

    output_dict = OrderedDict()
    output_dict['ego'] = {
        'cls_preds':            cls,
        'bbox_preds':           bbox,
        'reg_preds_multiclass': reg_preds_multiclass,
        'fused_feature':        scaled_feature,
    }

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_multiclass({'ego': batch_data['ego']}, output_dict,
                                        online_eval_only)

    robosac_log = {
        'n_cavs':            N_cavs,
        'collab_indices':    collab_indices,
        'accepted_set':      accepted_set,
        'n_trials':          n_trials,
        'reached_consensus': reached_consensus,
    }
    robosac_log.update(robosac_log_extra)

    return {
        'pred_box_tensor': pred_box_tensor,
        'pred_score':      pred_score,
        'gt_box_tensor':   gt_box_tensor,
        'robosac_log':     robosac_log,
        'output_dict':     output_dict,
    }