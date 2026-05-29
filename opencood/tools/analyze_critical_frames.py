# -*- coding: utf-8 -*-
"""
analyze_critical_frames.py

Model-free dataset analysis to identify "critical frames" for the ROBOSAC
false-positive-rate experiment.

A CRITICAL FRAME is one where:
  (a) At least one actor is visible (lidar_visible=1) to some non-ego CAV
      but NOT visible to the ego.
  (b) That actor is visible to at most `max_coverage` non-ego CAVs.

These are exactly the scenarios where ROBOSAC's consensus check is expected
to produce false positives: the critical CAV holds unique information that
the ego cannot see.  Because collaborative output Y_s diverges from the
ego-only baseline Y_0, ROBOSAC mistakes the critical CAV for an attacker.

The script reads `actors_data/*.json` from every vehicle directory in the
dataset.  No model or GPU is required.

Output
------
A JSON file (default: dataset/critical_frames.json) with structure:

  {
    "weather-0/data/routes_town01_0_w0_.../": {
      "0001": {
        "is_critical": true,
        "critical_actors": [
          {
            "actor_id": "259",
            "actor_type": 0,            # 0=vehicle, 1=pedestrian, 3=bicycle
            "world_loc": [x, y, z],
            "seen_by_cavs": ["rsu_1000"],
            "coverage_count": 1         # how many non-ego CAVs see this actor
          }
        ]
      },
      ...
    },
    ...
  }

Usage
-----
python opencood/tools/analyze_critical_frames.py \\
    --data_root ./external_paths/data_root \\
    --out ./dataset/critical_frames.json \\
    --max_coverage 1

Cross-reference with inference output
--------------------------------------
After running inference_multiclass_robosac.py with --log_frames, load
frame_log_robosac.json and critical_frames.json.  For each critical frame:

  rejected = set(collab_indices) - set(accepted_set)
  critical_cav_id = critical_actors[0]['seen_by_cavs'][0]

  If critical_cav_index (mapped from critical_cav_id) is in `rejected`:
      → FALSE POSITIVE event

FPR = (frames where critical CAV is rejected) / (total critical frames)
"""

import argparse
import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description='Label critical frames in the V2Xverse dataset '
                    'for ROBOSAC FPR analysis'
    )
    parser.add_argument(
        '--data_root', type=str,
        default='./external_paths/data_root',
        help='Root directory of the dataset (contains weather-0, weather-1, …)'
    )
    parser.add_argument(
        '--out', type=str,
        default='./dataset/critical_frames.json',
        help='Output JSON path'
    )
    parser.add_argument(
        '--max_coverage', type=int, default=2,
        help='An actor is "unique" when <= max_coverage non-ego CAVs see it. '
             'Set to 1 for the strictest scenario (only one CAV sees it).'
    )
    parser.add_argument(
        '--actor_types', type=int, nargs='+', default=[0, 1, 3],
        help='Actor types to consider: 0=vehicle, 1=pedestrian, 3=bicycle. '
             'Default: all three.'
    )
    parser.add_argument(
        '--weather_ids', type=int, nargs='+', default=None,
        help='Restrict to specific weather IDs (e.g. 0 1 2).  '
             'Default: all weather-* directories.'
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def list_weather_dirs(data_root: Path, weather_ids=None):
    dirs = sorted([d for d in data_root.iterdir()
                   if d.is_dir() and d.name.startswith('weather-')])
    if weather_ids is not None:
        dirs = [d for d in dirs
                if int(d.name.split('-')[1]) in weather_ids]
    return dirs


def list_route_dirs(weather_dir: Path):
    data_dir = weather_dir / 'data'
    if not data_dir.exists():
        return []
    return sorted([d for d in data_dir.iterdir() if d.is_dir()
                   and not d.name.endswith('.zip')])


def list_vehicle_dirs(route_dir: Path):
    """
    Return (ego_dirs, non_ego_dirs) where ego_dirs are ego_vehicle_* and
    non_ego_dirs include rsu_* and any extra ego_vehicle_* beyond the first.

    The dataset has at most 2 ego vehicles per route (ego_vehicle_0,
    ego_vehicle_1).  We treat ego_vehicle_0 as the reference ego and
    everything else (ego_vehicle_1, rsu_*) as potential collaborators.
    """
    ego_dirs    = sorted([d for d in route_dir.iterdir()
                          if d.is_dir() and d.name.startswith('ego_vehicle_0')])
    collab_dirs = sorted([d for d in route_dir.iterdir()
                          if d.is_dir() and
                          (d.name.startswith('rsu_') or
                           (d.name.startswith('ego_vehicle_') and
                            not d.name.startswith('ego_vehicle_0')))])
    return ego_dirs, collab_dirs


def load_actors(vehicle_dir: Path, frame_id: str):
    """
    Load actors_data JSON for a given frame.  Returns dict of
    {actor_id_str: actor_info_dict} or {} on failure.
    """
    path = vehicle_dir / 'actors_data' / f'{frame_id}.json'
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def visible_actor_ids(actors: dict, actor_types):
    """Return set of actor_id strings where lidar_visible=1 and type in actor_types."""
    return {
        aid for aid, info in actors.items()
        if info.get('lidar_visible', 0) == 1
        and info.get('tpe', -1) in actor_types
    }


def get_frame_ids(vehicle_dir: Path):
    """Return sorted list of frame IDs (e.g. ['0000', '0001', ...])."""
    actors_dir = vehicle_dir / 'actors_data'
    if not actors_dir.exists():
        return []
    return sorted([f.stem for f in actors_dir.glob('*.json')])


def analyze_route(route_dir: Path, max_coverage: int, actor_types):
    """
    Analyse all frames in one route.

    Returns
    -------
    dict  { frame_id : { 'is_critical': bool, 'critical_actors': [...] } }
    """
    ego_dirs, collab_dirs = list_vehicle_dirs(route_dir)
    if not ego_dirs:
        return {}

    ego_dir    = ego_dirs[0]
    frame_ids  = get_frame_ids(ego_dir)
    if not frame_ids:
        return {}

    results = {}

    for fid in frame_ids:
        # Load ego's visible actors
        ego_actors_raw = load_actors(ego_dir, fid)
        ego_visible    = visible_actor_ids(ego_actors_raw, actor_types)

        # Per-collaborator visible sets
        collab_visible = {}  # cav_name → set of actor_ids
        for cdir in collab_dirs:
            raw = load_actors(cdir, fid)
            if raw:
                collab_visible[cdir.name] = visible_actor_ids(raw, actor_types)

        # Find actors visible to non-ego but NOT ego
        critical_actors = []
        actor_to_cavs = {}   # actor_id → list of CAV names that see it
        for cav_name, vis in collab_visible.items():
            for aid in vis:
                if aid not in ego_visible:
                    actor_to_cavs.setdefault(aid, []).append(cav_name)

        for aid, cavs in actor_to_cavs.items():
            coverage = len(cavs)
            if coverage <= max_coverage:
                # Look up world location and type from any CAV that sees it
                info = None
                for cav_name in cavs:
                    raw = load_actors(route_dir / cav_name, fid)
                    if aid in raw:
                        info = raw[aid]
                        break
                if info is None:
                    continue

                critical_actors.append({
                    'actor_id':      aid,
                    'actor_type':    info.get('tpe', -1),
                    'world_loc':     info.get('loc', []),
                    'seen_by_cavs':  cavs,
                    'coverage_count': coverage,
                })

        results[fid] = {
            'is_critical':    len(critical_actors) > 0,
            'critical_actors': critical_actors,
            'n_ego_visible':  len(ego_visible),
            'n_collab_cavs':  len(collab_visible),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    data_root = Path(args.data_root)
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weather_dirs = list_weather_dirs(data_root, args.weather_ids)
    print(f'Found {len(weather_dirs)} weather directories in {data_root}')

    output = {}
    total_frames    = 0
    critical_frames = 0

    for wdir in weather_dirs:
        route_dirs = list_route_dirs(wdir)
        print(f'  {wdir.name}: {len(route_dirs)} routes')

        for rdir in route_dirs:
            rel_key = str(rdir.relative_to(data_root))
            try:
                frame_results = analyze_route(
                    rdir, args.max_coverage, set(args.actor_types)
                )
            except Exception as exc:
                print(f'    [WARN] skipping {rdir.name}: {exc}')
                continue

            if not frame_results:
                continue

            output[rel_key] = frame_results
            n_crit  = sum(1 for v in frame_results.values() if v['is_critical'])
            n_total = len(frame_results)
            total_frames    += n_total
            critical_frames += n_crit

            if n_crit > 0:
                print(f'    {rdir.name}: {n_crit}/{n_total} critical frames '
                      f'(max_coverage={args.max_coverage})')

    # Persist
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\n=== Summary ===')
    print(f'Total frames   : {total_frames}')
    print(f'Critical frames: {critical_frames} '
          f'({100*critical_frames/max(total_frames,1):.1f}%)')
    print(f'Output written to: {out_path}')

    # Print a per-actor-type breakdown
    type_names = {0: 'vehicle', 1: 'pedestrian', 3: 'bicycle'}
    type_counts = {t: 0 for t in args.actor_types}
    for route_data in output.values():
        for fdata in route_data.values():
            for actor in fdata.get('critical_actors', []):
                t = actor.get('actor_type', -1)
                if t in type_counts:
                    type_counts[t] += 1

    print('\nCritical actor type breakdown (across all critical frames):')
    for t, cnt in type_counts.items():
        print(f'  {type_names.get(t, t)}: {cnt}')


# ---------------------------------------------------------------------------
# FPR computation helper (importable)
# ---------------------------------------------------------------------------

def compute_fpr_from_logs(critical_frames_json: str,
                           frame_log_robosac_json: str,
                           dataset_index_json: str = None):
    """
    Cross-reference critical_frames.json with frame_log_robosac.json.

    The frame_log_robosac.json uses integer frame indices (0, 1, 2, ...)
    matching the order of the DataLoader.  The critical_frames.json uses
    (route_relative_path, frame_id_str) keys.

    Because the two index spaces differ, this function requires a
    dataset_index mapping: list of (route_rel_path, frame_id_str) in the
    same order as the DataLoader.  Pass dataset_index_json if you have it,
    otherwise the function returns raw statistics only from the robosac log.

    Returns
    -------
    dict with:
        fpr                – false positive rate over critical frames
        n_critical_frames  – total frames labelled critical
        n_fp_events        – frames where critical CAV was rejected
        n_consensus_found  – frames where ROBOSAC found consensus
        n_ego_fallback     – frames where ROBOSAC fell back to ego-only
    """
    with open(critical_frames_json) as f:
        crit_data = json.load(f)
    with open(frame_log_robosac_json) as f:
        rlog_data = json.load(f)

    # Build flat list of (route_key, frame_id) in DataLoader order
    # from dataset_index.txt or provided JSON
    frame_index = []
    if dataset_index_json and os.path.exists(dataset_index_json):
        with open(dataset_index_json) as f:
            frame_index = json.load(f)

    # Aggregate from robosac log (model-level stats, no frame matching needed)
    n_total    = len(rlog_data)
    n_consensus = sum(1 for v in rlog_data.values()
                      if v.get('reached_consensus', False))
    n_fallback  = n_total - n_consensus

    # FPR requires frame matching; report raw stats if index unavailable
    stats = {
        'n_total_eval_frames': n_total,
        'n_consensus_found':   n_consensus,
        'n_ego_fallback':      n_fallback,
        'fallback_rate':       n_fallback / max(n_total, 1),
    }

    if not frame_index:
        print('[compute_fpr] No dataset_index provided; '
              'returning raw fallback stats only.')
        return stats

    # Full FPR computation when frame index is available
    n_critical = 0
    n_fp_events = 0

    for idx_str, rlog in rlog_data.items():
        idx = int(idx_str)
        if idx >= len(frame_index):
            continue
        route_key, frame_id = frame_index[idx]
        if route_key not in crit_data:
            continue
        frame_data = crit_data[route_key].get(frame_id, {})
        if not frame_data.get('is_critical', False):
            continue

        n_critical += 1

        # Check whether any critical CAV was rejected
        accepted = set(rlog.get('accepted_set', []))
        collab   = set(rlog.get('collab_indices', []))
        rejected = collab - accepted

        # Map CAV names to indices: in the dataloader, collaborators are
        # indexed 1..N-1 in the order they appear in record_len.  We cannot
        # do a name→index mapping here without the full batch_data.
        # Proxy: if ROBOSAC fell back to ego-only (accepted=[]), count as FP.
        if len(accepted) == 0 and len(collab) > 0:
            n_fp_events += 1

    fpr = n_fp_events / max(n_critical, 1)

    stats.update({
        'n_critical_frames': n_critical,
        'n_fp_events':       n_fp_events,
        'fpr':               fpr,
    })
    return stats


if __name__ == '__main__':
    main()
