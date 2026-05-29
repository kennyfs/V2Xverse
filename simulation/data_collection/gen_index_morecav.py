"""Generate dataset_index_train.txt and dataset_index_test.txt for weather-0-morecav.

Routes 0-4  → test split (dataset_index_test.txt)
Routes 5-34 → train split (dataset_index_train.txt)
Also writes a combined dataset_index.txt for convenience.
"""
import os
import re
from tqdm import tqdm

dataset_directory = 'dataset'
weather_dir = 'weather-0-morecav'
data_path = os.path.join(dataset_directory, weather_dir, 'data')

route_num = {}  # town_route_id -> best record
town_num = {}

subs = os.listdir(data_path)
for sub in tqdm(subs):
    seq_len = 1000000
    sub_path = os.path.join(data_path, sub) + '/'
    try:
        agent_list = os.listdir(sub_path)
    except Exception:
        continue
    ego_list = [e for e in agent_list if e.startswith('ego')]
    rsu_list = [e for e in agent_list if e.startswith('rsu')]
    for ego in ego_list:
        ego_path = os.path.join(sub_path, ego)
        try:
            n = len(os.listdir(os.path.join(ego_path, 'rgb_front')))
        except Exception:
            n = 0
        if n < seq_len:
            seq_len = n
    if seq_len > 50:
        if len(ego_list) == 1 and len(rsu_list) == 0:
            continue
        town_route_id = sub.split('_')[1] + '_' + sub.split('_')[2]
        if town_route_id not in route_num:
            town = int(sub.split('_')[1][-2:])
            route_num[town_route_id] = {
                'seq_len': seq_len,
                'sub_path': sub_path,
                'n_ego': len(ego_list),
            }
            town_num[town] = town_num.get(town, 0) + 1
        elif route_num[town_route_id]['seq_len'] < seq_len:
            route_num[town_route_id] = {
                'seq_len': seq_len,
                'sub_path': sub_path,
                'n_ego': len(ego_list),
            }

# Split into train (routes 5-34) and test (routes 0-4) by route number
# Route number extracted from sub_path like "dataset/weather-0-morecav/data/routes_town05_7_w0_..."
route_num_pattern = re.compile(r'routes_town\d+_(\d+)_')

train_lines = []
test_lines = []
all_lines = []
train_frames = train_samples = 0
test_frames = test_samples = 0

for tid in route_num:
    r = dict(route_num[tid])
    r['seq_len'] -= 25
    m = route_num_pattern.search(r['sub_path'])
    route_id = int(m.group(1)) if m else -1
    line = "{} {} {}\n".format(r['sub_path'], r['seq_len'], r['n_ego'])
    all_lines.append(line)
    if route_id <= 4:
        test_lines.append(line)
        test_frames += r['seq_len']
        test_samples += r['seq_len'] * r['n_ego']
    else:
        train_lines.append(line)
        train_frames += r['seq_len']
        train_samples += r['seq_len'] * r['n_ego']

base_out = os.path.join(dataset_directory, weather_dir)
os.makedirs(base_out, exist_ok=True)

with open(os.path.join(base_out, 'dataset_index_train.txt'), 'w') as f:
    f.writelines(train_lines)
with open(os.path.join(base_out, 'dataset_index_test.txt'), 'w') as f:
    f.writelines(test_lines)
with open(os.path.join(base_out, 'dataset_index.txt'), 'w') as f:
    f.writelines(all_lines)

print('Train index:', os.path.join(base_out, 'dataset_index_train.txt'))
print('  Routes:', len(train_lines), '| Frames:', train_frames, '| Samples:', train_samples)
print('Test index:', os.path.join(base_out, 'dataset_index_test.txt'))
print('  Routes:', len(test_lines), '| Frames:', test_frames, '| Samples:', test_samples)
print('Combined:', os.path.join(base_out, 'dataset_index.txt'), '|', len(all_lines), 'routes total')
print('Town distribution:', town_num)
