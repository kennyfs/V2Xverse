# 遠端安裝指令紀錄

相較於官方的 `README.md`，這個版本的安裝指令做出了以下改良與修復，以確保在無畫面遠端主機上能自動化、無錯地完成安裝：
1. **修復 Cython 與 Numpy 錯誤**：將 `cython` 限制在 `<3.0.0`，避免 Python 3.7 下編譯 `OpenCOOD` 發生 `language_level` 錯誤。並加入 Numpy 的強制重裝，以防 Metadata 損毀。
2. **解決 `_distutils_hack` 報錯**：按照 README 提示，在 `easy_install` CARLA 之後，自動將 `setuptools` 升級回最新版。
3. **修復 PyTorch 的 `execstack` 載入錯誤**：針對舊版 PyTorch 常見的 `cannot enable executable stack` 報錯，加入了自訂的 Python 腳本來清除 `libtorch_cpu.so` 等檔案的 `PT_GNU_STACK` 標記，避免執行環境封鎖載入。
4. **改用預編譯 Spconv wheel**：原本教學要求編譯 `spconv`，這會受限於系統必須安裝好 CUDA 編譯器 (`nvcc`)。為了保證安裝成功率，這裡直接改成透過 pip 安裝預編譯好的 `spconv-cu113==2.1.21` 和相依庫 `cumm-cu113==0.2.8`，免去編譯與版本不匹配困擾。
5. **修復 CARLA API 的 libtiff 依賴錯誤**：舊版的 CARLA 0.9.10 依賴 `libtiff.so.5`，但在較新的系統中預設是 `libtiff.so.6`。腳本加入了 `conda install 'libtiff<4.4.0'` 來補齊動態連結庫，避免 `import carla` 失敗。
6. **補齊缺漏的 pypcd 與 efficientnet_pytorch**：一併加入自動下載並安裝的指令。
7. **修復 OpenCV 版本錯誤**：修正 `cv2` 模組在 4.3 以上版本常見的 `gapi_wip_gst_GStreamerPipeline` 錯誤，將 `opencv-python` 鎖定在穩定的 `4.5.5.64` 版本。
8. **下載 HuggingFace 模型權重**：加入指令透過 `huggingface_hub` Python 套件自動下載 `gjliu/v2xverse` 的所有 checkpoints。
9. **修復 Spconv 2.x API 錯位**：更新了專案內的 `simulation/leaderboard/team_code/pnp_infer_action_e2e.py`，改寫 `SpVoxelPreprocessor` 類別以防呆並相容新版 `spconv 2.x` API (`Point2VoxelCPU3d` 與 `tensorview`)。

以下是成功在遠端 `meow1` 機器上架設 `v2xverse` 環境的完整指令紀錄：

```bash
# 1. 建立 Conda 環境並安裝 PyTorch 與相關 CUDA 套件
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && conda create -y --name v2xverse python=3.7 cmake=3.22.1"
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && conda run -n v2xverse conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge"
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && conda run -n v2xverse conda install -y cudnn -c conda-forge"

# 2. 修復 PyTorch execstack 問題 (避免 import torch 發生 Invalid argument)
ssh meow1 "cat << 'EOF' > /tmp2/b12902023/IV/V2Xverse/clear_execstack.py
import sys, struct, glob
def clear_execstack(path):
    with open(path, 'r+b') as f:
        magic = f.read(4)
        if magic != b'\x7fELF': return
        is_64bit = f.read(1) == b'\x02'
        f.seek(32 if is_64bit else 28)
        phoff = struct.unpack('<Q' if is_64bit else '<I', f.read(8 if is_64bit else 4))[0]
        f.seek(54 if is_64bit else 42)
        phentsize = struct.unpack('<H', f.read(2))[0]
        phnum = struct.unpack('<H', f.read(2))[0]
        for i in range(phnum):
            f.seek(phoff + i * phentsize)
            p_type = struct.unpack('<I', f.read(4))[0]
            if p_type == 0x6474e551:
                f.seek(phoff + i * phentsize + (4 if is_64bit else 24))
                flags = struct.unpack('<I', f.read(4))[0]
                if flags & 1:
                    flags &= ~1
                    f.seek(phoff + i * phentsize + (4 if is_64bit else 24))
                    f.write(struct.pack('<I', flags))
                break
if __name__ == '__main__':
    for g in sys.argv[1:]:
        for p in glob.glob(g): clear_execstack(p)
EOF"
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && conda run -n v2xverse python /tmp2/b12902023/IV/V2Xverse/clear_execstack.py /tmp2/b12902023/miniconda3/envs/v2xverse/lib/python3.7/site-packages/torch/lib/*.so"

# 3. 修復 Numpy 中繼資料毀損問題與降級 Cython
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && conda run -n v2xverse pip install --force-reinstall numpy==1.21.6 'cython<3.0.0'"

# 4. 安裝 Python 相依套件
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && cd /tmp2/b12902023/IV/V2Xverse && conda run -n v2xverse pip install -r opencood/requirements.txt"
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && cd /tmp2/b12902023/IV/V2Xverse && conda run -n v2xverse pip install -r simulation/requirements.txt"

# 5. 設定 CARLA (下載與解壓縮)
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && cd /tmp2/b12902023/IV/V2Xverse && chmod +x simulation/setup_carla.sh && ./simulation/setup_carla.sh"

# 6. 安裝 CARLA Python API、升級 setuptools 與建立軟連結
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && cd /tmp2/b12902023/IV/V2Xverse && conda run -n v2xverse easy_install carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg && conda run -n v2xverse pip install --upgrade setuptools && mkdir -p external_paths && ln -s \${PWD}/carla/ external_paths/carla_root"

# 7. 編譯 OpenCOOD
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && cd /tmp2/b12902023/IV/V2Xverse && conda run -n v2xverse python setup.py develop"
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && cd /tmp2/b12902023/IV/V2Xverse && conda run -n v2xverse python opencood/utils/setup.py build_ext --inplace"

# 8. 安裝 Spconv (使用預編譯 wheel，免去複雜原始碼編譯與 nvcc 依賴)
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && conda run -n v2xverse pip install cumm-cu113==0.2.8 spconv-cu113==2.1.21"

# 9. 修復 CARLA 遺失的 libtiff.so.5 依賴
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && conda run -n v2xverse conda install -y -c conda-forge 'libtiff<4.4.0'"

# 10. 安裝 pypcd 與 efficientnet_pytorch，並修復 OpenCV 版本錯誤
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && cd /tmp2/b12902023/IV/V2Xverse && git clone https://github.com/klintan/pypcd.git && cd pypcd && conda run -n v2xverse pip install python-lzf && conda run -n v2xverse python setup.py install && conda run -n v2xverse pip install efficientnet_pytorch==0.7.0 opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64"

# 11. 下載預訓練模型權重 (Checkpoints)
# 透過 python 的 huggingface_hub 直接下載整個 gjliu/v2xverse 倉庫到 checkpoints 資料夾，避開系統缺少 git-lfs 的問題
ssh meow1 "export PATH=/tmp2/b12902023/miniconda3/bin:\$PATH && cd /tmp2/b12902023/IV/V2Xverse && conda run -n v2xverse pip install huggingface_hub && conda run -n v2xverse python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='gjliu/v2xverse', local_dir='checkpoints')\""
```
