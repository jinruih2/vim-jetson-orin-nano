# Vision Mamba (Vim) on Jetson Orin Nano

Running [Vision Mamba (Vim)](https://github.com/hustvl/Vim) ImageNet classification eval on **NVIDIA Jetson Orin Nano 8GB** with JetPack 6.x.

> **Result:** Top-1 Acc **75.76%** / Top-5 Acc **92.79%** on ImageNet-1K val set (paper reports 76.1%)

---

## Environment

| Item | Version |
|------|---------|
| Device | Jetson Orin Nano 8GB |
| JetPack | 6.2.1 (L4T 36.4.4) |
| CUDA | 12.8 |
| Container | `dustynv/mamba:r36.4.3-cu128-24.04` |
| Python | 3.12 |
| PyTorch | 2.x (pre-installed in container) |
| mamba-ssm | 2.2.4 (pre-installed in container) |

---

## Key Challenges & Solutions

### 1. mamba-ssm version incompatibility

The original Vim repo uses `mamba-1p1p1`, a custom fork with bidirectional Mamba support (`bimamba_type`). This fork only supports CUDA 11.8 and cannot be compiled on Jetson Orin Nano (CUDA 12.x).

**Solution:** Use the `dustynv/mamba` container which ships `mamba-ssm 2.2.4`, and patch `models_mamba.py` with a custom `mamba_simple_patch.py` that re-implements bidirectional Mamba using available ops from the newer mamba-ssm.

### 2. Jetson pip index unreachable

The container sets `PIP_INDEX_URL=http://jetson.webredirect.org/...` which redirects to `pypi.jetson-ai-lab.dev` — this domain was unreachable at time of writing.

**Solution:** Override the index URL when installing packages:
```bash
PIP_INDEX_URL=https://pypi.org/simple pip install timm einops
```

### 3. ImageNet dataset structure

The eval script expects:
```
imagenet/
└── val/
    ├── n01440764/
    └── ...
```

### 4. main.py compatibility fixes

- `torch.load()` needs `weights_only=False` for PyTorch 2.6+
- The script loads `dataset_train` even in `--eval` mode — patch to skip it

---

## Step-by-Step Guide

### Step 1: Setup Docker data directory (optional, recommended for small eMMC)

If you have an external SSD mounted at `~/ssd`:

```bash
sudo tee /etc/docker/daemon.json <<'EOF'
{
    "data-root": "/mnt/ssd/docker",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia",
    "dns": ["8.8.8.8", "8.8.4.4"]
}
EOF
sudo systemctl restart docker
```

### Step 2: Install jetson-containers

```bash
git clone https://github.com/dusty-nv/jetson-containers ~/ssd/jetson-containers
bash ~/ssd/jetson-containers/install.sh
# log out and log back in (or reboot)
```

### Step 3: Enable swap (important to prevent OOM during eval)

```bash
sudo swapon ~/ssd/swapfile   # if you have one
free -h                       # verify swap is active
```

### Step 4: Clone Vim and download weights

```bash
cd ~/ssd
git clone https://github.com/hustvl/Vim.git
```

Download the Vim-tiny pretrained weight (76.1% Top-1):
```bash
# inside the container (see Step 6), or on host with huggingface_hub installed
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='hustvl/Vim-tiny-midclstok',
    filename='vim_t_midclstok_76p1acc.pth',
    local_dir='/workspace/Vim'
)
"
```

> ⚠️ Use `vim_t_midclstok_76p1acc.pth` (pretrain, 224px), **not** `vim_t_midclstok_ft_78p3acc.pth` (fine-tuned, larger resolution — causes `pos_embed` size mismatch).

### Step 5: Apply patches to Vim source

**5a. Copy `mamba_simple_patch.py` into the vim folder**

Download [`mamba_simple_patch.py`](./mamba_simple_patch.py) from this repo and place it at:
```
Vim/vim/mamba_simple_patch.py
```

This file re-implements bidirectional Mamba (`bimamba_type="v2"`) compatible with `mamba-ssm >= 2.x`.

**5b. Patch `models_mamba.py`** — use local Mamba instead of mamba_ssm:

```bash
sed -i 's/from mamba_ssm.modules.mamba_simple import Mamba/from mamba_simple_patch import Mamba/' \
    ~/ssd/Vim/vim/models_mamba.py
```

Restore `bimamba_type` parameters in `create_block`:
```bash
sed -i 's/mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, \*\*ssm_cfg, \*\*factory_kwargs)/mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, bimamba_type=bimamba_type, if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)/' \
    ~/ssd/Vim/vim/models_mamba.py
```

**5c. Patch `main.py`** — skip train dataset loading in eval mode:

```bash
sed -i 's/dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)/dataset_train, args.nb_classes = None, 1000  # skipped for eval/' \
    ~/ssd/Vim/vim/main.py

sed -i 's/sampler_train = torch.utils.data.RandomSampler(dataset_train)/sampler_train = None  # skipped for eval/' \
    ~/ssd/Vim/vim/main.py
```

Fix `torch.load` for PyTorch 2.6+:
```bash
sed -i "s/torch.load(args.resume, map_location='cpu')/torch.load(args.resume, map_location='cpu', weights_only=False)/" \
    ~/ssd/Vim/vim/main.py
```

### Step 6: Launch the container

```bash
cd ~/ssd
jetson-containers run \
  --volume ~/ssd/Vim:/workspace/Vim \
  --volume ~/ssd/imagenet:/workspace/imagenet \
  $(autotag mamba)
```

### Step 7: Install missing dependencies (inside container)

```bash
PIP_INDEX_URL=https://pypi.org/simple pip install timm einops
```

> These need to be reinstalled every time you restart the container. To avoid this, see the tip below.

**Tip:** Save to a script so you only type one command next time:
```bash
# run once to create the script
echo 'PIP_INDEX_URL=https://pypi.org/simple pip install timm einops' > /workspace/Vim/setup.sh

# every subsequent container session
bash /workspace/Vim/setup.sh
```

### Step 8: Run eval

```bash
cd /workspace/Vim
python3 vim/main.py --eval \
  --resume vim_t_midclstok_76p1acc.pth \
  --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
  --data-path /workspace/imagenet \
  --batch-size 4 \
  --num_workers 2
```

Expected output after ~8-9 minutes:
```
* Acc@1 75.762 Acc@5 92.790 loss 1.083
Accuracy of the ema network on the 50000 test images: 75.8%
```

---

## Files in this repo

| File | Description |
|------|-------------|
| `mamba_simple_patch.py` | Bidirectional Mamba implementation compatible with mamba-ssm 2.x |
| `README.md` | This guide |

---

## References

- [hustvl/Vim](https://github.com/hustvl/Vim) — Original Vision Mamba repo
- [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers) — Jetson container tooling
- [Vim-tiny weights on HuggingFace](https://huggingface.co/hustvl/Vim-tiny-midclstok)
