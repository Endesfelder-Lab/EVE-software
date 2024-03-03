import open3d

import open3d.cuda

#pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#pip install -U -f https://www.open3d.org/docs/latest/getting_started.html open3d

import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

from open3d._build_config import _build_config

for k,v in _build_config.items():
    print(f"{k}: {v}")

print(open3d.cuda.is_available())