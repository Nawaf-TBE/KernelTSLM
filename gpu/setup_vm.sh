#!/bin/bash
# One-time setup on a new GPU VM, after upload.sh. Safe to run again: it skips what is already there.
#   ssh <user>@<vm-ip> 'bash ~/hackathon/gpu/setup_vm.sh'
set -e
cd ~/hackathon/flamingo
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h ~ | awk 'NR == 2 {print "free disk:", $4, "(multitask.sh needs 20 GB free before each job; 2 saved models use 28 GB)"}'
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -x .venv/bin/python ] || uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name())"
.venv/bin/hf download opentime/tslm-nov26 > /dev/null && echo "model cached"
mkdir -p ../runs
echo "setup done"
