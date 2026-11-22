#!/bin/bash
# Only for the GPU fallback of EXP's HAR-CoT job (multitask.sh har); the first choice is the Mac, at 0 USD. Run it
# after setup_vm.sh and upload.sh, in tmux or detached, online. Safe to run again: it skips what is there.
#   ssh -t <host> '~/hackathon/flamingo/.venv/bin/hf auth login'    # once; google/gemma-3-270m is gated
#   ssh <host> 'bash ~/hackathon/gpu/setup_har.sh'
# Needs: an HF account that accepted the Gemma license, about 8 GB of disk.
set -e
cd ~/hackathon
export PATH="$HOME/.local/bin:$PATH"
flamingo/.venv/bin/hf auth whoami > /dev/null || { echo "log in first: hf auth login (google/gemma-3-270m is gated)"; exit 1; }
# OpenTSLM at the commit of the Mac clone (run_har.py imports its src/, writes nothing into it)
[ -d OpenTSLM/.git ] || git clone -q https://github.com/StanfordBDHG/OpenTSLM.git OpenTSLM
git -C OpenTSLM checkout -q 2968f4b891baab4307f7e9d0043e87677b593a30
# HAR-CoT test windows, from the URL in OpenTSLM's har_cot_loader.py (zip 140 MB, test CSV 56 MB)
mkdir -p data/har_cot
if [ ! -f data/har_cot/har_cot_test_cot.csv ]; then
  curl -fL -o data/har_cot/har_cot.zip https://polybox.ethz.ch/index.php/s/kD74GnMYxn3HBEM/download
  unzip -o -j data/har_cot/har_cot.zip har_cot_test_cot.csv -d data/har_cot && rm data/har_cot/har_cot.zip
fi
sha256sum data/har_cot/har_cot_test_cot.csv  # EXP's SPEC: c8da9bcc...
# The script's own environment (inline dependencies: torch >= 2.6, transformers < 4.58, open-flamingo 2.0.1), the
# checkpoint and Gemma 3 270M are cached by a 2-window run, so the queue can run offline
uv run usecases/explanation/har/run_har.py --csv data/har_cot/har_cot_test_cot.csv --opentslm OpenTSLM \
  --out runs/har_cot_setup --limit 2
rm -rf runs/har_cot_setup
echo "HAR setup done"
