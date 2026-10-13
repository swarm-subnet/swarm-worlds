#!/usr/bin/env bash
# Prepare a fresh CUDA machine to run generate.py: Kimodo at a pinned commit, its weights, and the Llama 3 text model
# its text encoder is built on (an ungated copy of the same Meta weights, used under the Meta Llama 3 licence).
set -euo pipefail

KIMODO_COMMIT=58e781898b3d7e328a676a75d3e338c45dce3ad9
TE=${TEXT_ENCODERS_DIR:-/root/text_encoders}
MNTP=McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
LLAMA=$TE/meta-llama/Meta-Llama-3-8B-Instruct

pip install -q "git+https://github.com/nv-tlabs/kimodo.git@${KIMODO_COMMIT}" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download nvidia/Kimodo-SOMA-RP-v1.1 > /dev/null
hf download nvidia/TMR-SOMA-RP-v1 > /dev/null
hf download "$MNTP" --local-dir "$TE/$MNTP" > /dev/null
hf download "$MNTP-supervised" --local-dir "$TE/$MNTP-supervised" > /dev/null
hf download NousResearch/Meta-Llama-3-8B-Instruct --local-dir "$LLAMA" --exclude "original/*" > /dev/null

# The LLM2Vec adapter names Meta's gated repository as its base; point it at the local copy instead.
python3 - "$TE/$MNTP/adapter_config.json" "$LLAMA" <<'EOF'
import json, sys
path, base = sys.argv[1], sys.argv[2]
config = json.load(open(path))
config["base_model_name_or_path"] = base
json.dump(config, open(path, "w"), indent=2)
EOF

echo "export TEXT_ENCODERS_DIR=$TE TEXT_ENCODER_MODE=local"
