#!/usr/bin/env bash
# Builds lambda.zip for Python 3.12 on x86_64 Lambda. Run from the rag-platform folder.
# Needs internet (pip + one-time model download from Hugging Face).
set -euo pipefail

rm -rf build lambda.zip
mkdir -p build/models

# Lambda-compatible wheels, even when building on Windows or macOS
pip install -r requirements.txt -t build \
  --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
  --python-version 3.12 --implementation cp --only-binary=:all: --upgrade

cp src/*.py build/

# Bundle all-MiniLM-L6-v2 (quantized ONNX, about 23 MB) so Lambda never downloads anything
pip install -q huggingface_hub
python - <<'EOF'
import shutil
from huggingface_hub import hf_hub_download

repo = "sentence-transformers/all-MiniLM-L6-v2"
shutil.copy(hf_hub_download(repo, "onnx/model_qint8_avx2.onnx"), "build/models/model.onnx")
shutil.copy(hf_hub_download(repo, "tokenizer.json"), "build/models/tokenizer.json")
EOF

find build -name "__pycache__" -type d -prune -exec rm -rf {} +
find build -type d -name "tests" -prune -exec rm -rf {} + 2>/dev/null || true

(cd build && zip -qr ../lambda.zip .)

echo "unzipped size in MB (Lambda limit is 250):"
du -sm build | cut -f1
echo "zip size in MB:"
du -sm lambda.zip | cut -f1
