#!/usr/bin/env bash
#
# One-command GPU training for MobilePortrait avatar model.
#
# Usage on a rented GPU instance (RunPod, Lambda, Vast.ai, etc.):
#
#   1. Upload the avatar-trainer/ folder to the instance
#   2. Run: bash train_gpu.sh
#
# Or with custom settings:
#   bash train_gpu.sh --max-videos 200 --epochs 150 --batch-size 16
#
# Requirements: NVIDIA GPU with CUDA, ~20 GB disk space
#
set -euo pipefail

MAX_VIDEOS="${MAX_VIDEOS:-300}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-4}"
DATA_DIR="./data/hdtf"
CHECKPOINT_DIR="./checkpoints"
EXPORT_DIR="./exported"

while [[ $# -gt 0 ]]; do
  case $1 in
    --max-videos) MAX_VIDEOS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "============================================"
echo "  MobilePortrait GPU Training"
echo "============================================"
echo "  Videos:     $MAX_VIDEOS"
echo "  Epochs:     $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  LR:         $LR"
echo "  Data:       $DATA_DIR"
echo "============================================"
echo ""

# --- Step 1: Install dependencies ---
echo "[1/5] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
  torch torchvision torchaudio \
  onnx onnxruntime \
  opencv-python-headless \
  Pillow \
  tensorboard \
  tqdm \
  yt-dlp

echo "  Done. PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"
echo ""

# --- Step 2: Download HDTF ---
if [[ -z "${SKIP_DOWNLOAD:-}" ]]; then
  echo "[2/5] Downloading HDTF dataset from HuggingFace..."
  python download_hdtf_hf.py --output-dir "$DATA_DIR"
  echo ""
else
  echo "[2/5] Skipping download (--skip-download)"
  echo ""
fi

# --- Step 3: Verify data ---
echo "[3/5] Verifying dataset..."
FRAME_COUNT=$(find "$DATA_DIR" -name "*.jpg" | wc -l | tr -d ' ')
VIDEO_COUNT=$(find "$DATA_DIR" -maxdepth 1 -type d ! -name "_*" ! -name "." | wc -l | tr -d ' ')
echo "  Videos: $VIDEO_COUNT"
echo "  Frames: $FRAME_COUNT"
if [[ "$FRAME_COUNT" -lt 1000 ]]; then
  echo "  WARNING: Very few frames. Consider increasing --max-videos."
fi
echo ""

# --- Step 4: Train ---
echo "[4/5] Starting training..."
echo "  Epochs: $EPOCHS | Batch: $BATCH_SIZE | LR: $LR | Device: cuda"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  TensorBoard: tensorboard --logdir ./runs"
echo ""

python -u train.py \
  --data-root "$DATA_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --device cuda \
  --num-workers 4 \
  --save-every 10 \
  --checkpoint-dir "$CHECKPOINT_DIR"

echo ""

# --- Step 5: Export to ONNX ---
LAST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_*.pt 2>/dev/null | head -1)
if [[ -n "$LAST_CHECKPOINT" ]]; then
  echo "[5/5] Exporting to ONNX..."
  python export_onnx.py --checkpoint "$LAST_CHECKPOINT" --output-dir "$EXPORT_DIR"
  echo ""
  echo "============================================"
  echo "  Training Complete!"
  echo "============================================"
  echo ""
  echo "  ONNX models: $EXPORT_DIR/"
  ls -lh "$EXPORT_DIR"/*.onnx
  echo ""
  echo "  Copy these to your project:"
  echo "    scp -r $EXPORT_DIR/ your-machine:Avatrian/demo/public/models/mobileportrait/"
  echo ""
  echo "  Or download from this instance and place in:"
  echo "    demo/public/models/mobileportrait/"
  echo ""
else
  echo "[5/5] No checkpoint found, skipping export."
fi
