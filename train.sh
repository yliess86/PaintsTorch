#!/usr/bin/env bash

pip install -r requirements.txt

# ================= TRAIN =====================
python3 paintstorch \
  -s 2333 \
  -c 64 \
  -x '/Projects/PaintsTorchExp/paper_random_simple' \
  -t '/Projects/PaintsTorchDataset/paper/colored' \
  -v '/Projects/PaintsTorchDataset/paper/lineart' \
  -b 32 \
  -e 100 \
  -m random \
  # -d

python3 paintstorch \
  -s 2333 \
  -c 64 \
  -x '/Projects/PaintsTorchExp/paper_strokes_simple' \
  -t '/Projects/PaintsTorchDataset/paper/colored' \
  -v '/Projects/PaintsTorchDataset/paper/lineart' \
  -b 32 \
  -e 100 \
  -m strokes \
  # -d

python3 paintstorch \
  -s 2333 \
  -c 64 \
  -x '/Projects/PaintsTorchExp/paper_strokes_double' \
  -t '/Projects/PaintsTorchDataset/paper/colored' \
  -v '/Projects/PaintsTorchDataset/paper/lineart' \
  -b 32 \
  -e 100 \
  -m strokes \
  -d

python3 paintstorch \
  -s 2333 \
  -c 64 \
  -x '/Projects/PaintsTorchExp/custom_strokes_double' \
  -t '/Projects/PaintsTorchDataset/custom' \
  -v '/Projects/PaintsTorchDataset/paper/lineart' \
  -b 32 \
  -e 100 \
  -m strokes \
  -d

python3 paintstorch \
  -s 2333 \
  -c 64 \
  -x '/Projects/PaintsTorchExp/custom_only_strokes_double' \
  -t '/Projects/PaintsTorchDataset/custom_only' \
  -v '/Projects/PaintsTorchDataset/paper/lineart' \
  -b 32 \
  -e 100 \
  -m strokes \
  -d

# ============== EVALUATE =======================
python3 paintstorch \
  -f \
  -s 2333 \
  -c 64 \
  -b 32 \
  -x '/Projects/PaintsTorchExp/paper_random_simple' \
  -v '/Projects/PaintsTorchDataset/paper/lineart'

python3 paintstorch \
  -f \
  -s 2333 \
  -c 64 \
  -b 32 \
  -x '/Projects/PaintsTorchExp/paper_strokes_simple' \
  -v '/Projects/PaintsTorchDataset/paper/lineart'

python3 paintstorch \
  -f \
  -s 2333 \
  -c 64 \
  -b 32 \
  -x '/Projects/PaintsTorchExp/paper_strokes_double' \
  -v '/Projects/PaintsTorchDataset/paper/lineart'

python3 paintstorch \
  -f \
  -s 2333 \
  -c 64 \
  -b 32 \
  -x '/Projects/PaintsTorchExp/custom_strokes_double' \
  -v '/Projects/PaintsTorchDataset/paper/lineart'

python3 paintstorch \
  -f \
  -s 2333 \
  -c 64 \
  -b 32 \
  -x '/Projects/PaintsTorchExp/custom_only_strokes_double' \
  -v '/Projects/PaintsTorchDataset/paper/lineart'
