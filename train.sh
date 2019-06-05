#!/usr/bin/env bash

pip install -r requirements.txt

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
