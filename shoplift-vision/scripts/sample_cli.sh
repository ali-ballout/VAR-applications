#!/usr/bin/env bash
set -euo pipefail

# Zero-shot single video
python src/shoplift_vision/infer_zeroshot.py --video demo_videos/shop1.mp4 --config configs/zero_shot.yaml

# Fine-tune VideoMAE
python src/shoplift_vision/train_videomae.py --config configs/shoplifting_videomae.yaml

# Inference with finetuned weights
python src/shoplift_vision/infer_finetuned.py --video demo_videos/shop2.mp4 --checkpoint runs/shoplifting_videomae/best.pt --config configs/shoplifting_videomae.yaml

# Launch Gradio app (zero-shot)
python src/app.py --mode zeroshot
