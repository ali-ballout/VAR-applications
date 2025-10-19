# shoplift-vision

**shoplift-vision** is a production-ready video action recognition toolkit focused on rapid prototyping and deployment of binary "shoplifting" vs "no_shoplifting" classifiers. It provides instant zero-shot baselines, a fine-tuning pipeline for state-of-the-art VideoMAE models, and a polished developer experience so you can adapt it to any action-recognition dataset with minimal effort.

## Key Features
- ⚡ **Instant zero-shot inference** using X-CLIP with natural-language prompts—no training required.
- 🎯 **Fine-tune VideoMAE** models from Hugging Face Transformers on your own dataset with a single command.
- 🧱 **Config-driven pipeline** with YAML configs for datasets, models, and hyperparameters.
- 🖥️ **Gradio demo app** supporting webcam and file uploads for quick stakeholder demos.
- 🧪 **Robust developer tooling**: Makefile, Dockerfile, CI workflow, tests, pre-commit hooks, and comprehensive docs.

## Hardware Requirements
- Works out of the box on CPU for demos and testing (training will be slower).
- GPU (CUDA-enabled) strongly recommended for fine-tuning and large-scale inference.
- Ensure FFmpeg is installed for video handling (Dockerfile and requirements cover this).

## Repository Layout
```
.
├── configs/                  # YAML configs for zero-shot and fine-tuning
├── data/                     # Expected dataset structure (see data/README.md)
├── scripts/                  # Utility scripts (dataset prep, evaluation)
├── src/shoplift_vision/      # Library source code
├── tests/                    # Pytest-based unit tests with synthetic data
├── requirements.txt          # Python dependencies (see note for GPU installations)
├── Makefile, Dockerfile      # Developer workflow automation
└── README.md                 # You are here
```

## Quickstart

### 1. Create & Activate a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> **Note:** The default requirements install the CPU wheels for PyTorch. If you have a CUDA-enabled GPU, follow the [official instructions](https://pytorch.org/get-started/locally/) to install the appropriate wheels before running the rest of the commands.

### 3. Prepare Your Dataset
Organize videos into the following folder structure (binary example shown):
```
data/
  train/
    shoplifting/
      clip001.mp4
    no_shoplifting/
      clip002.mp4
  val/
    shoplifting/
      clip101.mp4
    no_shoplifting/
      clip102.mp4
  test/
    shoplifting/
      clip201.mp4
    no_shoplifting/
      clip202.mp4
```
A minimal pseudo-example to verify the pipeline might look like:
```
data/
  train/{shoplifting,no_shoplifting}/tiny_clip.mp4
  val/{shoplifting,no_shoplifting}/tiny_clip.mp4
```
You can use `scripts/prepare_dataset.py` to convert CSV annotations into this layout or to auto-split a folder of videos.

### 4. Zero-Shot Inference (No Training Needed)
```bash
python src/shoplift_vision/infer_zeroshot.py \
  --video demo_videos/sample_checkout.mp4 \
  --config configs/zero_shot.yaml \
  --labels "shoplifting,no_shoplifting"
```
This samples frames from the video, encodes them with X-CLIP, compares against natural-language prompts, and prints class probabilities.

> **Offline sanity check:** If you're working in a network-isolated environment you can still validate the CPU pipeline with the
> lightweight fallback model and a tiny synthetic clip. Generate the clip once using ImageIO (installed with the default requirements):
> ```bash
> mkdir -p demo_videos
> python - <<'PY'
> import imageio.v2 as imageio
> import numpy as np
> frames = []
> for i in range(8):
>     base = np.zeros((240, 320, 3), dtype=np.uint8)
>     base[:, :, 1] = (i * 32) % 255  # green pulse
>     base[:, :, 2] = ((7 - i) * 32) % 255  # magenta pulse
>     frames.append(base)
> imageio.mimsave("demo_videos/dummy.gif", frames, fps=4)
> PY
> HF_HUB_OFFLINE=1 python src/shoplift_vision/infer_zeroshot.py \
>   --config configs/dummy_zero_shot.yaml \
>   --video demo_videos/dummy.gif
> ```
> The dummy model produces deterministic probabilities without downloading large checkpoints, which makes it ideal for quick integration tests.

### 5. Fine-Tune VideoMAE on Your Dataset
```bash
python src/shoplift_vision/train_videomae.py --config configs/shoplifting_videomae.yaml
```
This command will:
1. Load training/validation data according to the config.
2. Fine-tune a pretrained VideoMAE model using Hugging Face Transformers' `Trainer` API.
3. Save checkpoints, logs, and metrics under `runs/`.

After training, run inference with the best checkpoint:
```bash
python src/shoplift_vision/infer_finetuned.py \
  --video demo_videos/sample_entry.mp4 \
  --checkpoint runs/shoplifting_videomae/best.pt \
  --config configs/shoplifting_videomae.yaml
```

### 6. Launch the Gradio Demo
```bash
python src/app.py --mode zeroshot
# or
python src/app.py --mode finetuned --checkpoint runs/shoplifting_videomae/best.pt
```
This starts a local web interface where you can upload videos or use your webcam to run the selected inference mode.

## Adapting to Another Action Dataset (Retargeting Guide)
1. **Update dataset folders:** Place videos under `data/{split}/{class_name}/*.mp4` with your new class labels.
2. **Edit the config:** Update the `labels` list, prompts (for zero-shot), and tweak `num_frames`, `fps`, `clip_seconds`, or image size in the relevant YAML config.
3. **Adjust augmentations:** Modify transforms or batch size if your videos differ significantly in resolution or length.
4. **Run zero-shot or re-train:** For quick baselines, only edit `configs/zero_shot.yaml`. For best performance, fine-tune by updating `configs/<your_config>.yaml` and re-running `train_videomae.py`.
5. **Evaluate & iterate:** Use `scripts/eval_dir.py` to evaluate directories of videos and compare models.

## Configuration System
- `configs/base.yaml` contains global defaults (seed, device, paths, video parameters).
- Specialized configs override the base (e.g., `configs/shoplifting_videomae.yaml`).
- Pass any config to CLI scripts via `--config path/to/config.yaml`.
- All configs support overriding parameters through command-line flags using dot notation (e.g., `--override train.batch_size=8`).

## Logging, Reproducibility & Outputs
- Deterministic seeds are set for Python, NumPy, and PyTorch.
- Training logs and checkpoints are stored under `runs/<experiment_name>/`.
- Validation metrics include accuracy, macro F1, ROC-AUC, and confusion matrices.
- `infer_*.py` scripts output JSON files with per-class probabilities for reproducible scoring.

## Ethical Considerations & Deployment Notes
- **Privacy:** Surveillance footage often includes personally identifiable information. Ensure compliance with local laws and obtain appropriate consent.
- **Bias & Fairness:** Model performance can vary across demographics and store layouts. Perform thorough evaluations before deployment and maintain human oversight.
- **False Positives:** Shoplifting detection has high stakes. Calibrate decision thresholds carefully and consider multi-factor verification to avoid unjust accusations.
- **Data Security:** Protect training data and inference outputs, especially when handling sensitive footage.

## Troubleshooting FAQ
**Q: `decord` fails to import.**
A: Install system dependencies (`sudo apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6`) or set the environment variable `SHOPLIFT_VISION_USE_TORCHVISION_IO=1` to force the torchvision fallback.

**Q: Training is too slow or OOM on GPU.**
A: Reduce `train.batch_size`, `video.num_frames`, or use mixed precision by setting `train.fp16=true` in the config.

**Q: My dataset splits are imbalanced.**
A: Use `scripts/prepare_dataset.py --stratify` to stratify splits and monitor metrics like F1 score.

**Q: Can I export to ONNX or TorchScript?**
A: Use `python src/shoplift_vision/export.py --checkpoint runs/.../best.pt --format onnx` (TorchScript also supported). Some models may require opset adjustments; see the script for guidance.

**Q: How do I change the prompts for zero-shot inference?**
A: Edit `configs/zero_shot.yaml` to add multiple natural-language templates per label. The script averages prompt embeddings per class.

## Makefile Highlights
- `make setup`: Install dependencies.
- `make fmt`: Run formatting (black, isort) and lint (ruff).
- `make test`: Execute unit tests.
- `make train`: Launch fine-tuning with the default config.
- `make zeroshot`: Run zero-shot inference on the sample command.
- `make demo`: Start the Gradio UI.

## Contributing
Pull requests are welcome! Run `pre-commit install` to enable formatting and linting hooks. Tests must pass in CI (`.github/workflows/ci.yml`).

## License
This project is released under the [MIT License](LICENSE).

## Citation
If you use **shoplift-vision** in academic work, please cite the project using the metadata in `CITATION.cff`.

Happy building!

## How To: Obtain Demo Videos and Public Datasets

### How To: Download Sample Demo Clips
1. Create a folder to store example clips:
   ```bash
   mkdir -p demo_videos
   ```
2. Download two royalty-free clips that work well with the commands in this README:
   ```bash
   curl -L -o demo_videos/sample_checkout.mp4 \
    https://samplelib.com/lib/preview/mp4/sample-5s.mp4
   curl -L -o demo_videos/sample_entry.mp4 \
    https://samplelib.com/lib/preview/mp4/sample-10s.mp4
   ```
3. Re-run the zero-shot and fine-tuned inference commands using these files. The sample clips are generic retail scenes, making them convenient stand-ins for quick demos.

### How To: Download Public Action-Recognition Datasets
1. **UCF101** (action classification, 13K videos):
   - Request access and download the archive from the official dataset page: <https://www.crcv.ucf.edu/data/UCF101.php>
   - Extract the videos and use `scripts/prepare_dataset.py --ucf101` to convert them into the `data/{split}/{class}` layout.
2. **HMDB51** (human motion recognition, 7K videos):
   - Download the dataset from <https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>
   - Use the provided train/test splits or generate your own with `scripts/prepare_dataset.py --hmdb51` before fine-tuning.
3. **Kinetics-400** (large-scale action dataset):
   - Obtain the metadata CSV and download links from <https://deepmind.google-research.google/video-understanding/kinetics>
   - Filter to the retail-relevant classes, then feed the downloaded videos through `scripts/prepare_dataset.py --from-csv path/to/csv` to create curated training splits.

Each dataset URL requires accepting the dataset-specific license terms. After downloading, point the configs to the new class labels or prompts before launching training or inference.
