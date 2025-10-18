# Evaluation Summary

## Environment Setup
1. **System dependencies**
   - Update package index: `sudo apt-get update`
   - Install OpenCV runtime libraries: `sudo apt-get install -y libgl1`
2. **Python dependencies** (executed from `shoplift-vision/`)
   ```bash
   pip install numpy==1.26.4 opencv-python pytest-cov torch==2.2.0 torchvision==0.17.1 --no-deps
   pip install pyyaml pillow transformers av
   ```
   > **Notes**
   > - `torchvision==0.17.1` is installed with `--no-deps` to reuse the already installed `torch==2.2.0` wheel.
   > - Downgrading to `numpy==1.26.4` avoids PyTorch/torchvision runtime errors encountered with NumPy 2.x.
   > - Additional libraries (`av`, `pillow`, `transformers`) are required to satisfy runtime imports used by the tests.

## Test Commands
All commands executed from the `shoplift-vision/` directory.

| Command | Purpose |
| --- | --- |
| `PYTHONPATH=src python -m pytest -q --disable-warnings` | Runs the complete unit test suite (`tests/`). |

## Results
- Pytest completed successfully with **7 tests passing** in ~4.3 seconds on CPU-only hardware. Full output:
  ```
  PYTHONPATH=src python -m pytest -q --disable-warnings
  .......                                                                                                                  [100%]
  7 passed in 4.30s
  ```

## Deep Analysis
- **Dependency Resolution:**
  - Initial attempts using the project's `requirements.txt` pulled large CUDA-enabled wheels; installing the lighter CPU builds (`torch==2.2.0`, `torchvision==0.17.1 --no-deps`) significantly reduced download overhead while remaining compatible with Python 3.12.
  - OpenCV required the `libGL` system library; without it, `cv2` failed to import. Installing `libgl1` resolved this issue.
  - Torchvision's video IO backend needs PyAV. Installing `av` enabled the fallback path that reads MP4 fixtures in the tests.
  - NumPy 2.x triggered runtime warnings and PyTorch dtype inference errors; pinning `numpy==1.26.4` restored compatibility for both torch and torchvision.
- **Test Coverage Insights:**
  - `tests/test_dataset.py` and `tests/test_inference.py` generate synthetic MP4 clips to exercise dataset loading, frame sampling, and inference helper utilities. This verifies OpenCV writer interoperability with the `load_video_frames` abstraction.
  - `tests/test_sampling.py` focuses on deterministic frame index selection logic for temporal sampling strategies.
  - The suite validates both CPU fallback logic in `_resolve_device` and probability normalization in `_run_inference`, ensuring inference utilities behave predictably without GPU access.
- **Dataset Considerations:**
  - No external shoplifting dataset was required: fixtures synthesize short random videos sufficient to validate the preprocessing and inference stack end-to-end. This keeps evaluation lightweight while still confirming codec compatibility and model integration points.

## Reproducibility Checklist
- Ensure Python 3.12 is available.
- Execute the environment setup steps above in order (system packages before Python wheels).
- Run the pytest command with `PYTHONPATH=src` to make the project modules discoverable.
- Expect the same 7 passing tests given deterministic seeds and synthetic fixtures.
