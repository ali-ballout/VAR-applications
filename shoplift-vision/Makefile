.PHONY: setup fmt lint test train infer zeroshot demo docker-build docker-run clean

PYTHON ?= python
PIP ?= pip
CONFIG ?= configs/shoplifting_videomae.yaml
VIDEO ?= demo_videos/sample.mp4
CHECKPOINT ?= runs/shoplifting_videomae/best.pt

setup:
$(PIP) install --upgrade pip
$(PIP) install -r requirements.txt

fmt:
ruff check src tests --fix
isort src tests
black src tests

lint:
ruff check src tests
black --check src tests
isort --check-only src tests

test:
PYTHONPATH=src $(PYTHON) -m pytest -q --disable-warnings

train:
PYTHONPATH=src $(PYTHON) src/shoplift_vision/train_videomae.py --config $(CONFIG)

infer:
PYTHONPATH=src $(PYTHON) src/shoplift_vision/infer_finetuned.py --video $(VIDEO) --checkpoint $(CHECKPOINT) --config $(CONFIG)

zeroshot:
PYTHONPATH=src $(PYTHON) src/shoplift_vision/infer_zeroshot.py --video $(VIDEO) --config configs/zero_shot.yaml --labels "shoplifting,no_shoplifting"

demo:
PYTHONPATH=src $(PYTHON) src/app.py --mode zeroshot

docker-build:
docker build -t shoplift-vision:latest .

docker-run:
docker run --rm -it -p 7860:7860 shoplift-vision:latest

clean:
rm -rf __pycache__ .pytest_cache build dist
