# Meta Pseudo Labels with TensorFlow

Unofficial implementation of [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580).
Currently the project only support for images clasification task.

## 1. Setup Environment

```bash
# 1. Create a dedicated virtual environment (Python ≥ 3.10)
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip & install dependencies
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 2. Quick Start
```
mpl-tf train --config configs/mnist.yml
```

or

```
./scripts/train.sh configs/mnist.yml
```