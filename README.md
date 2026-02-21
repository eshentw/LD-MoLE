# LD-MoLE
<img src="imgs/framework.png" alt="framework" width="720" />
LD-MoLE, a learnable dynamic routing framework for Mixture of LoRA Experts that replaces conventional non-differentiable Top-K routing with a Sparsegen-based differentiable routing mechanism. The approach employs a shared MLP to predict token-specific and layer-wise sparsity parameters (λ), enabling adaptive and flexible expert allocation. Additionally, an analytical sparsity control objective is incorporated to regularize expert activation and encourage efficient expert utilization.

# Get Start

# Setting Up a uv Environment (Python 3.9)

This guide explains how to create a Python 3.9 virtual environment using `uv` and install dependencies from `requirements.txt`.

---

## 1. Install uv

If you do not already have `uv` installed:

### macOS / Linux
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```
or via pip
```
pip install uv
```

## 2. Create a Python 3.9 Virtual Environment

```
uv venv --python 3.9
```

## 3. Activate the Virtual Environment

```
source .venv/bin/activate
```

## 4. Install Dependencies

```
uv pip install -r requirements.txt
```

## 5 Training

```
bash run.sh
```