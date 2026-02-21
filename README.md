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

# Datasets
## [ARC](https://huggingface.co/datasets/allenai/ai2_arc)
## [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa)
## [OpenbookQA](https://huggingface.co/datasets/allenai/openbookqa)
## [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag)
## [Swag](https://huggingface.co/datasets/allenai/swag)
## [MMLUPro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
## [GLUE](https://huggingface.co/datasets/nyu-mll/glue/)

# Training
1. To switch between datasets, change `dataset` and `task` in [run.sh](run.sh). For mrpc and rte use "glue" for dataset and same as task otherwise, e.g. "arc_c", "mmlu_pro", "hellaswag".
2. To switch between different models, change `model` in [run.sh](run.sh). Right now we only support llama3.1_8b, llama3.2_1b, llama3.2_3b, qwen3_0.6b, qwen3_1.7b.

To run the code, please use the following cmd
```
bash run.sh
```