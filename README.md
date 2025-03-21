# PDR-II Drug Repurposing Framework

This repository contains a personalized drug repurposing framework that leverages knowledge graphs and patient-specific data to recommend drugs for target diseases.

## Overview

PDR-II enables both general knowledge graph-based drug repurposing and patient-specific fine-tuning to personalize drug recommendations. The framework uses a foundation model that can be further tailored with patient-specific loss functions incorporating genetic risk scores and biomarker data.

## Prerequisites

- Python 3.7+
- PyTorch
- PyTorch Geometric
- CUDA-capable GPU (recommended)

## Usage

The main script for running the framework is `script/run.py`, which accepts various configuration options.

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `-c`, `--config` | Path to YAML configuration file (required) |
| `-s`, `--seed` | Random seed for PyTorch (default: 1024) |
| `-pid`, `--patient_id` | Patient ID for personalized recommendations, just for indexing and outputting use any id you want |
| `--use_ploss` | Flag to enable patient-specific loss during fine-tuning |
| `-l1`, `--lambda1_prs` | Lambda value for PRS influence in patient-specific loss |
| `-l2`, `--lambda2_biomarker` | Lambda value for biomarker expression influence in patient-specific loss |
| `--dataset` | Dataset name to use |
| `--epochs` | Number of training epochs |
| `--bpe` | Batches per epoch |
| `--gpus` | List of GPU devices to use, e.g., [0] or [0,1] |
| `--ckpt` | Path to checkpoint file for model initialization |

### Example Commands

#### Foundation Model Training

Train the foundation model on general knowledge graph data:

```bash
python script/run.py -c config/transductive/training.yaml --dataset KG_GENERAL --epochs 100 --bpe 500 --gpus [0]
```

#### Patient-Specific Fine-tuning (without PLoss)

Fine-tune the foundation model for a specific patient:

```bash
python script/run.py -c config/transductive/inference.yaml --patient_id {patient_id} --dataset KG_ROBIN --epochs 3 --bpe 100 --gpus [0] --ckpt /path/to/foundation_model_checkpoint.pt
```

#### Patient-Specific Fine-tuning (with PLoss)

Fine-tune with patient-specific loss to incorporate genetic and biomarker data:

```bash
python script/run.py -c config/transductive/inference.yaml --patient_id {patient_id} --dataset KG_ZORO --epochs 3 --bpe 100 --gpus [0] --use_ploss -l1 0.5 -l2 0.1 --ckpt /path/to/foundation_model_checkpoint.pt
```

### Running in Background

To run the process in the background and save output to a log file:

```bash
nohup python script/run.py -c config/transductive/inference.yaml --patient_id {patient_id} --dataset KG_ZORO --epochs 3 --bpe 100 --gpus [0] --use_ploss -l1 0.5 -l2 0.1 --ckpt /path/to/best_model_state_dict.pt > zoro_out_ploss_0_5_0_1.txt 2>&1 &
```

## Configuration Files

The framework uses YAML configuration files located in the `config/` directory:
- `config/transductive/training.yaml`: Configuration for training the foundation model
- `config/transductive/inference.yaml`: Configuration for inference and fine-tuning

## Output

The model will:
1. Load or train the specified model
2. Perform drug repurposing predictions
3. Evaluate using ranking metrics (MRR, Hits@K, etc.)
4. For personalized models, rank drugs specifically for the target disease

Results are saved to the specified output directory and logged to the console or log file.

Reach out for any questions: cxo147@case.edu 

## Publication

💊 The predecessor of this work (a.k.a. PDR I) has been published in the Journal of Biomedical Informatics (JBI):
[Precision Drug Repurposing (PDR): Patient-level modeling and prediction combining foundational knowledge graph with biobank data](https://pubmed.ncbi.nlm.nih.gov/39952626/)
