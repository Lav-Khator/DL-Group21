# DL-Group21

Spatio-Temporal Graph Neural Network pipeline for the Ego4D **Talking-To-Me (TTM)** task.

## Repository contents

- `main.py` — CLI entry point for preprocessing, training, evaluation, visualization, and EDA
- `preprocess.py` — feature extraction and preprocessing
- `train.py` — training and evaluation logic
- `dataset.py` — dataset loading utilities
- `ttm_model.py` — model definition
- `visualize.py` — plotting and analysis helpers
- `config.py` — paths and hyperparameters

## Setup

1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data layout

By default, `config.py` expects an Ego4D-like structure:

- `.../annotations`
- `.../clips`
- `.../full_scale`

You can override paths at runtime with `--data_root`.

## Usage

### 1) Preprocess features

```bash
python main.py preprocess --split both --mode full
```

If video files are unavailable, use metadata-only mode:

```bash
python main.py preprocess --split both --mode lite
```

### 2) Train

```bash
python main.py train --mode full
```

Or train directly from JSON annotations (lite mode) :

```bash
python main.py train --direct --mode lite
```

### 3) Evaluate

```bash
python main.py evaluate --checkpoint ./checkpoints/best_model.pt
```

### 4) Visualize

```bash
python main.py visualize --history ./checkpoints/training_history.json
```

### 5) EDA

```bash
python main.py eda
```

## Full pipeline

```bash
python main.py full --mode lite
```
