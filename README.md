# Customer Review Sentiment Analysis Using LSTM for Online Shopping

3-class sentiment classification (negative / neutral / positive) on the
[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
Electronics category, comparing a classical baseline, three recurrent
architectures, and a fine-tuned transformer on identical data splits.

> **Status:** Phase 1 complete (data pipeline). Modelling phases in progress -
> this README is expanded with the results table, plots and screenshots as they land.

## Labels

| Stars | Class    | Code |
|-------|----------|------|
| 1-2   | negative | 0    |
| 3     | neutral  | 1    |
| 4-5   | positive | 2    |

## Setup

```bash
python -m venv venv
venv\Scripts\activate                 # Windows;  source venv/bin/activate on Linux/macOS

pip install -r requirements.txt
```

PyTorch is installed separately, because the CUDA build does not come from PyPI
(installing it via `requirements.txt` would quietly give you the CPU-only wheel):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124 --resume-retries 10
```

The wheel is ~2.5 GB. If the download keeps dropping, grab it with a download
manager and install from the file instead:

```
https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl
```

```bash
pip install "C:\path\to\torch-2.6.0+cu124-cp312-cp312-win_amd64.whl"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Only `torch` is needed - `torchvision` and `torchaudio` are not used by this
project. **Phase 1 (the data pipeline) does not require torch at all**, so you
can build the dataset before PyTorch is installed.

## Build the dataset

```bash
python -m src.data --smoke --n-rows 2000   # ~1 min sanity check
python -m src.data                         # full 300k sample + splits
```

The Electronics file is 22.6 GB, so it is streamed from the Hub rather than
downloaded. The sampled rows are cached to `data/raw/`, and the stratified
80/10/10 splits are written to `data/processed/`. Both are gitignored.

## Interactive demo

Type in your own review and get a live prediction with class probabilities and
the words the attention layer focused on:

```bash
python -m src.predict            # interactive terminal mode
streamlit run app/app.py         # browser UI
```

Both reuse the exact cleaning functions from `src/data.py`, so a review typed by
hand is preprocessed identically to the training data.

## Project structure

```
src/         data.py (pipeline), models.py, train.py, evaluate.py, predict.py
notebooks/   exploratory analysis
app/         Streamlit demo
results/     metrics, plots, comparison tables (tracked in git)
data/        streamed sample and splits (gitignored)
```

## Models

1. TF-IDF + Logistic Regression (baseline)
2. Plain LSTM
3. Bi-GRU
4. Bi-LSTM + attention with pretrained GloVe embeddings (main model)
5. Fine-tuned DistilBERT (upper bound)

All models are trained with class-weighted loss and reported with macro-F1,
per-class precision/recall/F1 and a confusion matrix - accuracy alone is
misleading on this distribution. Seed is fixed at 42 throughout.
