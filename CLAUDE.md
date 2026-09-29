# Project: Customer Review Sentiment Analysis Using LSTM for Online Shopping

## Goal
3-class sentiment classifier (negative / neutral / positive) on Amazon Reviews 2023
(McAuley-Lab/Amazon-Reviews-2023, Electronics category, ~300k sampled reviews).
Labels: 1-2 stars = negative, 3 = neutral, 4-5 = positive.
This is a portfolio project for GitHub and a resume, so quality and clarity matter.

## Tech stack
- Python, PyTorch (CUDA), Hugging Face datasets/transformers, scikit-learn
- Streamlit for the demo app
- Hardware: RTX 4050 (6 GB VRAM). Use mixed precision (fp16), small batch sizes, max_len 128 for DistilBERT.

## Models to build
1. Baseline: TF-IDF + Logistic Regression
2. Plain LSTM
3. Bi-GRU
4. Main model: Bi-LSTM + attention with pretrained GloVe embeddings
5. Upper bound: fine-tuned DistilBERT

## Evaluation rules
- Data is imbalanced (neutral is small). Use class weights.
- Report macro-F1, per-class precision/recall/F1, and confusion matrix. Not just accuracy.
- Fixed random seed (42). Stratified 80/10/10 train/val/test split.
- Never tune on the test set.

## Repo structure
src/ (data.py, models.py, train.py, evaluate.py), notebooks/, app/, results/, data/ (gitignored)

## Working rules
- Write modular, commented code with type hints.
- After each phase, explain in simple words what you built and why, so I can explain it in interviews.
- Save all metrics and plots to results/.
- Ask me before installing big dependencies or downloading large files.
- Commit after each phase with a clear message.