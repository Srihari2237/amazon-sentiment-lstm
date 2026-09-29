### Model comparison (test split, 30,000 reviews)

| Model | Accuracy | **Macro-F1** | Weighted-F1 | F1 negative | F1 neutral | F1 positive | Params |
|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.8697 | 0.7196 | 0.8823 | 0.7753 | 0.4421 | 0.9412 | - |
| LSTM | 0.8688 | 0.7282 | 0.8845 | 0.7996 | 0.4457 | 0.9394 | 4,599,447 |
| Bi-GRU | 0.8916 | 0.7506 | 0.9006 | 0.8103 | 0.4874 | 0.9541 | 4,658,711 |
| Bi-LSTM + Attention | 0.8834 | 0.7435 | 0.8958 | 0.8016 | 0.4786 | 0.9504 | 4,750,615 |
| **DistilBERT (fine-tuned)** | 0.9086 | **0.7661** | 0.9121 | 0.8327 | 0.5022 | 0.9634 | 66,955,779 |

Always predicting the majority class scores **0.7914 accuracy** but only **0.2945 macro-F1** - which is why macro-F1 is the headline metric.

**Training conditions.** All models are evaluated on the same full 30,000-row validation and test splits, and all use the same class-weighted loss and early stopping on validation macro-F1.
DistilBERT was fine-tuned on the full 240,000-row training split at `max_len` 256 (batch 16, fp16), so it is directly comparable with the recurrent models.
