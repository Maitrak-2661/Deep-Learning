# PR-5: Sentiment Analysis on IMDB Reviews using RNN, LSTM and GRU

A Deep Learning practical that compares an ANN baseline against SimpleRNN, LSTM, Stacked LSTM and GRU models for binary sentiment classification on the IMDB 50K Movie Reviews dataset.

## Project Overview

The notebook loads the raw review CSV and does its own text cleaning, tokenization and padding. It does not use the pre-tokenized Keras IMDB dataset. It then trains five models on the same data split and compares them on accuracy, precision, recall, F1 and ROC-AUC.

## Repository Structure

```
PR-5/
├── DL_PR5.ipynb               # Main notebook (all tasks)
├── IMDB Dataset.csv           # Raw dataset (50,000 reviews)
├── pr5_model_comparison.csv   # Final metrics for all models
├── requirements.txt           # Python dependencies
└── README.md
```

## Dataset

- **Source:** Kaggle, IMDB Dataset of 50K Movie Reviews
- **Columns:** `review` (text), `sentiment` (`positive` / `negative`)
- **Raw size:** 50,000 reviews (25,000 positive, 25,000 negative)
- **After cleaning:** 418 duplicate rows removed, leaving 49,582 reviews (24,884 positive, 24,698 negative)

> If the dataset is too large for your GitHub repo, download it from Kaggle and place it next to the notebook.

### Review length analysis

Reviews average about 232 words (median 173). The 95th percentile is about 591 words and the 99th about 907. `MAX_LEN = 250` was chosen as a practical compromise between keeping context and keeping training fast, so longer reviews are truncated.

## Methodology

**Preprocessing**
- Drop null and duplicate rows (done before the split)
- Unescape HTML, strip tags, lowercase and normalize whitespace
- Keep negation words such as "not", since they change sentiment

**Tokenization and padding**
- Vocabulary capped at 10,000 words, with an `<OOV>` token for unknown words
- `MAX_LEN = 250` with pre-padding and pre-truncation
- Split: 20% test (stratified, `random_state=42`), then 20% of the remaining data for validation

**Hyperparameters**

| Setting | Value |
|---|---|
| Embedding dimension | 32 |
| Recurrent units | 64 |
| Batch size | 128 |
| Max epochs | 8 (EarlyStopping on `val_loss`, patience 2) |
| Optimizer / Loss | Adam / Binary cross-entropy |
| Dropout | 0.3 |
| Seed | 42 |

**Model architectures**

| Model | Architecture |
|---|---|
| ANN baseline | Embedding, Flatten, Dense(64, ReLU), Dropout, Dense(1, sigmoid) |
| SimpleRNN | Embedding, SimpleRNN(64), Dropout, Dense(1, sigmoid) |
| LSTM | Embedding, LSTM(64), Dropout, Dense(1, sigmoid) |
| Stacked LSTM | Embedding, LSTM(64, return_sequences), LSTM(32), Dropout, Dense(1, sigmoid) |
| GRU | Embedding, GRU(64), Dropout, Dense(1, sigmoid) |

## Notebook Contents

| Task | Topic |
|---|---|
| 1 | Data loading, cleaning, length analysis, tokenization and padding |
| 2 | ANN vs RNN, and a manual SimpleRNN forward step checked against Keras |
| 3 | RNN types, and SimpleRNN sentiment classifier |
| 4 | Backpropagation Through Time, gradient-flow probe, vanishing/exploding gradients |
| 5 | LSTM theory, parameter count, and Stacked LSTM |
| 6 | GRU theory, parameter count, and hidden-unit sensitivity experiment |
| 7 | Final comparison, ROC curves, error analysis, inference function and recommendation |

The notebook also produces training/validation curves, gradient-flow plots, ROC curves and confusion matrices when you run it.

## Results

Test-set results for each model (from `pr5_model_comparison.csv`):

| Model | Params | Time/Epoch (s) | Epochs | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| LSTM | 344,897 | 28.69 | 4 | **0.8856** | **0.9015** | 0.8666 | 0.8837 | **0.9538** |
| Stacked LSTM | 357,281 | 52.18 | 3 | 0.8833 | 0.8791 | 0.8899 | 0.8845 | 0.9517 |
| ANN baseline | 832,129 | 1.68 | 3 | 0.8823 | 0.8696 | **0.9005** | **0.8848** | 0.9511 |
| GRU | 338,881 | 51.34 | 3 | 0.8636 | 0.8575 | 0.8732 | 0.8653 | 0.9379 |
| SimpleRNN | 326,273 | 8.55 | 5 | 0.8453 | 0.8451 | 0.8471 | 0.8461 | 0.9132 |

**Key observations**
- **LSTM** has the best accuracy, precision and ROC-AUC, so it is the best overall model here.
- The **ANN baseline** is very competitive and about 17x faster per epoch, but it has more than double the parameters because its dense layer scales with `MAX_LEN`.
- **Stacked LSTM** adds parameters and training time without beating the single LSTM.
- **GRU** and **SimpleRNN** trail the others in this run. SimpleRNN is the weakest, which fits its difficulty with long-range dependencies.

Exact numbers may vary slightly between runs and hardware.

## Setup and Usage

1. **Clone the repo**
   ```bash
   git clone <your-repo-url>
   cd PR-5
   ```

2. **Install dependencies** (Python 3.9+ recommended)
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook DL_PR5.ipynb
   ```
   Run all cells. A GPU is recommended, since the recurrent models take a while on CPU.

   For a quick test, set `MAX_ROWS = 10000` in the data loading cell.

### Inference example

After training, use the `predict_sentiment` function defined in the notebook:

```python
predict_sentiment("This movie was wonderful and the acting was excellent.")
# [{'text': '...', 'predicted_sentiment': 'positive', 'positive_probability': 0.9...}]
```

## Requirements

```
tensorflow>=2.15
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.3
jupyter>=1.0
nbformat>=5.9
```

## Demo Video

[Watch the walkthrough here](ADD_YOUR_VIDEO_LINK)

## Author

**Maitrak**
B.Tech, AI & Machine Learning
